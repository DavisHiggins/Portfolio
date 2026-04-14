import json
import base64
import os
import uuid
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import math
import inspect
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from streamlit.components.v1 import html
import streamlit.components.v1 as components
from supabase import Client, create_client

from themodel import analyze_prop
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonallplayers, scoreboardv2

st.set_page_config(page_title="Propify", layout="wide")


PRIMARY_EXPLANATIONS = {
    "Projection": "*model's expected result for this line*",
    "Over %": "*estimated chance the player clears the line*",
    "Under %": "*estimated chance the player stays under the line*",
    "Push %": "*estimated chance the player lands exactly on the line*",
    "Lean": "*the side the model currently prefers*",
    "Confidence": "*how strong the model views the play*",
    "EV %": "*value estimate based on risk and payout*",
    "Hit Probability": "*estimated chance the full entry hits*",
    "Miss Probability": "*estimated chance the full entry misses*",
    "Entry Type": "*the size of the entry being analyzed*",
}

SECTION_EXPLANATIONS = {
    "Key Statistics": "*main numbers the model used most*",
    "Risk Flags": "*simple caution notes worth knowing*",
    "Recent Sample": "*recent game log against the analyzed line*",
}


SECONDARY_EXPLANATIONS = {
    "Median": "*middle expected outcome in the model range*",
    "Floor (20th)": "*lower-end expected outcome*",
    "Ceiling (80th)": "*upper-end expected outcome*",
    "Confidence": "*how strong the model views the play*",
    "EV %": "*value estimate based on risk and payout*",
}

SCORE_EXPLANATIONS = {
    "Recent Form Score": "*how strong the player's recent trend looks*",
    "Matchup Score": "*how favorable this opponent matchup looks*",
    "Volatility Score": "*how swingy this stat is game to game*",
    "Minutes Risk Score": "*risk tied to unstable minutes*",
}

HIT_RATE_EXPLANATIONS = {
    "Last 5": "*hit rate over the last 5 games*",
    "Last 10": "*hit rate over the last 10 games*",
    "Last 15": "*hit rate over the last 15 games*",
    "Season": "*hit rate across the full season*",
}

CONTEXT_EXPLANATIONS = {
    "Projected Minutes": "*minutes the model expects tonight*",
    "Rest Days": "*days since the player's last game*",
    "Home/Away": "*whether the next matchup is home or away*",
    "Pace Proxy": "*approximate game speed for the matchup*",
}
LEG_METRIC_EXPLANATIONS = {
    "Projection": "*model's expected result for this leg*",
    "Last 5 Average": "*average over the last 5 games*",
    "Last 15 Average": "*average over the last 15 games*",
    "Season Average": "*full-season average for this stat*",
    "Selected Side": "*the side chosen for this leg*",
    "Selected Side %": "*estimated chance the chosen side hits*",
    "Confidence": "*how strong the model views this leg*",
    "Season Hit Rate": "*how often this exact side cleared the analyzed line during the current season window*",
}

_WIDGET_KEY_COUNTER = 0

def fresh_widget_key(prefix: str) -> str:
    global _WIDGET_KEY_COUNTER
    _WIDGET_KEY_COUNTER += 1
    return f"{prefix}_{_WIDGET_KEY_COUNTER}"


def safe_float(x, default=0.0, *args, **kwargs):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            if pd.isna(x):
                return default
            return float(x)
        x = str(x).strip()
        if x == "" or x.lower() in {"n/a", "nan", "none"}:
            return default
        value = float(x)
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default


def ml_result_available(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    ml_projection = result.get("ml_projection")
    model_method = str(result.get("model_method", "")).strip().lower()
    train_rows = safe_float(result.get("ml_train_rows"), 0.0)
    return (
        ml_projection is not None
        and str(ml_projection).strip().lower() not in {"", "n/a", "none", "nan"}
        and (model_method.startswith("blend_") or model_method == "ml_only" or train_rows > 0)
    )


def analyze_prop_with_ml_priority(player_name: str, stat: str, line: float, opponent: str, user_notes: str = ""):
    """
    Run analyze_prop once with the best ML-preference arguments supported by the imported function.
    The previous version tried many argument combinations, which could trigger multiple expensive
    analyses and make the UI appear stuck while loading.
    """
    base_kwargs = {
        "player_name": str(player_name).strip(),
        "stat": stat,
        "line": safe_float(line, 0.0),
        "opponent": str(opponent).strip(),
    }
    if user_notes is not None:
        base_kwargs["user_notes"] = str(user_notes).strip()

    try:
        signature = inspect.signature(analyze_prop)
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        accepted_params = set(signature.parameters.keys())
    except Exception:
        accepts_var_kwargs = True
        accepted_params = set(base_kwargs.keys())

    preferred_flag_sets = [
        {"force_ml": True, "allow_rules_only": False},
        {"use_ml": True, "prefer_ml": True},
        {"enable_ml": True, "prefer_ml": True},
        {"run_ml": True, "prefer_ml": True},
        {"model_preference": "ml"},
        {"prefer_model": "ml"},
        {"force_model": "ml"},
    ]

    call_kwargs = dict(base_kwargs)

    for extra_flags in preferred_flag_sets:
        if accepts_var_kwargs:
            call_kwargs.update(extra_flags)
            break

        supported = {k: v for k, v in extra_flags.items() if k in accepted_params}
        if supported:
            call_kwargs.update(supported)
            break

    try:
        result = analyze_prop(**call_kwargs)
    except TypeError:
        # Fallback to the plain baseline call if the preferred ML args are not accepted at runtime.
        result = analyze_prop(**base_kwargs)
    except Exception as exc:
        return {"error": f"Unable to analyze prop. Last error: {exc}"}

    if isinstance(result, dict):
        result["ml_attempted"] = True
    return result




def _format_percent_compact(value: float) -> str:
    value = safe_float(value, 0.0)
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}%"
    return f"{value:.1f}%"


def recalculate_current_season_hit_rate(result: dict | None) -> dict | None:
    if not isinstance(result, dict):
        return result

    sample_df = result.get("recent_games_df")
    if not isinstance(sample_df, pd.DataFrame) or sample_df.empty:
        return result

    working_df = sample_df.copy()

    if "Game Date" not in working_df.columns:
        return result

    working_df["Game Date"] = pd.to_datetime(working_df["Game Date"], errors="coerce")
    working_df = working_df[working_df["Game Date"] >= pd.Timestamp("2025-10-01")].copy()
    working_df = working_df[working_df["Game Date"].notna()].copy()

    if working_df.empty:
        result["season_hit_rate"] = "N/A"
        result["season_hit_rate_games"] = 0
        return result

    hits = None
    valid_rows = None

    if "Pick Result" in working_df.columns:
        pick_result = working_df["Pick Result"].astype(str).str.strip().str.lower()
        valid_mask = pick_result.isin(["over", "under", "push"])
        valid_rows = int(valid_mask.sum())
        if valid_rows > 0:
            hits = int((pick_result[valid_mask] == "over").sum())

    if hits is None:
        numeric_cols = [c for c in working_df.columns if c not in {"Game Date", "Pick Result"}]
        line_value = safe_float(result.get("line"), None)
        if line_value is not None:
            for col in reversed(numeric_cols):
                series = pd.to_numeric(working_df[col], errors="coerce")
                non_null = int(series.notna().sum())
                if non_null == 0:
                    continue
                valid_rows = non_null
                hits = int((series > line_value).sum())
                break

    if hits is None or not valid_rows:
        return result

    result["season_hit_rate"] = _format_percent_compact((hits / valid_rows) * 100.0)
    result["season_hit_rate_games"] = valid_rows
    return result


def normalize_projection_fields(result: dict | None) -> dict | None:
    """
    Make the displayed rules projection, ML projection, blend %, and final projection consistent.
    This fixes cases where the summary text and main Projection card disagree.
    """
    if not isinstance(result, dict):
        return result

    raw_projection = safe_float(result.get("projection"), None)
    raw_rules = safe_float(result.get("rules_projection"), None)
    raw_ml = safe_float(result.get("ml_projection"), None)

    model_method = str(result.get("model_method", "")).strip().lower()
    blend_weight = safe_float(result.get("ml_blend_weight"), None)

    has_ml = raw_ml is not None and str(result.get("ml_projection")).strip().lower() not in {"", "n/a", "none", "nan"}

    if raw_rules is None:
        if not has_ml:
            raw_rules = raw_projection
        elif model_method == "ml_only":
            raw_rules = None
        elif raw_projection is not None:
            raw_rules = raw_projection

    final_projection = raw_projection

    if has_ml:
        if model_method == "ml_only":
            final_projection = raw_ml
            blend_weight = 1.0
        elif blend_weight is not None and 0.0 <= blend_weight <= 1.0 and raw_rules is not None:
            final_projection = ((1.0 - blend_weight) * raw_rules) + (blend_weight * raw_ml)
        elif raw_projection is None:
            if raw_rules is not None:
                final_projection = (raw_rules + raw_ml) / 2.0
                blend_weight = 0.5
            else:
                final_projection = raw_ml
                blend_weight = 1.0
    else:
        final_projection = raw_rules if raw_rules is not None else raw_projection
        blend_weight = 0.0

    if final_projection is not None:
        result["projection"] = round(float(final_projection), 2)
    if raw_rules is not None:
        result["rules_projection"] = round(float(raw_rules), 2)
    if has_ml and raw_ml is not None:
        result["ml_projection"] = round(float(raw_ml), 2)
    else:
        result["ml_projection"] = None
    result["ml_blend_weight"] = round(float(blend_weight or 0.0), 4)

    return result


LINE_OPTIONS = [f"{x/2:g}" for x in range(1, 101)]

PROPIFY_ICON_BASE64 = "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"
PROPIFY_ICON_CROPPED_BASE64 = "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"
PROPIFY_FULL_LOGO_BASE64 = "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"
DH_LOGO_BASE64 = "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jCJDuVJIBeOoiHOwoJGfypCHx99UTBQg1FZB9RLBP0PIAhRIRYU4itd7Ufi/RsU3js9mzz+x6eALX3v4uDvPP3Laeug2tOxtBLW8A1kCWCkH8vDKY1u3sW1KR77wotDEzUop6HJDWPCkFPxexNhY5XqSR3gxRmyJk6PekhipDjtJyrT2SULZWZqYHPOxi1atfeFC51JlCnv/OXv27OiNZxx757vOPPJ77z3zsO+959QZ33nPaYd888LTZn3zracf9a03v/job7z11CNuvfDUw2+96JSZX3/HabNuecdpR91y4WlHfu2i04748oWC04+66cLTj7zxwlMRnjrrpnch/S4JTz30pitfd+pNV5x/8s2Xn3viTZ8978Sbrjlv9k1fePXsG685/9gbrjn/uBuuOe/4m655zYk3Xf3ak27+3GtPvvlzF5x28ztOnn7ze1986M3vPfOIm9+O8N0vPermi15yxJff+7LZN7/jzGNufsdpx9z8kZe/6OaLX4m6pxz9lXcA73vJCV9+52nHfvm9Z85Gu9k3v+8lxyF+/Fffc+bxX3vPGcd95f0vOfEr7znruC+/86xjv3zRWUd9+V1nH3WT4N0vPeLm9wLve+nMmz545uE3v/+0GTd/+lXg74IX3nDVa8Bbwt/sm64Gz9e+6ugbu0LEJX3VuUffcM0rj77h6nOPvf6aVx3zpWtefcyXvvDq2V+85rxjv3Ttq4794jXnHvOFa8+dfd3V582+7jrEJf2FVx973bXnz/6i1KsOrzt39hek/Npzy/WlHuT1hST/PJSdd/wXPnf+yV+4/FUnXnvV/536hfeedugt7zrlkG+++7TDv/HOU2d9/6LTZ/0E+MVbTp35h7ecOuu+C198xEMXnjF78ZtedMSWV84em8daslS7BrUELjzlsD994JTpX34H9tvbzzjqlgvPPPJrF7541tfed8axX3/7ybNueefpx379IuAdZxxzi+CiF8/+moQCib/7JSfckqx5WfdnHf81iVfy3os2HzjjiK9jj9/6mXNf8PWrLzjl6x944bSvX/e6k2+59vyTv3bdeSfccsPrTrrlmlccc8v15x339evPO+HrWNO3XH/BC77+pfNPvPXiUw+79b0vPvIb7zr92FsTnDb7m++qwrtfPPsb1aiUSd57zkDdM4//zssPG1v7U0qDegXuGnNznPOe2eAaHtlS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"
PROPIFY_TRANSPARENT_LOGO_BASE64 = "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"

SUPABASE_PICKS_TABLE = "picks"


PICK_HISTORY_FILE = "pick_history.csv"
FAVORITES_FILE = "favorite_players.json"
RECENT_SEARCHES_FILE = "recent_searches.json"

PICK_HISTORY_COLUMNS = [
    "pick_id",
    "entry_type",
    "legs_count",
    "saved_at",
    "settled_at",
    "label",
    "confidence",
    "confidence_score_numeric",
    "selected_prob",
    "parlay_hit_prob",
    "parlay_miss_prob",
    "bet_size",
    "multiplier",
    "amount_won",
    "amount_lost",
    "net_result",
    "cash_out_price",
    "outcome",
    "user_notes",
    "player",
    "stat",
    "line",
    "opponent",
    "selected_side",
    "projection",
    "season_average",
    "last10_average",
    "last15_average",
    "last5_average",
    "recent_form_score",
    "matchup_score",
    "volatility_score",
    "minutes_risk_score",
    "is_flex_pick",
    "flex_full_hit_multiplier",
    "flex_one_miss_multiplier",
    "flex_full_hit_legs",
    "flex_one_miss_hit_legs",
    "closing_line",
    "closing_line_result",
    "actual_result",
    "leg1_player",
    "leg1_stat",
    "leg1_line",
    "leg1_opponent",
    "leg1_side",
    "leg1_projection",
    "leg1_selected_prob",
    "leg1_confidence",
    "leg1_confidence_score_numeric",
    "leg1_season_average",
    "leg1_last15_average",
    "leg1_last10_average",
    "leg1_last5_average",
    "leg1_closing_line",
    "leg1_closing_line_result",
    "actual_result_leg1",
    "leg2_player",
    "leg2_stat",
    "leg2_line",
    "leg2_opponent",
    "leg2_side",
    "leg2_projection",
    "leg2_selected_prob",
    "leg2_confidence",
    "leg2_confidence_score_numeric",
    "leg2_season_average",
    "leg2_last15_average",
    "leg2_last10_average",
    "leg2_last5_average",
    "leg2_closing_line",
    "leg2_closing_line_result",
    "actual_result_leg2",
    "leg3_player",
    "leg3_stat",
    "leg3_line",
    "leg3_opponent",
    "leg3_side",
    "leg3_projection",
    "leg3_selected_prob",
    "leg3_confidence",
    "leg3_confidence_score_numeric",
    "leg3_season_average",
    "leg3_last15_average",
    "leg3_last10_average",
    "leg3_last5_average",
    "leg3_closing_line",
    "leg3_closing_line_result",
    "actual_result_leg3",
    "leg4_player",
    "leg4_stat",
    "leg4_line",
    "leg4_opponent",
    "leg4_side",
    "leg4_projection",
    "leg4_selected_prob",
    "leg4_confidence",
    "leg4_confidence_score_numeric",
    "leg4_season_average",
    "leg4_last15_average",
    "leg4_last10_average",
    "leg4_last5_average",
    "leg4_closing_line",
    "leg4_closing_line_result",
    "actual_result_leg4",
]

TARHEEL_BLUE = "#7BAFD4"
COLORS = {
    "strong": "#d9f2e3",
    "strong_dark": "#1f7a4d",
    "neutral": "#fff5cf",
    "neutral_dark": "#8a6d1d",
    "risky": "#f9d8d8",
    "risky_dark": "#9b2c2c",
    "info": "#d9ebfa",
    "info_dark": "#1f5f99",
    "surface": "#ffffff",
    "border": "#d9e6f2",
}

STAT_OPTIONS = [
    "points", "rebounds", "assists", "fgm", "fga", "2pm", "2pa", "3pm", "3pa",
    "steals", "blocks", "stocks", "turnovers", "fouls", "ftm", "fta", "oreb", "dreb", "fantasy",
    "PRA", "RA", "PA", "PR",
]

STAT_DISPLAY_MAP = {
    "points": "Points",
    "rebounds": "Rebounds",
    "assists": "Assists",
    "fgm": "Field Goals Made",
    "fga": "Field Goals Attempted",
    "2pm": "Two-Pointers Made",
    "2pa": "Two-Point Attempts",
    "3pm": "Three-Pointers Made",
    "3pa": "Three-Point Attempts",
    "steals": "Steals",
    "blocks": "Blocks",
    "stocks": "Steals + Blocks",
    "turnovers": "Turnovers",
    "fouls": "Fouls",
    "ftm": "Free Throws Made",
    "fta": "Free Throws Attempted",
    "oreb": "Offensive Rebounds",
    "dreb": "Defensive Rebounds",
    "fantasy": "Fantasy Score",
    "PRA": "PRA",
    "RA": "RA",
    "PA": "PA",
    "PR": "PR",
}


def _format_line_option(value):
    try:
        value = float(value)
    except Exception:
        return str(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def line_input_widget(label, key_base, default=10):
    options = [round(x * 0.5, 1) for x in range(1, 101)]
    current_value = safe_float(st.session_state.get(key_base, default), safe_float(default, 10.0))
    normalized = min(50.0, max(0.5, round(current_value * 2) / 2))
    if normalized not in options:
        normalized = safe_float(default, 10.0)
    selected = st.selectbox(
        label,
        options,
        index=options.index(normalized),
        key=f"{key_base}_selectbox",
        format_func=_format_line_option,
    )
    st.session_state[key_base] = float(selected)
    return float(selected)


def current_season_string_app():
    now = datetime.now()
    year = now.year
    if now.month >= 10:
        start = year
    else:
        start = year - 1
    end = str(start + 1)[-2:]
    return f"{start}-{end}"


@st.cache_data(ttl=3600, show_spinner=False)
def get_player_directory():
    try:
        season = current_season_string_app()
        df = commonallplayers.CommonAllPlayers(is_only_current_season=1, season=season).get_data_frames()[0]
        name_col = "DISPLAY_FIRST_LAST" if "DISPLAY_FIRST_LAST" in df.columns else "DISPLAY_LAST_COMMA_FIRST"
        team_col = "TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in df.columns else None

        directory = []
        for _, row in df.iterrows():
            full_name = str(row.get(name_col, "")).strip()
            if not full_name:
                continue
            team_abbr = str(row.get(team_col, "")).strip() if team_col else ""
            display_name = f"{full_name} ({team_abbr})" if team_abbr else full_name
            directory.append({
                "display_name": display_name,
                "player_name": full_name,
                "team_abbr": team_abbr,
            })

        directory = sorted(directory, key=lambda x: x["display_name"])
        if directory:
            return directory
    except Exception:
        pass

    try:
        fallback = []
        for player in players.get_active_players():
            name = str(player.get("full_name", "")).strip()
            if name:
                fallback.append({
                    "display_name": name,
                    "player_name": name,
                    "team_abbr": "",
                })
        return sorted(fallback, key=lambda x: x["display_name"])
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_player_options():
    return [row["display_name"] for row in get_player_directory()]


def parse_player_display(player_display: str):
    if not player_display:
        return "", ""

    directory = get_player_directory()
    for row in directory:
        if row["display_name"] == player_display:
            return row["player_name"], row["team_abbr"]

    text_value = str(player_display).strip()
    if text_value.endswith(")") and " (" in text_value:
        base_name, suffix = text_value.rsplit(" (", 1)
        return base_name.strip(), suffix[:-1].strip()
    return text_value, ""


@st.cache_data(ttl=3600, show_spinner=False)
def get_team_options():
    try:
        return sorted({t["nickname"] for t in teams.get_teams()})
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_team_abbr_to_nickname_map():
    try:
        return {str(t["abbreviation"]): str(t["nickname"]) for t in teams.get_teams()}
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_next_matchup_context(team_abbr: str):
    if not team_abbr:
        return None, "Neutral"

    abbr_to_nickname = get_team_abbr_to_nickname_map()
    eastern_now = datetime.now(ZoneInfo("America/New_York"))
    base_date = eastern_now.date()
    next_future = (None, "Neutral")

    for day_offset in range(0, 14):
        target_date = base_date + timedelta(days=day_offset)
        game_date = target_date.strftime("%m/%d/%Y")
        try:
            data_frames = scoreboardv2.ScoreboardV2(game_date=game_date).get_data_frames()
        except Exception:
            continue

        if len(data_frames) < 2:
            continue

        game_header_df = data_frames[0]
        line_score_df = data_frames[1]

        if line_score_df is None or line_score_df.empty:
            continue
        if "GAME_ID" not in line_score_df.columns or "TEAM_ABBREVIATION" not in line_score_df.columns:
            continue

        header_lookup = {}
        if game_header_df is not None and not game_header_df.empty and "GAME_ID" in game_header_df.columns:
            for _, header_row in game_header_df.iterrows():
                header_lookup[str(header_row["GAME_ID"])] = header_row.to_dict()

        for game_id, game_df in line_score_df.groupby("GAME_ID"):
            teams_in_game = game_df["TEAM_ABBREVIATION"].astype(str).tolist()
            if team_abbr not in teams_in_game or len(teams_in_game) < 2:
                continue

            opp_abbr = next((abbr for abbr in teams_in_game if abbr != team_abbr), None)
            if not opp_abbr:
                continue

            header = header_lookup.get(str(game_id), {})
            status_id = header.get("GAME_STATUS_ID")
            status_text = str(header.get("GAME_STATUS_TEXT", "")).strip().lower()

            is_final = False
            if status_id is not None:
                try:
                    is_final = int(status_id) == 3
                except Exception:
                    is_final = False
            if not is_final and status_text:
                is_final = status_text.startswith("final")

            team_rows = game_df.copy()
            team_rows["TEAM_ABBREVIATION"] = team_rows["TEAM_ABBREVIATION"].astype(str)
            team_row = team_rows[team_rows["TEAM_ABBREVIATION"] == team_abbr].head(1)
            matchup_text = str(team_row["MATCHUP"].iloc[0]) if (not team_row.empty and "MATCHUP" in team_row.columns) else ""
            home_away = "Neutral"
            if " vs. " in matchup_text:
                home_away = "Home"
            elif " @ " in matchup_text:
                home_away = "Away"

            opp_name = abbr_to_nickname.get(opp_abbr, opp_abbr)
            if day_offset == 0:
                if not is_final:
                    return opp_name, home_away
            else:
                if next_future[0] is None:
                    next_future = (opp_name, home_away)
                    return next_future

    return next_future


def get_next_opponent_nickname(team_abbr: str):
    opp, _ = get_next_matchup_context(team_abbr)
    return opp

@st.cache_data(ttl=3600, show_spinner=False)
def get_team_name_variant_lookup():
    lookup = {}
    try:
        for t in teams.get_teams():
            abbr = str(t.get("abbreviation", "")).strip()
            nickname = str(t.get("nickname", "")).strip()
            full_name = str(t.get("full_name", "")).strip()
            city = str(t.get("city", "")).strip()
            names = {abbr, nickname, full_name, city}
            normalized = {n.lower() for n in names if n}
            for n in normalized:
                lookup[n] = normalized
    except Exception:
        return {}
    return lookup


def expand_team_filter_variants(team_value: str | None) -> set[str]:
    raw = str(team_value or "").strip().lower()
    if not raw:
        return set()
    lookup = get_team_name_variant_lookup()
    variants = set()
    if raw in lookup:
        variants |= lookup[raw]
    else:
        variants.add(raw)
    return {v for v in variants if v}


def text_contains_team_variant(text_value: str, team_variants: set[str]) -> bool:
    raw = str(text_value or "").strip().lower()
    if not raw or not team_variants:
        return False
    cleaned = (
        raw.replace("vs.", " ")
           .replace("vs", " ")
           .replace("@", " ")
           .replace("-", " ")
           .replace("/", " ")
           .replace("(", " ")
           .replace(")", " ")
           .replace(",", " ")
    )
    tokens = [tok for tok in cleaned.split() if tok]
    token_set = set(tokens)
    for variant in team_variants:
        if variant in raw:
            return True
        if variant in token_set:
            return True
        variant_tokens = tuple(tok for tok in variant.split() if tok)
        if variant_tokens and all(tok in token_set for tok in variant_tokens):
            return True
    return False



def update_auto_opponent(player_key: str, opp_key: str, last_player_key: str, auto_opp_key: str):
    selected_player_display = st.session_state.get(player_key)
    if not selected_player_display:
        return

    last_player_display = st.session_state.get(last_player_key)
    if selected_player_display == last_player_display:
        return

    _, team_abbr = parse_player_display(selected_player_display)
    next_opp, _ = get_next_matchup_context(team_abbr)
    if next_opp:
        st.session_state[opp_key] = next_opp
        st.session_state[auto_opp_key] = next_opp
    st.session_state[last_player_key] = selected_player_display


def sync_single_auto_opponent():
    selected_player_display = st.session_state.get("single_player")
    last_player_display = st.session_state.get("_single_last_auto_player")

    if not selected_player_display:
        return

    if selected_player_display != last_player_display:
        _, team_abbr = parse_player_display(selected_player_display)
        next_opp, _ = get_next_matchup_context(team_abbr)
        if next_opp:
            st.session_state["single_opponent"] = next_opp
            st.session_state["_single_auto_opponent"] = next_opp
        st.session_state["_single_last_auto_player"] = selected_player_display


def sync_parlay_auto_opponent(section_key: str, leg_index: int):
    player_key = f"{section_key}_player_{leg_index}"
    opp_key = f"{section_key}_opp_{leg_index}"
    last_player_key = f"_{section_key}_last_auto_player_{leg_index}"
    auto_opp_key = f"_{section_key}_auto_opponent_{leg_index}"

    selected_player_display = st.session_state.get(player_key)
    last_player_display = st.session_state.get(last_player_key)

    if not selected_player_display:
        return

    if selected_player_display != last_player_display:
        _, team_abbr = parse_player_display(selected_player_display)
        next_opp, _ = get_next_matchup_context(team_abbr)
        if next_opp:
            st.session_state[opp_key] = next_opp
            st.session_state[auto_opp_key] = next_opp
        st.session_state[last_player_key] = selected_player_display


def stat_display_name(stat_value):
    if stat_value is None or (isinstance(stat_value, float) and pd.isna(stat_value)):
        return ""
    stat_str = str(stat_value)
    return STAT_DISPLAY_MAP.get(stat_str, STAT_DISPLAY_MAP.get(stat_str.lower(), stat_str))


def searchable_selectbox(label: str, options, key: str, placeholder: str, on_change=None, args=None):
    return st.selectbox(
        label,
        options=options,
        index=None,
        placeholder=placeholder,
        key=key,
        on_change=on_change,
        args=args or (),
    )


def inject_css():
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: "Segoe UI", "Trebuchet MS", "Helvetica Neue", Arial, sans-serif;
        }}

        .stApp {{
            background: #f5fafe;
        }}

        .block-container {{
            padding-top: 0.0rem;
            padding-bottom: 2rem;
        }}

        .propify-hero-card {{
            background: #ffffff;
            border: 1px solid rgba(18,34,53,0.08);
            border-radius: 24px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
            padding: 24px 18px 18px 18px;
            margin: 0 auto 14px auto;
            text-align: center;
        }}

        .propify-hero-card img {{
            width: min(100%, 420px);
            height: auto;
            display: block;
            margin: 0 auto;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 2.4rem;
            justify-content: flex-start;
            align-items: center;
            padding-left: 0 !important;
            margin-left: -0.45rem;
            border-bottom: 1.5px solid rgba(138,157,181,0.30);
            margin-top: 0.95rem;
            margin-bottom: 1.1rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 2.8rem;
            margin-right: 0.35rem;
            white-space: nowrap;
            border-radius: 10px 10px 0 0;
            padding-left: 0.15rem;
            padding-right: 0.15rem;
            font-weight: 700;
            color: #f5f7fb !important;
            border-bottom: 2px solid transparent !important;
        }}

        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            color: #f5f7fb !important;
            border-bottom: 2px solid transparent !important;
        }}

        .stTabs [data-baseweb="tab-highlight"] {{
            background: #7BAFD4 !important;
            height: 3px !important;
            border-radius: 999px !important;
        }}

        .propify-guide-item {{
            margin: 0 0 1.85rem 0;
        }}

        .propify-guide-title {{
            color: #f5f7fb;
            font-size: 1.14rem;
            font-weight: 600;
            line-height: 1.2;
            margin: 0 0 0.4rem 0;
            display: inline-block;
            padding-bottom: 0.22rem;
            border-bottom: 2px solid #7BAFD4;
        }}

        .propify-guide-line {{
            margin: 0.32rem 0 0 0;
            line-height: 1.65;
            font-size: 1.02rem;
        }}

        .propify-guide-label {{
            color: #7BAFD4;
            font-weight: 600;
        }}

        .propify-guide-meaning {{
            color: #f5f7fb;
        }}

        .propify-guide-usecase {{
            color: #c9d3de;
        }}

        .propify-action-spacer-top {{
            height: 1.35rem;
        }}

        .propify-action-spacer-bottom {{
            height: 1.35rem;
        }}

        .propify-topbar {{
            display:flex;
            align-items:center;
            justify-content:flex-start;
            min-height: 34px;
            margin: -0.15rem 0 1.9rem 0;
            padding-top: 0;
        }}

        .propify-topbar img {{
            width: 34px;
            height: 34px;
            display:block;
            object-fit: contain;
            flex-shrink: 0;
            margin: 0;
        }}

        .propify-topbar-engine {{
            margin-left: 0.55rem;
            margin-top: 0;
            font-size: 1.32rem;
            font-weight: 700;
            color: #7BAFD4;
            line-height: 1;
            align-self: center;
            display:flex;
            align-items:center;
            height: 34px;
        }}


        .propify-global-mode {{
            display:flex;
            justify-content:flex-end;
            align-items:flex-start;
            padding-top: 0.25rem;
        }}
        .propify-global-mode [data-testid="stSelectbox"] {{
            margin-top: 0 !important;
        }}
        .propify-global-mode [data-testid="stWidgetLabel"] p,
        .propify-global-mode label,
        .propify-global-mode .stSelectbox label {{
            color: #f5f7fb !important;
            font-weight: 600 !important;
        }}
        .propify-section-title {{
            margin: 0.1rem 0 1.0rem 0;
        }}

        .propify-section-title h1 {{
            margin: 0;
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.05;
        }}

        .propify-section-divider {{
            width: 120px;
            height: 4px;
            border-radius: 999px;
            background: linear-gradient(90deg, #7BAFD4 0%, #9ec7e6 100%);
            margin-top: 0.18rem;
        }}

        .propify-leg-label {{
            margin: 0.05rem 0 0.18rem 0;
            font-size: 1rem;
            font-weight: 400;
            color: #7BAFD4;
            line-height: 1.1;
        }}

        .stExpander {{
            border: 1px solid {COLORS["border"]} !important;
            border-radius: 18px !important;
            background: white !important;
            box-shadow: 0 4px 14px rgba(0,0,0,0.04) !important;
            margin-bottom: 16px !important;
            overflow: hidden !important;
        }}

        .stExpander summary {{
            font-size: 2rem !important;
            font-weight: 850 !important;
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            line-height: 1.15 !important;
        }}

        .block-container [data-testid="stDataFrame"] table {{
            width: 100% !important;
        }}

        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid rgba(0,0,0,0.06);
            padding: 10px 12px;
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        }}

        .fade-in {{
            animation: fadeIn 0.35s ease-out;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .primary-card {{
            background: white;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.05);
            min-height: 110px;
        }}

        .primary-card .label {{
            font-size: 0.92rem;
            font-weight: 700;
            opacity: 0.78;
        }}

        .primary-card .value {{
            font-size: 1.95rem;
            font-weight: 900;
            margin-top: 8px;
            line-height: 1.08;
            white-space: normal;
            word-break: break-word;
        }}

        .secondary-card {{
            background: white;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 16px;
            padding: 12px 14px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.04);
            min-height: 88px;
        }}

        .secondary-card .label {{
            font-size: 0.88rem;
            font-weight: 700;
            opacity: 0.74;
        }}

        .secondary-card .value {{
            font-size: 1.48rem;
            font-weight: 850;
            margin-top: 7px;
            line-height: 1.08;
            white-space: normal;
            word-break: break-word;
        }}

        .section-space {{
            height: 12px;
        }}

        .pattern-header {{
            background-image:
                linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px);
            background-size: 24px 24px;
        }}

        .chart-card {{
            background: white;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 18px;
            padding: 18px 18px 8px 18px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.05);
            margin-top: 18px;
            margin-bottom: 26px;
        }}

        .chart-title {{
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )



def _get_secret(name: str) -> str:
    value = st.secrets.get(name)
    if not value:
        raise RuntimeError(f"Missing Streamlit secret: {name}")
    return value


@st.cache_resource
def get_supabase() -> Client:
    return create_client(
        _get_secret("supabase_url"),
        _get_secret("thesupabase_anon_key"),
    )


def _extract_tokens(auth_response):
    session = getattr(auth_response, "session", None)
    if session is None and isinstance(auth_response, dict):
        session = auth_response.get("session")
    if session is None:
        return None

    access_token = getattr(session, "access_token", None)
    refresh_token = getattr(session, "refresh_token", None)

    if isinstance(session, dict):
        access_token = access_token or session.get("access_token")
        refresh_token = refresh_token or session.get("refresh_token")

    if access_token and refresh_token:
        return {"access_token": access_token, "refresh_token": refresh_token}
    return None


def restore_session():
    tokens = st.session_state.get("sb_session_tokens")
    if not tokens:
        return
    try:
        response = get_supabase().auth.set_session(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
        )
        new_tokens = _extract_tokens(response)
        if new_tokens:
            st.session_state["sb_session_tokens"] = new_tokens
    except Exception:
        st.session_state.pop("sb_session_tokens", None)


def get_current_user():
    try:
        response = get_supabase().auth.get_user()
        user = getattr(response, "user", None)
        if user is None and isinstance(response, dict):
            user = response.get("user")
        return user
    except Exception:
        return None


def get_current_user_id():
    user = st.session_state.get("auth_user")
    if user is None:
        return None
    return getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)


def sign_up(email: str, password: str):
    try:
        response = get_supabase().auth.sign_up({"email": email, "password": password})
        tokens = _extract_tokens(response)
        if tokens:
            st.session_state["sb_session_tokens"] = tokens
        return True, "Account created. Confirm your email first if email confirmation is enabled."
    except Exception as e:
        return False, str(e)


def sign_in(email: str, password: str):
    try:
        response = get_supabase().auth.sign_in_with_password({"email": email, "password": password})
        tokens = _extract_tokens(response)
        if not tokens:
            return False, "No session tokens were returned."
        st.session_state["sb_session_tokens"] = tokens
        return True, "Signed in."
    except Exception as e:
        return False, str(e)


def sign_out():
    try:
        get_supabase().auth.sign_out()
    except Exception:
        pass
    st.session_state.pop("sb_session_tokens", None)
    st.session_state.pop("auth_user", None)
    st.session_state.pop("app_view", None)
    st.session_state.pop("propify_force_tab", None)
    st.rerun()




def render_auth_gate():
    restore_session()
    user = get_current_user()

    if user is not None:
        return user

    left, center, right = st.columns([1.15, 3.8, 1.15])
    with center:
        st.markdown(
            f"""
            <div style="
                max-width: 900px;
                margin: 2.2rem auto 0 auto;
                background: linear-gradient(135deg, rgba(141,192,226,0.18) 0%, rgba(123,175,212,0.12) 45%, rgba(217,235,250,0.34) 100%);
                border: 1px solid rgba(123,175,212,0.28);
                border-radius: 28px;
                padding: 34px 34px 22px 34px;
                box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
                text-align: center;
            ">
                <img src="data:image/png;base64,{PROPIFY_ICON_CROPPED_BASE64}"
                     alt="Propify Logo"
                     style="
                        width: 132px;
                        max-width: 40%;
                        height: auto;
                        object-fit: contain;
                        filter: drop-shadow(0px 8px 18px rgba(0,0,0,0.10));
                        margin-bottom: 8px;
                     " />
                <div style="
                    font-size: 2.15rem;
                    font-weight: 900;
                    color: #17344f;
                    line-height: 1.08;
                    letter-spacing: 0.1px;
                    margin-top: 4px;
                ">
                    Welcome to Propify
                </div>
                <div style="
                    font-size: 1.08rem;
                    font-weight: 600;
                    color: #2c5f86;
                    margin-top: 12px;
                    line-height: 1.55;
                ">
                    Log in or create an account to start analyzing props, building parlays,
                    and tracking your winnings over time.
                </div>
                <div style="
                    margin-top: 18px;
                    font-size: 0.96rem;
                    color: #4b647a;
                    line-height: 1.6;
                ">
                    Your analysis history, saved entries, and tracking data are tied to your account,
                    giving you a secure, personalized experience every time you use the engine.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        with st.container(border=False):
            st.markdown(
                """
                <div style="
                    max-width: 430px;
                    margin: 0 auto;
                    background: rgba(255,255,255,0.92);
                    border: 1px solid rgba(123,175,212,0.22);
                    border-radius: 22px;
                    padding: 22px 22px 18px 22px;
                    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
                ">
                    <div style="
                        text-align:center;
                        font-size:1.22rem;
                        font-weight:800;
                        color:#17344f;
                        margin-bottom:12px;
                    ">
                        Account Access
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            login_email = st.text_input("Email", key="auth_email", placeholder="Enter your email")
            login_password = st.text_input("Password", type="password", key="auth_password", placeholder="Enter your password")

            st.markdown("<div style='margin-top:-0.18rem;'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                login_clicked = st.button("Log In", use_container_width=True)
            with c2:
                signup_clicked = st.button("Sign Up", use_container_width=True)

            if login_clicked:
                ok, message = sign_in(login_email.strip(), login_password)
                if ok:
                    st.session_state["app_view"] = "home"
                    st.session_state["propify_force_tab"] = "Single Prop Analyzer"
                    st.rerun()
                st.error(message)

            if signup_clicked:
                ok, message = sign_up(login_email.strip(), login_password)
                if ok:
                    st.session_state["app_view"] = "home"
                    st.session_state["propify_force_tab"] = "Single Prop Analyzer"
                    st.rerun()
                else:
                    st.error(message)

    return None


def _json_safe_value(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        if pd.isna(value) or math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value

def normalize_pick_payload(payload: dict):
    cleaned = {col: _json_safe_value(payload.get(col, "")) for col in PICK_HISTORY_COLUMNS}
    if not cleaned.get("pick_id"):
        cleaned["pick_id"] = str(uuid.uuid4())[:8]
    if not cleaned.get("saved_at"):
        cleaned["saved_at"] = datetime.now(timezone.utc).isoformat()
    return cleaned

def format_saved_display(value):
    if value is None or value == "":
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    hour = parsed.hour % 12
    hour = 12 if hour == 0 else hour
    ampm = "am" if parsed.hour < 12 else "pm"
    return f"{parsed.month}/{parsed.day}/{str(parsed.year)[-2:]} {hour}:{parsed.minute:02d}{ampm}"


def ensure_json_file(path: str, default_obj):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_obj, f)


def read_json(path: str, default_obj):
    ensure_json_file(path, default_obj)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_obj


def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ensure_pick_history_file():
    return None


def ensure_pick_history_columns(df: pd.DataFrame):
    for col in PICK_HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[PICK_HISTORY_COLUMNS]


def format_timestamp(dt=None):
    if dt is None:
        dt = datetime.now()
    hour = dt.hour % 12
    hour = 12 if hour == 0 else hour
    ampm = "am" if dt.hour < 12 else "pm"
    return f"{dt.month}/{dt.day}/{str(dt.year)[-2:]} {hour}:{dt.minute:02d}{ampm}"


def pretty_headers(df: pd.DataFrame):
    return df.rename(columns={col: col.replace("_", " ").title() for col in df.columns})


def entry_type_label(legs_count: int):
    mapping = {1: "single", 2: "two-leg", 3: "three-leg", 4: "four-leg"}
    return mapping.get(legs_count, "parlay")



def load_pick_history(user_id: str = None):
    resolved_user_id = user_id or get_current_user_id()
    if not resolved_user_id:
        return pd.DataFrame(columns=PICK_HISTORY_COLUMNS)

    response = (
        get_supabase()
        .table(SUPABASE_PICKS_TABLE)
        .select("pick_id,user_id,saved_at,updated_at,payload")
        .eq("user_id", resolved_user_id)
        .order("saved_at", desc=True)
        .execute()
    )
    rows = getattr(response, "data", None) or []
    payload_rows = []
    for row in rows:
        payload = row.get("payload", {}) or {}
        payload["pick_id"] = row.get("pick_id", payload.get("pick_id", ""))
        payload["saved_at"] = row.get("saved_at", payload.get("saved_at", ""))
        payload_rows.append(normalize_pick_payload(payload))

    if not payload_rows:
        return pd.DataFrame(columns=PICK_HISTORY_COLUMNS)

    df = pd.DataFrame(payload_rows, columns=PICK_HISTORY_COLUMNS)

    numeric_cols = [
        "legs_count", "selected_prob", "parlay_hit_prob", "parlay_miss_prob", "bet_size",
        "multiplier", "amount_won", "amount_lost", "net_result", "cash_out_price", "line", "projection",
        "season_average", "last10_average", "last15_average", "last5_average",
        "recent_form_score", "matchup_score", "volatility_score", "minutes_risk_score",
        "flex_full_hit_multiplier", "flex_one_miss_multiplier", "flex_full_hit_legs", "flex_one_miss_hit_legs",
        "actual_result", "closing_line", "confidence_score_numeric",
        "leg1_line", "leg1_projection", "leg1_selected_prob", "leg1_season_average",
        "leg1_last15_average", "leg1_last10_average", "leg1_last5_average", "leg1_closing_line",
        "actual_result_leg1", "leg1_confidence_score_numeric",
        "leg2_line", "leg2_projection", "leg2_selected_prob", "leg2_season_average",
        "leg2_last15_average", "leg2_last10_average", "leg2_last5_average", "leg2_closing_line",
        "actual_result_leg2", "leg2_confidence_score_numeric",
        "leg3_line", "leg3_projection", "leg3_selected_prob", "leg3_season_average",
        "leg3_last15_average", "leg3_last10_average", "leg3_last5_average", "leg3_closing_line",
        "actual_result_leg3", "leg3_confidence_score_numeric",
        "leg4_line", "leg4_projection", "leg4_selected_prob", "leg4_season_average",
        "leg4_last15_average", "leg4_last10_average", "leg4_last5_average", "leg4_closing_line",
        "actual_result_leg4", "leg4_confidence_score_numeric",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["saved_at", "settled_at"]:
        df[col] = df[col].fillna("")

    return df


def save_pick_to_history(payload: dict, user_id: str = None):
    resolved_user_id = user_id or get_current_user_id()
    if not resolved_user_id:
        return False

    cleaned = normalize_pick_payload(payload)
    record = {
        "pick_id": cleaned["pick_id"],
        "user_id": resolved_user_id,
        "saved_at": cleaned["saved_at"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "label": cleaned.get("label", ""),
        "entry_type": cleaned.get("entry_type", ""),
        "stat": cleaned.get("stat", ""),
        "outcome": cleaned.get("outcome", ""),
        "confidence": cleaned.get("confidence", ""),
        "bet_size": safe_float(cleaned.get("bet_size"), 0.0, 0.0),
        "multiplier": safe_float(cleaned.get("multiplier"), 0.0, 0.0),
        "payload": cleaned,
    }
    get_supabase().table(SUPABASE_PICKS_TABLE).upsert(record, on_conflict="pick_id").execute()
    return True


def delete_pick_from_history(pick_id: str, user_id: str = None):
    resolved_user_id = user_id or get_current_user_id()
    if not resolved_user_id:
        return False
    (
        get_supabase()
        .table(SUPABASE_PICKS_TABLE)
        .delete()
        .eq("user_id", resolved_user_id)
        .eq("pick_id", str(pick_id))
        .execute()
    )
    return True


def update_pick_in_history(pick_id: str, updates: dict, user_id: str = None):
    resolved_user_id = user_id or get_current_user_id()
    if not resolved_user_id:
        return False

    df = load_pick_history(resolved_user_id)
    match_idx = df.index[df["pick_id"].astype(str) == str(pick_id)].tolist()
    if not match_idx:
        return False

    idx = match_idx[0]
    payload = df.iloc[idx].to_dict()
    for key, value in updates.items():
        if key in payload:
            payload[key] = value

    outcome = str(payload.get("outcome", "")).strip().lower()
    if outcome in ["win", "loss", "push", "cash out"]:
        amount_won, amount_lost, net_result = calculate_money_from_outcome(
            outcome,
            pd.to_numeric(payload.get("bet_size"), errors="coerce"),
            pd.to_numeric(payload.get("multiplier"), errors="coerce"),
            pd.to_numeric(payload.get("cash_out_price"), errors="coerce"),
        )
        payload["amount_won"] = amount_won
        payload["amount_lost"] = amount_lost
        payload["net_result"] = net_result
    else:
        payload["amount_won"] = ""
        payload["amount_lost"] = ""
        payload["net_result"] = ""

    return save_pick_to_history(payload, resolved_user_id)


def _line_has_push(line_value) -> bool:
    try:
        return float(line_value).is_integer()
    except Exception:
        return False


def adjusted_display_probs(result: dict, side_for_push: str | None = None):
    over_prob = safe_float(result.get("over_prob"), 0.0)
    under_prob = safe_float(result.get("under_prob"), 0.0)
    push_prob = safe_float(result.get("push_prob"), 0.0)

    if not _line_has_push(result.get("line", 0)):
        push_prob = 0.0

    total = over_prob + under_prob + push_prob
    if total <= 0:
        return 0.0, 0.0, 0.0

    over_prob = (over_prob / total) * 100
    under_prob = (under_prob / total) * 100
    push_prob = (push_prob / total) * 100

    over_prob = round(over_prob, 1)
    under_prob = round(under_prob, 1)
    push_prob = round(push_prob, 1)

    rounded_total = round(over_prob + under_prob + push_prob, 1)
    diff = round(100.0 - rounded_total, 1)
    if abs(diff) > 0:
        buckets = {
            "over": over_prob,
            "under": under_prob,
            "push": push_prob,
        }
        largest = max(buckets, key=buckets.get)
        if largest == "over":
            over_prob = round(over_prob + diff, 1)
        elif largest == "under":
            under_prob = round(under_prob + diff, 1)
        else:
            push_prob = round(push_prob + diff, 1)

    return over_prob, under_prob, push_prob


def selected_side_prob(result: dict, side: str):
    over_prob, under_prob, _ = adjusted_display_probs(result, side)
    return over_prob if side.upper() == "OVER" else under_prob


def bool_from_value(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def get_pick_type_label(row_or_value):
    if isinstance(row_or_value, pd.Series):
        is_flex = bool_from_value(row_or_value.get("is_flex_pick"))
    else:
        is_flex = bool_from_value(row_or_value)
    return "Flex" if is_flex else "Power"


def get_flex_multiplier_choices(row: pd.Series):
    choices = []
    for key in ["flex_full_hit_multiplier", "flex_one_miss_multiplier"]:
        val = safe_float(pd.to_numeric(row.get(key), errors="coerce"), 0.0)
        if val > 0 and val not in choices:
            choices.append(val)
    return choices


def highest_flex_multiplier(row: pd.Series):
    choices = get_flex_multiplier_choices(row)
    if choices:
        return max(choices)
    return safe_float(pd.to_numeric(row.get("multiplier"), errors="coerce"), 0.0)


def flex_outcome_requires_win(outcome: str):
    return str(outcome).strip().lower() == "flex (partial or full win)"


def flex_outcome_requires_loss(outcome: str):
    return str(outcome).strip().lower() == "flex (lost)"


def flex_display_outcome_for_chart(outcome: str):
    normalized = str(outcome).strip().lower()
    if normalized == "cash out":
        return "Cash Out"
    if normalized in {"win", "loss", "push"}:
        return normalized.title()
    return ""


def flex_field_labels(legs_count: int):
    if legs_count == 3:
        return "3-leg Win Multiplier", "2-leg Win Multiplier"
    if legs_count == 4:
        return "4-leg Win Multiplier", "3-leg Win Multiplier"
    return "Full Hit Multiplier", "One-Miss Multiplier"


def combine_confidence_labels(labels):
    order = ["Low", "Low-Medium", "Medium", "Medium-High", "High"]
    mapping = {k: i for i, k in enumerate(order)}
    vals = [mapping.get(str(x), 2) for x in labels]
    avg = round(sum(vals) / len(vals))
    return order[max(0, min(4, avg))], avg + 1


def calculate_money_from_outcome(outcome, bet_size, multiplier, cash_out_price=None):
    outcome = str(outcome or "").strip().lower()
    bet_size = float(pd.to_numeric(bet_size, errors="coerce") or 0)
    multiplier = float(pd.to_numeric(multiplier, errors="coerce") or 0)
    cash_out_price = pd.to_numeric(cash_out_price, errors="coerce")

    if outcome == "win":
        gross_payout = round(bet_size * multiplier, 2)
        amount_won = gross_payout
        amount_lost = 0.0
        net_result = round(gross_payout - bet_size, 2)
    elif outcome == "loss":
        amount_won = 0.0
        amount_lost = round(bet_size, 2)
        net_result = -amount_lost
    elif outcome == "cash out":
        cash_out_value = round(float(cash_out_price or 0), 2)
        amount_won = cash_out_value
        amount_lost = 0.0
        net_result = round(cash_out_value - bet_size, 2)
    else:
        amount_won = 0.0
        amount_lost = 0.0
        net_result = 0.0
    return amount_won, amount_lost, net_result


def compute_net_profit_value(outcome, bet_size, multiplier, cash_out_price=None):
    _, _, net_result = calculate_money_from_outcome(outcome, bet_size, multiplier, cash_out_price)
    return net_result


def calculate_ev_percent(prob_percent: float, multiplier: float):
    if multiplier is None or multiplier <= 0:
        return None
    return (((prob_percent / 100.0) * multiplier) - 1) * 100


def compare_closing_line(side: str, open_line: float, close_line: float):
    if pd.isna(close_line):
        return ""
    if close_line == open_line:
        return "same"
    if side == "OVER":
        return "better" if close_line > open_line else "worse"
    return "better" if close_line < open_line else "worse"


def line_prob_estimate(result: dict, alt_line: float):
    std_guess = max((safe_float(result.get("ceiling_projection"), 0.0, 0.0) - safe_float(result.get("floor_projection"), 0.0, 0.0)) / 1.68, 1.0)
    z = (alt_line - safe_float(result.get("projection"), 0.0, 0.0)) / std_guess
    over_prob = (1 - normal_cdf(z)) * 100
    under_prob = 100 - over_prob
    return round(over_prob, 1), round(under_prob, 1)


def normal_cdf(x):
    import math
    return 0.5 * (1 + math.erf(x / (2 ** 0.5)))


def normal_pdf(x, mean, std):
    import math
    std = max(float(std), 1e-6)
    return (1 / (std * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / std) ** 2)


def render_projected_outcome_distribution(result: dict, line: float, stat_label: str):
    projection = safe_float(result.get("projection"), 0.0, 0.0)
    floor_val = safe_float(result.get("floor_projection"), 0.0, 0.0)
    ceiling_val = safe_float(result.get("ceiling_projection"), 0.0, 0.0)
    std_guess = max((ceiling_val - floor_val) / 1.68, 1.0)

    x_min = min(floor_val, line, projection) - (1.35 * std_guess)
    x_max = max(ceiling_val, line, projection) + (1.35 * std_guess)
    if x_max <= x_min:
        x_min = projection - 3
        x_max = projection + 3

    points = 400
    step = (x_max - x_min) / (points - 1)
    xs = [x_min + (i * step) for i in range(points)]
    ys = [normal_pdf(x, projection, std_guess) for x in xs]

    fig, ax = plt.subplots(figsize=(6.6, 2.35))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    under_x = [x for x in xs if x <= line]
    under_y = [y for x, y in zip(xs, ys) if x <= line]
    over_x = [x for x in xs if x >= line]
    over_y = [y for x, y in zip(xs, ys) if x >= line]

    if under_x:
        ax.fill_between(under_x, under_y, color="#f8d7da", alpha=0.95)
    if over_x:
        ax.fill_between(over_x, over_y, color="#d9ebfa", alpha=0.98)

    ax.plot(xs, ys, color="#17344f", linewidth=2.0)
    ax.axvline(line, color="#0f172a", linestyle=":", linewidth=1.8)

    y_max = max(ys) if ys else 1
    ax.text(
        line,
        y_max * 0.98,
        f"Line: {line:g}",
        ha="center",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cbd5e1", linewidth=0.9),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8, alpha=0.9)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(f"Projected {stat_display_name(stat_label)} Outcome", fontsize=9, fontweight="bold", color="#334155")
    ax.set_title("Projected Outcome Distribution", fontsize=11.5, fontweight="bold", color="#0f172a", loc="left", pad=8)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def highlight_pick_result(row):
    pick_result = row.get("Pick Result", "")
    if pick_result == "over":
        return ["background-color: #d9f2d9"] * len(row)
    if pick_result == "under":
        return ["background-color: #f8d7da"] * len(row)
    if pick_result == "push":
        return ["background-color: #fff3cd"] * len(row)
    return [""] * len(row)


def highlight_saved_entry_row(row):
    outcome = str(row.get("Outcome", "")).strip().lower()
    if outcome == "win":
        return ["background-color: #d9f2d9"] * len(row)
    if outcome == "loss":
        return ["background-color: #f8d7da"] * len(row)
    if outcome == "push":
        return ["background-color: #fff3cd"] * len(row)
    return [""] * len(row)


def _saved_entry_row_style(row):
    outcome = str(row.get("Outcome", "")).strip().lower()
    if outcome == "win":
        return ["background-color: #d9f2d9"] * len(row)
    if outcome == "loss":
        return ["background-color: #f8d7da"] * len(row)
    if outcome in {"cash out", "push"}:
        return ["background-color: #fff3cd"] * len(row)
    return [""] * len(row)


def render_saved_entries_editor(history_df: pd.DataFrame, current_df: pd.DataFrame):
    st.subheader("All Saved Entries")

    if current_df.empty:
        st.info("No picks saved yet.")
        return

    display_df = current_df.copy()
    if "saved_at" in display_df.columns:
        display_df["_sort_dt"] = pd.to_datetime(display_df["saved_at"], errors="coerce")
        display_df = display_df.sort_values("_sort_dt", ascending=False).drop(columns=["_sort_dt"])

    display_df["profit"] = display_df.apply(
        lambda row: compute_net_profit_value(
            row.get("outcome", ""),
            pd.to_numeric(row.get("bet_size"), errors="coerce"),
            pd.to_numeric(row.get("multiplier"), errors="coerce"),
            pd.to_numeric(row.get("cash_out_price"), errors="coerce"),
        ),
        axis=1,
    )
    display_df["display_hit_prob"] = pd.to_numeric(display_df["parlay_hit_prob"], errors="coerce")
    if "selected_prob" in display_df.columns:
        display_df["display_hit_prob"] = display_df["display_hit_prob"].fillna(pd.to_numeric(display_df["selected_prob"], errors="coerce"))
    display_df["type_label"] = display_df.apply(get_pick_type_label, axis=1)

    table_df = pd.DataFrame({
        "Saved": display_df["saved_at"].apply(format_saved_display),
        "Label": display_df["label"].fillna(""),
        "Stat": display_df["stat"].apply(stat_display_name),
        "Type": display_df["type_label"],
        "Hit %": display_df["display_hit_prob"].apply(lambda x: f"{safe_float(x, 0.0):.1f}%" if pd.notna(x) else ""),
        "Bet": display_df["bet_size"].apply(lambda x: f"${safe_float(x, 0.0):,.2f}" if pd.notna(x) else ""),
        "Mult": display_df["multiplier"].apply(lambda x: f"{safe_float(x, 0.0):.1f}x" if pd.notna(x) else ""),
        "Net Profit": display_df["profit"].apply(lambda x: f"${safe_float(x, 0.0):,.2f}" if pd.notna(x) else ""),
        "Confidence": display_df["confidence"].fillna(""),
        "Outcome": display_df["outcome"].apply(flex_display_outcome_for_chart),
    })

    st.dataframe(
        table_df.style.apply(_saved_entry_row_style, axis=1),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Edit Saved Entries", expanded=False):
        pick_options = {
            f"{format_saved_display(row.get('saved_at'))} | {row.get('label', '')}": str(row.get("pick_id"))
            for _, row in display_df.iterrows()
        }
        if not pick_options:
            st.info("No picks available to edit.")
            return

        selected_label = st.selectbox("Select Saved Pick", list(pick_options.keys()), key="edit_saved_pick_select")
        editing_pick_id = pick_options[selected_label]

        matches = history_df[history_df["pick_id"].astype(str) == str(editing_pick_id)].copy()
        if matches.empty:
            return

        pick_row = matches.iloc[0]
        flex_multiplier_choices = get_flex_multiplier_choices(pick_row)
        current_multiplier_value = safe_float(pd.to_numeric(pick_row.get("multiplier"), errors="coerce"), 0.0)
        current_outcome = str(pick_row.get("outcome", "")).strip().lower()

        with st.form(f"edit_saved_pick_form_{editing_pick_id}"):
            left, right = st.columns(2)
            with left:
                edit_bet_size = st.number_input(
                    "Bet Size ($)",
                    min_value=0.0,
                    step=1.0,
                    value=safe_float(pd.to_numeric(pick_row.get("bet_size"), errors="coerce"), 0.0),
                    key=f"edit_bet_size_{editing_pick_id}",
                )
                edit_multiplier = st.number_input(
                    "Multiplier",
                    min_value=0.0,
                    step=0.1,
                    value=current_multiplier_value,
                    key=f"edit_multiplier_{editing_pick_id}",
                )
                outcome_options = ["", "win", "loss", "push", "cash out", "flex (partial or full win)", "flex (lost)"]
                current_outcome_for_select = current_outcome if current_outcome in outcome_options else ""
                edit_outcome = st.selectbox(
                    "Outcome",
                    outcome_options,
                    index=outcome_options.index(current_outcome_for_select),
                    format_func=lambda x: "Select Outcome" if x == "" else ("Cash Out" if x == "cash out" else ("Flex (Partial or Full Win)" if x == "flex (partial or full win)" else ("Flex (Lost)" if x == "flex (lost)" else x.title()))),
                    key=f"edit_outcome_{editing_pick_id}",
                )
            with right:
                edit_notes = st.text_area(
                    "Notes",
                    value="" if pd.isna(pick_row.get("user_notes")) else str(pick_row.get("user_notes")),
                    key=f"edit_notes_{editing_pick_id}",
                )
                edit_cash_out_price = None
                flex_multiplier_mode = None
                selected_flex_multiplier = None
                custom_flex_multiplier = None
                if edit_outcome == "cash out":
                    edit_cash_out_price = st.number_input(
                        "Cash Out Price ($)",
                        min_value=0.0,
                        step=1.0,
                        value=safe_float(pd.to_numeric(pick_row.get("cash_out_price"), errors="coerce"), 0.0),
                        key=f"edit_cashout_{editing_pick_id}",
                    )
                elif edit_outcome == "flex (partial or full win)":
                    if flex_multiplier_choices:
                        flex_choice_labels = [f"{value:.1f}x" for value in flex_multiplier_choices] + ["Custom"]
                        default_choice = 0
                        if current_multiplier_value in flex_multiplier_choices:
                            default_choice = flex_multiplier_choices.index(current_multiplier_value)
                        flex_multiplier_mode = st.selectbox(
                            "Flex Multiplier",
                            flex_choice_labels,
                            index=default_choice,
                            key=f"edit_flex_mult_mode_{editing_pick_id}",
                        )
                        if flex_multiplier_mode == "Custom":
                            custom_flex_multiplier = st.number_input(
                                "Custom Flex Multiplier",
                                min_value=0.0,
                                step=0.1,
                                value=current_multiplier_value if current_multiplier_value > 0 else 0.0,
                                key=f"edit_custom_flex_multiplier_{editing_pick_id}",
                            )
                        else:
                            selected_flex_multiplier = float(flex_multiplier_mode.replace("x", ""))
                    else:
                        custom_flex_multiplier = st.number_input(
                            "Flex Multiplier",
                            min_value=0.0,
                            step=0.1,
                            value=current_multiplier_value if current_multiplier_value > 0 else 0.0,
                            key=f"edit_flex_multiplier_manual_{editing_pick_id}",
                        )

            a, b, c = st.columns(3)
            update_clicked = a.form_submit_button("Update Entry", use_container_width=True)
            delete_clicked = b.form_submit_button("Delete Entry", use_container_width=True)
            cancel_clicked = c.form_submit_button("Close", use_container_width=True)

        if update_clicked:
            normalized_outcome = edit_outcome
            final_multiplier = safe_float(edit_multiplier, 0.0)
            if edit_outcome == "flex (partial or full win)":
                normalized_outcome = "win"
                if selected_flex_multiplier is not None:
                    final_multiplier = safe_float(selected_flex_multiplier, 0.0)
                else:
                    final_multiplier = safe_float(custom_flex_multiplier, 0.0)
                if final_multiplier <= 0:
                    st.error("Enter a valid Flex Multiplier before saving this edit.")
                    return
            elif edit_outcome == "flex (lost)":
                normalized_outcome = "loss"
                final_multiplier = highest_flex_multiplier(pick_row)
            updates = {
                "bet_size": safe_float(edit_bet_size, 0.0),
                "multiplier": final_multiplier,
                "user_notes": edit_notes.strip(),
                "outcome": normalized_outcome,
                "cash_out_price": safe_float(edit_cash_out_price, 0.0) if edit_outcome == "cash out" and edit_cash_out_price is not None else "",
            }
            if edit_outcome in {"flex (partial or full win)", "flex (lost)"}:
                updates["is_flex_pick"] = True
            update_pick_in_history(editing_pick_id, updates)
            st.rerun()

        if delete_clicked:
            delete_pick_from_history(editing_pick_id)
            st.rerun()

        if cancel_clicked:
            st.rerun()


def render_secondary_card(label: str, value: str, explanation: str | None = None):
    st.markdown(
        f"""
        <div class="secondary-card">
            <div class="label" style="color:#122235 !important;">{label}</div>
            <div class="value" style="color:#122235 !important;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_primary_card(label: str, value: str, explanation: str | None = None):
    st.markdown(
        f"""
        <div class="primary-card">
            <div class="label" style="color:#122235 !important;">{label}</div>
            <div class="value" style="color:#122235 !important;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, show_explanations: bool = False):
    st.markdown("<div style='height:0.18rem;'></div>", unsafe_allow_html=True)
    st.subheader(title)
    st.markdown("<div style='height:0.02rem;'></div>", unsafe_allow_html=True)
    render_thin_section_line()


def render_small_explanation(text: str):
    return None


def score_style(score: float, positive_good: bool = True):
    if pd.isna(score):
        return "#f3f4f6", "N/A"
    if positive_good:
        if score >= 80:
            return COLORS["strong"], "Excellent"
        if score >= 65:
            return "#e8f7ec", "Strong"
        if score >= 45:
            return COLORS["neutral"], "Neutral"
        if score >= 25:
            return "#fde7e7", "Weak"
        return "#f7d4d4", "Poor"
    else:
        if score <= 14:
            return COLORS["strong"], "Excellent"
        if score <= 24:
            return "#e8f7ec", "Strong"
        if score <= 39:
            return COLORS["neutral"], "Neutral"
        if score <= 59:
            return "#fde7e7", "Risky"
        return "#f7d4d4", "Very Risky"


def render_score_card(title: str, value, positive_good: bool = True, explanation: str | None = None):
    bg, label = score_style(value, positive_good=positive_good)
    if explanation:
        render_small_explanation(explanation)
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border-radius:16px;
            padding:14px 16px;
            min-height:110px;
            box-shadow:0 3px 12px rgba(0,0,0,0.05);
            border:1px solid rgba(0,0,0,0.06);
        ">
            <div style="font-size:0.92rem;font-weight:700;color:#122235;">{title}</div>
            <div style="font-size:1.95rem;font-weight:800;margin-top:4px;color:#122235;">{value}</div>
            <div style="font-size:0.9rem;opacity:0.8;margin-top:6px;color:#122235;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )




def format_model_method_label(model_method: str | None):
    value = str(model_method or "rules_only").strip().lower()
    if value == "rules_only":
        return "Rules Only"
    if value.startswith("blend_"):
        label = value.replace("blend_", "").replace("_", " ").title()
        return f"ML + Rules ({label})"
    return value.replace("_", " ").title()




def get_analysis_mode_key(scope: str) -> str:
    return f"{scope}_analysis_mode"


def get_analysis_mode(scope: str, default: str = "Deep Mode") -> str:
    key = get_analysis_mode_key(scope)
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def infer_stat_column_from_result(result: dict, stat_key: str | None = None) -> str | None:
    stat_key = str(stat_key or result.get("stat") or "").lower()
    mapping = {
        "points": "PTS",
        "rebounds": "REB",
        "assists": "AST",
        "fgm": "FGM",
        "fga": "FGA",
        "2pm": "2PM",
        "2pa": "2PA",
        "3pm": "FG3M",
        "3pa": "FG3A",
        "steals": "STL",
        "blocks": "BLK",
        "stocks": "STOCKS",
        "turnovers": "TOV",
        "fouls": "PF",
        "ftm": "FTM",
        "fta": "FTA",
        "oreb": "OREB",
        "dreb": "DREB",
        "pra": "PRA",
        "pr": "PR",
        "pa": "PA",
        "ra": "RA",
        "fantasy": "FANTASY",
    }
    sample_df = result.get("recent_games_df")
    cand = mapping.get(stat_key)
    if isinstance(sample_df, pd.DataFrame) and not sample_df.empty:
        cols = {c.upper(): c for c in sample_df.columns}
        if cand and cand.upper() in cols:
            return cols[cand.upper()]
        # derive common combos if missing
        if stat_key in {"pra","pr","pa","ra","stocks"}:
            if stat_key=="stocks" and {"STL","BLK"}.issubset(cols):
                return "STOCKS"
            if stat_key=="pra" and {"PTS","REB","AST"}.issubset(cols):
                return "PRA"
            if stat_key=="pr" and {"PTS","REB"}.issubset(cols):
                return "PR"
            if stat_key=="pa" and {"PTS","AST"}.issubset(cols):
                return "PA"
            if stat_key=="ra" and {"REB","AST"}.issubset(cols):
                return "RA"
        # old heuristic
        if "Pick Result" in sample_df.columns:
            idx = list(sample_df.columns).index("Pick Result")
            if idx > 0:
                prev_col = list(sample_df.columns)[idx - 1]
                if prev_col not in {"Game Date","Matchup","Pick Result"}:
                    return prev_col
    return cand


def prepare_recent_stat_column(sample_df: pd.DataFrame, stat_key: str) -> tuple[pd.DataFrame, str | None]:
    if not isinstance(sample_df, pd.DataFrame) or sample_df.empty:
        return sample_df, None
    df = sample_df.copy()
    cols = {c.upper(): c for c in df.columns}
    stat_key = str(stat_key or "").lower()
    if stat_key == "stocks" and {"STL", "BLK"}.issubset(cols):
        df["STOCKS"] = pd.to_numeric(df[cols["STL"]], errors="coerce").fillna(0) + pd.to_numeric(df[cols["BLK"]], errors="coerce").fillna(0)
        return df, "STOCKS"
    if stat_key == "pra" and {"PTS", "REB", "AST"}.issubset(cols):
        df["PRA"] = pd.to_numeric(df[cols["PTS"]], errors="coerce").fillna(0) + pd.to_numeric(df[cols["REB"]], errors="coerce").fillna(0) + pd.to_numeric(df[cols["AST"]], errors="coerce").fillna(0)
        return df, "PRA"
    if stat_key == "pr" and {"PTS", "REB"}.issubset(cols):
        df["PR"] = pd.to_numeric(df[cols["PTS"]], errors="coerce").fillna(0) + pd.to_numeric(df[cols["REB"]], errors="coerce").fillna(0)
        return df, "PR"
    if stat_key == "pa" and {"PTS", "AST"}.issubset(cols):
        df["PA"] = pd.to_numeric(df[cols["PTS"]], errors="coerce").fillna(0) + pd.to_numeric(df[cols["AST"]], errors="coerce").fillna(0)
        return df, "PA"
    if stat_key == "ra" and {"REB", "AST"}.issubset(cols):
        df["RA"] = pd.to_numeric(df[cols["REB"]], errors="coerce").fillna(0) + pd.to_numeric(df[cols["AST"]], errors="coerce").fillna(0)
        return df, "RA"
    return df, infer_stat_column_from_result({"recent_games_df": df}, stat_key)


def estimate_enhanced_minutes(result: dict) -> dict:
    sample_df = result.get("recent_games_df")
    if not isinstance(sample_df, pd.DataFrame) or sample_df.empty:
        result["enhanced_projected_minutes"] = safe_float(result.get("projected_minutes"), 0.0)
        result["minutes_risk_label"] = "Unknown"
        result["rotation_role_label"] = "Unknown"
        return result

    df = sample_df.copy()
    cols = {c.upper(): c for c in df.columns}
    min_col = cols.get("MIN")
    if min_col is None:
        result["enhanced_projected_minutes"] = safe_float(result.get("projected_minutes"), 0.0)
        result["minutes_risk_label"] = "Unknown"
        result["rotation_role_label"] = "Unknown"
        return result

    minutes = pd.to_numeric(df[min_col], errors="coerce").dropna()
    if minutes.empty:
        result["enhanced_projected_minutes"] = safe_float(result.get("projected_minutes"), 0.0)
        result["minutes_risk_label"] = "Unknown"
        result["rotation_role_label"] = "Unknown"
        return result

    recent5 = float(minutes.head(min(5, len(minutes))).mean())
    recent10 = float(minutes.head(min(10, len(minutes))).mean())
    season_mean = float(minutes.mean())
    base = safe_float(result.get("projected_minutes"), recent5 or season_mean or 0.0)
    blended = 0.55 * recent5 + 0.25 * recent10 + 0.20 * season_mean

    rest_days = safe_float(result.get("rest_days"), 2.0)
    if rest_days <= 0:
        blended *= 0.965
    elif rest_days >= 2:
        blended *= 1.01

    pf_col = cols.get("PF")
    foul_rate = float(pd.to_numeric(df[pf_col], errors="coerce").head(min(10, len(df))).mean()) if pf_col else 0.0
    if foul_rate >= 3.8:
        blended *= 0.97

    min_std = float(minutes.head(min(10, len(minutes))).std() or 0.0)
    role_label = "Starter"
    if recent5 < 22:
        role_label = "Bench / limited role"
    elif recent5 < 30:
        role_label = "Rotation contributor"

    minute_gap_penalty = 1.0
    if "Game Date" in df.columns:
        dates = pd.to_datetime(df["Game Date"], errors="coerce").dropna().sort_values(ascending=False)
        if len(dates) >= 2:
            median_gap = dates.diff(-1).dt.days.abs().median()
            if pd.notna(median_gap) and median_gap > 4:
                minute_gap_penalty = 0.985
    blended *= minute_gap_penalty

    enhanced = max(8.0, min(42.0, (0.45 * base) + (0.55 * blended)))
    result["enhanced_projected_minutes"] = round(enhanced, 1)
    if min_std >= 7:
        minutes_risk = "High"
    elif min_std >= 4:
        minutes_risk = "Moderate"
    else:
        minutes_risk = "Low"
    if role_label == "Bench / limited role" and minutes_risk == "Low":
        minutes_risk = "Moderate"

    result["minutes_risk_label"] = minutes_risk
    result["rotation_role_label"] = role_label
    result["minutes_model_notes"] = {
        "recent5_minutes": round(recent5, 1),
        "recent10_minutes": round(recent10, 1),
        "season_minutes": round(season_mean, 1),
        "minutes_std_10": round(min_std, 1),
        "foul_rate_10": round(foul_rate, 1),
        "rest_days": rest_days,
    }
    return result


def apply_matchup_archetype_adjustment(result: dict, stat_key: str | None = None) -> dict:
    stat_key = str(stat_key or result.get("stat") or "").lower()
    pace_proxy = safe_float(result.get("pace_proxy"), 100.0)
    opp_allow = safe_float(result.get("opponent_allowance_proxy"), 1.0)
    matchup_mult = safe_float(result.get("matchup_multiplier"), 1.0)
    extra = 0.0
    archetype = "neutral"

    if stat_key in {"assists", "pa", "ra", "pra"}:
        archetype = "playmaking"
        extra += max(-0.03, min(0.03, (pace_proxy - 100.0) / 250.0))
        extra += max(-0.03, min(0.03, (opp_allow - 1.0) / 4.0))
    elif stat_key in {"rebounds", "oreb", "dreb", "ra", "pr", "pra"}:
        archetype = "rebounding"
        extra += max(-0.035, min(0.035, (opp_allow - 1.0) / 3.5))
    elif stat_key in {"3pm", "3pa"}:
        archetype = "perimeter volume"
        extra += max(-0.04, min(0.04, (pace_proxy - 100.0) / 220.0))
    elif stat_key in {"points", "fgm", "fga", "2pm", "2pa", "ftm", "fta"}:
        archetype = "scoring"
        extra += max(-0.03, min(0.03, (opp_allow - 1.0) / 4.5))

    total_mult = matchup_mult * (1.0 + extra)
    result["stat_archetype"] = archetype
    result["matchup_multiplier_enhanced"] = round(total_mult, 4)
    return result


def apply_app_layer_projection_upgrade(result: dict, stat_key: str | None = None) -> dict:
    if not isinstance(result, dict):
        return result

    if result.get("_app_layer_projection_locked"):
        return result

    result = normalize_projection_fields(result)
    result = estimate_enhanced_minutes(result)
    result = apply_matchup_archetype_adjustment(result, stat_key)

    base_projection = safe_float(result.get("_base_projection_for_upgrade", result.get("projection")), None)
    base_rules_projection = safe_float(result.get("_base_rules_projection_for_upgrade", result.get("rules_projection")), None)
    base_floor_projection = safe_float(result.get("_base_floor_projection_for_upgrade", result.get("floor_projection")), None)
    base_ceiling_projection = safe_float(result.get("_base_ceiling_projection_for_upgrade", result.get("ceiling_projection")), None)

    if base_projection is not None:
        result["_base_projection_for_upgrade"] = base_projection
    if base_rules_projection is not None:
        result["_base_rules_projection_for_upgrade"] = base_rules_projection
    if base_floor_projection is not None:
        result["_base_floor_projection_for_upgrade"] = base_floor_projection
    if base_ceiling_projection is not None:
        result["_base_ceiling_projection_for_upgrade"] = base_ceiling_projection

    projection = base_projection
    base_minutes = safe_float(result.get("projected_minutes"), None)
    enh_minutes = safe_float(result.get("enhanced_projected_minutes"), None)

    if projection is not None and base_minutes and enh_minutes:
        minute_factor = enh_minutes / max(base_minutes, 1.0)
        minute_factor = max(0.90, min(1.10, minute_factor))
        matchup_factor = safe_float(result.get("matchup_multiplier_enhanced"), safe_float(result.get("matchup_multiplier"), 1.0))
        raw_matchup = safe_float(result.get("matchup_multiplier"), 1.0)
        matchup_relative = matchup_factor / max(raw_matchup, 0.01)
        matchup_relative = max(0.95, min(1.05, matchup_relative))
        upgraded = projection * minute_factor * matchup_relative
        result["projection"] = round(upgraded, 2)

        if base_rules_projection is not None:
            result["rules_projection"] = round(base_rules_projection * minute_factor * matchup_relative, 2)

        line_value = safe_float(result.get("line"), 0.0)
        result["edge_vs_line"] = round(result["projection"] - line_value, 2)

        std_guess = max(
            safe_float(result.get("ml_residual_std"), 0.0),
            (safe_float(base_ceiling_projection, result["projection"]) - safe_float(base_floor_projection, result["projection"])) / 2.0,
            1.0,
        )
        over_prob = (1 - normal_cdf((line_value - result["projection"]) / max(std_guess, 1.0))) * 100
        over_prob = max(1.0, min(99.0, over_prob))
        push_prob = safe_float(result.get("push_prob"), 0.0)
        if not _line_has_push(line_value):
            push_prob = 0.0
        under_prob = max(0.0, 100.0 - over_prob - push_prob)
        result["over_prob"] = round(over_prob, 1)
        result["under_prob"] = round(under_prob, 1)
        result["lean"] = "OVER" if over_prob >= under_prob else "UNDER"

        result["median_projection"] = round(result["projection"], 2)
        floor_ref = safe_float(base_floor_projection, result["projection"])
        ceil_ref = safe_float(base_ceiling_projection, result["projection"])
        result["floor_projection"] = round(min(result["projection"] - std_guess * 0.85, floor_ref), 2)
        result["ceiling_projection"] = round(max(result["projection"] + std_guess * 0.85, ceil_ref), 2)

    result["_app_layer_projection_locked"] = True
    return result


def build_play_summary(result: dict, pick_meta: dict | None = None) -> dict:
    line_val = safe_float((pick_meta or {}).get("line", result.get("line")), 0.0)
    proj = safe_float(result.get("projection"), 0.0)
    edge = round(proj - line_val, 2)
    best_side = result.get("lean", "N/A")
    confidence = str(result.get("confidence", "N/A"))
    minutes_risk = str(result.get("minutes_risk_label", result.get("minutes_risk_score", "N/A")))
    ml_projection = safe_float(result.get("ml_projection"), None)
    rules_projection = safe_float(result.get("rules_projection"), None)
    ml_changed = "No"
    if ml_projection is not None and rules_projection is not None:
        ml_changed = "Yes" if abs(ml_projection - rules_projection) >= max(1.0, 0.04 * max(abs(rules_projection), 1.0)) else "No"

    return {
        "Best Side": best_side,
        "Final Projection": f"{proj:.2f}",
        "Edge vs Line": f"{edge:+.2f}",
        "Confidence Tier": confidence,
        "Minutes Risk": minutes_risk,
        "ML Changed Result": ml_changed,
    }


def render_play_summary_strip(result: dict, pick_meta: dict | None = None):
    summary = build_play_summary(result, pick_meta)
    st.markdown("#### Play Summary")
    cols = st.columns(len(summary))
    for col, (label, value) in zip(cols, summary.items()):
        with col:
            st.markdown(
                f"""
                <div style="background:#10213a;border:1px solid rgba(255,255,255,0.12);border-radius:14px;padding:0.7rem 0.75rem;">
                    <div style="font-size:0.76rem;opacity:0.82;">{label}</div>
                    <div style="font-size:1.02rem;font-weight:700;color:#ffffff;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def get_top_risk_reason(result: dict) -> str:
    risks = [str(x) for x in (result.get("risks") or []) if str(x).strip()]
    risk_score = safe_float(result.get("minutes_risk_score"), 0.0)
    vol_score = safe_float(result.get("volatility_score"), 0.0)
    pace_proxy = safe_float(result.get("pace_proxy"), 100.0)
    if risks:
        for item in risks:
            low = item.lower()
            if "blowout" in low:
                return "Blowout risk"
            if "minute" in low or "rotation" in low:
                return "Minutes uncertainty"
    if risk_score >= 7:
        return "Minutes uncertainty"
    if vol_score >= 7:
        return "Volatile stat type"
    if pace_proxy < 97:
        return "Opponent pace suppression"
    role = str(result.get("rotation_role_label","")).lower()
    if "bench" in role:
        return "Role instability"
    return "Monitor normal variance"


def render_top_risk_box(result: dict):
    risk = get_top_risk_reason(result)
    st.markdown(
        f"""
        <div style="background:#17344f;border-left:4px solid #7BAFD4;border-radius:12px;padding:0.85rem 1rem;margin:0.25rem 0 0.8rem 0;">
            <div style="font-size:0.76rem;opacity:0.78;">Most Important Risk</div>
            <div style="font-size:1.02rem;font-weight:700;color:#ffffff;">{risk}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_comparable_spots(result: dict, pick_meta: dict | None = None) -> dict:
    stat_key = (pick_meta or {}).get("stat", result.get("stat"))
    sample_df = result.get("recent_games_df")
    line_val = safe_float((pick_meta or {}).get("line", result.get("line")), 0.0)
    if not isinstance(sample_df, pd.DataFrame) or sample_df.empty:
        return {"rate": "N/A", "games": 0, "label": "Comparable spots unavailable"}

    df, stat_col = prepare_recent_stat_column(sample_df, str(stat_key))
    if not stat_col or stat_col not in df.columns:
        return {"rate": "N/A", "games": 0, "label": "Comparable spots unavailable"}

    cols = {c.upper(): c for c in df.columns}
    work = df.copy()
    if "MIN" in cols:
        mins = pd.to_numeric(work[cols["MIN"]], errors="coerce")
        target_min = safe_float(result.get("enhanced_projected_minutes"), safe_float(result.get("projected_minutes"), mins.mean()))
        work = work[(mins >= target_min - 4) & (mins <= target_min + 4)].copy()
    if "Home/Away" in work.columns and result.get("home_away"):
        home_away = str(result.get("home_away")).strip().lower()
        if home_away in {"home","away"}:
            work = work[work["Home/Away"].astype(str).str.strip().str.lower() == home_away]
    if work.empty:
        work = df.copy()

    series = pd.to_numeric(work[stat_col], errors="coerce").dropna()
    if series.empty:
        return {"rate": "N/A", "games": 0, "label": "Comparable spots unavailable"}

    hit_rate = float((series > line_val).mean() * 100)
    return {
        "rate": f"{hit_rate:.1f}%",
        "games": int(series.shape[0]),
        "label": f"In similar minutes/matchup situations this player cleared this line {hit_rate:.1f}% of the time",
    }


def render_comparable_spots(result: dict, pick_meta: dict | None = None):
    comp = compute_comparable_spots(result, pick_meta)
    st.markdown("#### Comparable Picks")
    if comp["games"] == 0:
        st.caption(comp["label"])
    else:
        st.markdown(
            f"""
            <div style="background:#10213a;border:1px solid rgba(255,255,255,0.10);border-radius:12px;padding:0.8rem 0.95rem;">
                <div style="font-size:0.98rem;color:#ffffff;">{comp["label"]}</div>
                <div style="font-size:0.8rem;color:#c9d6e5;margin-top:0.25rem;">Sample size: {comp["games"]} games</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_quick_reason_list(result: dict) -> list[str]:
    out = []
    edge = safe_float(result.get("edge_vs_line"), 0.0)
    if abs(edge) >= 2:
        out.append(f"Projection is {edge:+.2f} versus the line.")
    season_avg = safe_float(result.get("season_average"), None)
    if season_avg is not None:
        out.append(f"Season average is {season_avg:.2f}.")
    why = result.get("why_lean") or []
    for item in why[:2]:
        out.append(str(item))
    return out[:3]


def render_quick_mode_single(result: dict, pick_meta: dict | None = None):
    render_play_summary_strip(result, pick_meta)
    render_top_risk_box(result)
    st.markdown("#### Quick Analysis")
    for item in get_quick_reason_list(result):
        st.write(f"• {item}")


def get_parlay_leg_summary(leg_result: dict, leg_meta: dict) -> str:
    proj = safe_float(leg_result.get("projection"), 0.0)
    line = safe_float(leg_meta.get("line"), 0.0)
    diff = proj - line
    side = str(leg_meta.get("side","OVER")).upper()
    confidence = str(leg_result.get("confidence",""))
    if side == "UNDER":
        diff = line - proj
    if diff >= 3:
        strength = "Strong"
    elif diff >= 1.25:
        strength = "Solid"
    else:
        strength = "Thin"
    if strength == "Thin":
        reason = "mostly driven by recent trend"
    else:
        reason = f"projection is {abs(proj-line):.1f} {'below' if side=='UNDER' else 'above'} the line"
    return f"{strength} {side.lower()} because {reason}."


def format_model_summary(result):
    result = normalize_projection_fields(result)

    rules_projection = safe_float(result.get("rules_projection"), None)
    ml_projection = safe_float(result.get("ml_projection"), None)
    blend_weight = safe_float(result.get("ml_blend_weight"), 0.0)

    rules_text = f"{rules_projection:.2f}" if rules_projection is not None else "N/A"
    ml_text = f"{ml_projection:.2f}" if ml_projection is not None else "N/A"

    if ml_projection is not None:
        return f"Engine Projections | Rules: {rules_text} + ML: {ml_text} | {blend_weight * 100:.0f}% ML"

    return f"Engine Projections | Rules: {rules_text}"


def render_model_details(result):
    result = normalize_projection_fields(result)
    with st.expander("Projection Method Details", expanded=False):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            projection_val = safe_float(result.get("projection"), None)
            st.metric("Final Projection", f"{projection_val:.2f}" if projection_val is not None else "N/A")
        with r1c2:
            rules_val = safe_float(result.get("rules_projection"), None)
            st.metric("Rules Projection", f"{rules_val:.2f}" if rules_val is not None else "N/A")
        with r1c3:
            ml_val = safe_float(result.get("ml_projection"), None)
            st.metric("ML Projection", f"{ml_val:.2f}" if ml_val is not None else "N/A")
        with r1c4:
            st.metric("ML Blend %", f"{safe_float(result.get('ml_blend_weight'), 0.0) * 100:.0f}%")

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            st.metric("Train Rows", result.get("ml_train_rows", 0))
        with r2c2:
            st.metric("Train R²", result.get("ml_train_r2") if result.get("ml_train_r2") is not None else "N/A")
        with r2c3:
            st.metric("Residual Std", result.get("ml_residual_std") if result.get("ml_residual_std") is not None else "N/A")
        with r2c4:
            st.metric("Validation MAE", result.get("ml_validation_mae") if result.get("ml_validation_mae") is not None else "N/A")

        st.metric("Method", format_model_method_label(result.get("model_method")))

        if result.get("ml_debug_note"):
            st.caption(result.get("ml_debug_note"))


def render_why_this_lean(result):
    render_section_header("Why This Lean")
    bullets = result.get("why_lean") or []
    if not bullets:
        st.write("• No explanation available.")
        return
    for item in bullets:
        st.write(f"• {item}")

def clear_single_inputs():
    defaults = {
        "single_player": None,
        "single_stat": "points",
        "single_line": 24.5,
        "single_opponent": None,
        "single_notes": "",
        "single_ev_multiplier": 0.0,
        "_single_last_auto_player": None,
        "_single_auto_opponent": None,
    }
    for k, v in defaults.items():
        st.session_state[k] = v
    st.session_state["last_analysis"] = None
    st.session_state["last_pick_meta"] = None


def get_today_single_history_key() -> str:
    user_id = get_current_user_id() or "guest"
    today_local = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    return f"single_prop_history::{user_id}::{today_local}"


def get_today_single_history():
    return st.session_state.get(get_today_single_history_key(), [])


def save_single_history_entry(entry: dict):
    history_key = get_today_single_history_key()
    history = list(st.session_state.get(history_key, []))
    normalized = dict(entry)
    signature = json.dumps(
        {
            "player_display": normalized.get("player_display"),
            "stat": normalized.get("stat"),
            "line": safe_float(normalized.get("line"), 0.0),
            "opponent_display": normalized.get("opponent_display"),
            "notes": normalized.get("notes", ""),
            "ev_multiplier": safe_float(normalized.get("ev_multiplier"), 0.0),
        },
        sort_keys=True,
        default=str,
    )
    normalized["signature"] = signature
    history = [item for item in history if item.get("signature") != signature]
    history.insert(0, normalized)
    st.session_state[history_key] = history[:20]


def load_single_history_entry(entry: dict):
    st.session_state["pending_single_history_entry"] = dict(entry)


def render_single_history_panel():
    history = get_today_single_history()
    if not history:
        st.caption("No single-prop history saved yet today for this account.")
        return

    st.caption("Today's History")
    for idx, item in enumerate(history):
        analyzed_at = item.get("analyzed_at", "")
        meta = f"{item.get('player_name', item.get('player_display', 'Unknown'))} | {stat_display_name(item.get('stat', 'points'))} | {safe_float(item.get('line'), 0.0):g}"
        if item.get("opponent_display"):
            meta += f" vs {item.get('opponent_display')}"
        if analyzed_at:
            meta += f" • {analyzed_at}"
        if st.button(meta, key=f"single_history_load_{idx}", use_container_width=True):
            load_single_history_entry(item)
            st.rerun()


def clear_parlay_inputs(section_key: str, legs_count: int):
    for i in range(legs_count):
        st.session_state[f"{section_key}_player_{i}"] = None
        st.session_state[f"{section_key}_stat_{i}"] = "points"
        st.session_state[f"{section_key}_side_{i}"] = "OVER"
        st.session_state[f"{section_key}_line_{i}"] = 10.5
        st.session_state[f"{section_key}_opp_{i}"] = None
        st.session_state[f"_{section_key}_last_auto_player_{i}"] = None
        st.session_state[f"_{section_key}_auto_opponent_{i}"] = None
    st.session_state[f"{section_key}_notes"] = ""
    st.session_state[f"{section_key}_ev_multiplier"] = 0.0
    st.session_state[section_key] = None



def get_active_top_nav_group() -> str:
    current = st.session_state.get("propify_force_tab")
    if current in {"Single Prop Analyzer"}:
        return "Analyze"
    if current in {"2-Leg Parlay", "3-Leg Parlay", "4-Leg Parlay"}:
        return "Parlays"
    return "Analyze"


def render_global_mode_dropdown():
    active_group = get_active_top_nav_group()
    if active_group == "Analyze":
        scope = "single"
    else:
        current = st.session_state.get("propify_force_tab") or "2-Leg Parlay"
        if current == "3-Leg Parlay":
            scope = "parlay_3"
        elif current == "4-Leg Parlay":
            scope = "parlay_4"
        else:
            scope = "parlay_2"

    current_mode = get_analysis_mode(scope)
    selected = st.selectbox(
        "Mode",
        ["Deep Mode", "Quick Mode"],
        index=0 if current_mode == "Deep Mode" else 1,
        key=f"global_mode_picker_{scope}",
    )
    st.session_state[get_analysis_mode_key(scope)] = selected


def render_header():
    left, right = st.columns([8.3, 1.7])
    with left:
        st.markdown(
            f"""
            <div class="propify-topbar">
                <img src="data:image/png;base64,{PROPIFY_ICON_BASE64}"
                     alt="Propify" />
                <div class="propify-topbar-engine">Propify AI Engine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div class='propify-global-mode'></div>", unsafe_allow_html=True)
        render_global_mode_dropdown()

def render_section_title(title: str):
    st.markdown(
        f"""
        <div class="propify-section-title">
            <h1>{title}</h1>
            <div class="propify-section-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def align_section_inputs():
    st.markdown("<div style='height:0.15rem;'></div>", unsafe_allow_html=True)


def render_thin_section_line():
    st.markdown(
        "<div style='height:2px; width:100%; background:rgba(255,255,255,0.22); margin:0.08rem 0 1.15rem 0; border-radius:999px;'></div>",
        unsafe_allow_html=True,
    )


def render_home_screen():
    st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
    left, center, right = st.columns([1.1, 2.8, 1.1])
    with center:
        st.markdown(
            f"""
            <div style="
                border-radius: 28px;
                box-shadow: 0 14px 34px rgba(15, 23, 42, 0.10);
                padding: 28px 20px 20px 20px;
                text-align: center;
                margin: 0 auto;
                background: transparent;
                border: none;
            ">
                <img src="data:image/png;base64,{PROPIFY_TRANSPARENT_LOGO_BASE64}"
                     alt="Propify Sports Betting Engine Logo"
                     style="
                        width: min(100%, 520px);
                        height: auto;
                        display: block;
                        margin: 0 auto;
                        filter: drop-shadow(0px 8px 18px rgba(18, 34, 53, 0.08));
                        background: transparent;
                     " />
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 1.0rem;'></div>", unsafe_allow_html=True)
        button_cols = st.columns([1.15, 1.7, 1.15])
        with button_cols[1]:
            if st.button("Enter Propify", key="start_analyzing_props_btn", use_container_width=True):
                st.session_state["app_view"] = "main"
                st.session_state["current_section"] = "Overview"
                st.session_state["nav_open"] = False
                st.session_state["propify_force_tab"] = None
                st.rerun()
        st.markdown(
            f"""
            <div style="
                text-align:center;
                margin-top: 0.85rem;
            ">
                <img src="data:image/png;base64,{DH_LOGO_BASE64}"
                     alt="DH Logo"
                     style="
                        width: 84px;
                        height: auto;
                        display: inline-block;
                        opacity: 0.96;
                        background: transparent;
                     " />
            </div>
            """,
            unsafe_allow_html=True,
        )


NAV_SECTION_ORDER = [
    "Overview",
    "Single Prop Analyzer",
    "2-Leg Parlay",
    "3-Leg Parlay",
    "4-Leg Parlay",
    "Pick Tracker",
    "How to Use Propify",
    "Metric Guide",
    "FAQs",
    "Logout",
]


def render_top_controls():
    cols = st.columns([0.06, 0.72, 0.22])
    with cols[0]:
        st.markdown("<div style='height:0.0rem;'></div>", unsafe_allow_html=True)
        if st.button("☰", key="nav_toggle_btn", use_container_width=True):
            st.session_state["nav_open"] = not st.session_state.get("nav_open", False)
            st.rerun()


def render_nav_sidebar():
    current = st.session_state.get("current_section", "Overview")
    links_html = []
    for label in NAV_SECTION_ORDER:
        href_label = label.replace(" ", "%20")
        is_active = label == current
        bg = "#10213a" if is_active else "#122235"
        border = "2px solid rgba(255,255,255,0.78)" if is_active else "1px solid rgba(255,255,255,0.12)"
        shadow = "0 6px 14px rgba(0,0,0,0.12)" if is_active else "0 4px 10px rgba(0,0,0,0.08)"
        links_html.append(
            f'<a href="?section={href_label}" style="display:block;width:100%;box-sizing:border-box;text-decoration:none;background:{bg};color:#ffffff;border-radius:14px;padding:0.95rem 0.9rem;text-align:center;font-weight:700;margin:0 0 0.75rem 0;border:{border};box-shadow:{shadow};">{label}</a>'
        )

    st.markdown(
        f"""
        <style>
        .propify-sidebar-bg {{
            background:{TARHEEL_BLUE};
            border-radius:22px;
            min-height: calc(100vh - 5.9rem);
            padding:0.95rem 0.85rem 0.95rem 0.85rem;
            position:relative;
            margin-top:0.02rem;
        }}
        .propify-sidebar-separator {{
            position:absolute;
            top:0.18rem;
            right:-0.95rem;
            width:4px;
            height:calc(100vh - 6.6rem);
            background:#17344f;
            border-radius:999px;
        }}
        </style>
        <div class="propify-sidebar-bg">
            <div class="propify-sidebar-separator"></div>
            {''.join(links_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_page():
    forced_tab = st.session_state.get("propify_force_tab")
    forced_tab_js = json.dumps(forced_tab) if forced_tab else "null"
    if not st.session_state.get("nav_open", False):
        components.html(
            f"""
            <script>
            (function() {{
              const KEY = "propify_active_tab";
              const forced = {forced_tab_js};
              function bindAndRestore() {{
                const tabs = Array.from(parent.document.querySelectorAll('[data-baseweb="tab"]'));
                if (!tabs.length) {{ setTimeout(bindAndRestore, 150); return; }}
                tabs.forEach((tab) => {{
                  if (!tab.dataset.propifyBound) {{
                    tab.dataset.propifyBound = "1";
                    tab.addEventListener("click", () => {{
                      try {{ parent.localStorage.setItem(KEY, tab.innerText.trim()); }} catch (e) {{}}
                    }});
                    tab.addEventListener("click", () => {{
                      const txt = (tab.innerText || "").trim();
                      if (["Analyze","Parlays","Tracking","Learn","Account","Single Prop Analyzer","2-Leg Parlay","3-Leg Parlay","4-Leg Parlay"].includes(txt)) {{
                        try {{ parent.localStorage.setItem("propify_active_tab", txt); }} catch (e) {{}}
                      }}
                    }});
                  }}
                }});
                let wanted = forced;
                if (!wanted) {{
                  try {{ wanted = parent.localStorage.getItem(KEY); }} catch (e) {{}}
                }} else {{
                  try {{ parent.localStorage.setItem(KEY, wanted); }} catch (e) {{}}
                }}
                if (!wanted) {{ return; }}
                const active = tabs.find(t => t.getAttribute("aria-selected") === "true");
                const target = tabs.find(t => (t.innerText || "").trim() === wanted);
                if (target && active && (active.innerText || "").trim() !== wanted) {{
                  setTimeout(() => target.click(), 0);
                }}
              }}
              bindAndRestore();
            }})();
            </script>
            """,
            height=0,
        )
    st.session_state["propify_force_tab"] = None

    components.html(
        """
        <script>
        (function() {
          function styleTabLists() {
            const lists = Array.from(parent.document.querySelectorAll('[data-baseweb="tab-list"]'));
            if (!lists.length) { setTimeout(styleTabLists, 150); return; }
            lists.forEach((list, idx) => {
              if (idx === 0) {
                list.style.paddingLeft = '86px';
                list.style.marginTop = '-3.55rem';
                list.style.marginBottom = '0.75rem';
                list.style.columnGap = '2.55rem';
                list.style.gap = '2.55rem';
              } else {
                list.style.paddingLeft = '0';
                list.style.marginTop = '0';
                list.style.marginBottom = '1.15rem';
                list.style.columnGap = '1.25rem';
                list.style.gap = '1.25rem';
              }
            });
          }
          styleTabLists();
        })();
        </script>
        """,
        height=0,
    )

    tab_analyze, tab_parlays, tab_tracking, tab_learn, tab_account = st.tabs([
        "Analyze",
        "Parlays",
        "Tracking",
        "Learn",
        "Account",
    ])

    with tab_analyze:
        render_single_section()

    with tab_parlays:
        p2, p3, p4 = st.tabs(["2-Leg Parlay", "3-Leg Parlay", "4-Leg Parlay"])
        with p2:
            render_parlay_section(2)
        with p3:
            render_parlay_section(3)
        with p4:
            render_parlay_section(4)

    with tab_tracking:
        render_grading_section()
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        render_tracker_dashboard()

    with tab_learn:
        learn_about, learn_metrics, learn_faq = st.tabs(["How to Use Propify", "Metric Guide", "FAQs"])
        with learn_about:
            render_about_section()
        with learn_metrics:
            render_metric_guide_section()
        with learn_faq:
            render_faq_section()

    with tab_account:
        render_section_title("Logout")
        align_section_inputs()
        st.markdown(
            """
            **Your account data stays saved**

            Saved picks, grading history, notes, outcomes, and tracking data remain securely tied to your account.
            You can log out at any time and sign back in later without losing your saved information.
            """
        )
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("Log Out", key="overview_logout_btn", use_container_width=False):
            sign_out()


def render_single_section_page():
    render_single_section()


def render_two_leg_page():
    render_parlay_section(2)


def render_three_leg_page():
    render_parlay_section(3)


def render_four_leg_page():
    render_parlay_section(4)


def render_pick_tracker_page():
    render_grading_section()
    render_thin_section_line()
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    render_tracker_dashboard()


def render_how_to_use_page():
    render_about_section()


def render_faq_page():
    render_faq_section()


def render_metric_guide_page():
    render_metric_guide_section()


def render_logout_page():
    render_section_title("Logout")
    align_section_inputs()
    st.markdown(
        """
        **Your account data stays saved**

        Saved picks, grading history, notes, outcomes, and tracking data remain securely tied to your account.
        You can log out at any time and sign back in later without losing your saved information.
        """
    )
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if st.button("Log Out", key="sidebar_logout_btn", use_container_width=False):
        sign_out()


def render_current_section_page():
    render_header()
    current = st.session_state.get("current_section", "Overview")
    if current == "Overview":
        render_overview_page()
    elif current == "Single Prop Analyzer":
        render_single_section_page()
    elif current == "2-Leg Parlay":
        render_two_leg_page()
    elif current == "3-Leg Parlay":
        render_three_leg_page()
    elif current == "4-Leg Parlay":
        render_four_leg_page()
    elif current == "Pick Tracker":
        render_pick_tracker_page()
    elif current == "How to Use Propify":
        render_how_to_use_page()
    elif current == "Metric Guide":
        render_metric_guide_page()
    elif current == "FAQs":
        render_faq_page()
    elif current == "Logout":
        render_logout_page()
    else:
        render_overview_page()


def render_main_navigation_layout():
    render_current_section_page()

def render_mode_toggle():
    dark_mode = st.session_state.get("dark_mode", True)
    if dark_mode:
        st.markdown("""
        <style>
        .stApp {background: #122235; color: #f5f7fb;}
        [data-testid="stHeader"] {background: transparent;}
        [data-testid="stToolbar"] {right: 1rem;}
        /* Keep dark mode styling without overriding Streamlit rerun/fade behavior */
        [data-testid="stStatusWidget"], [data-testid="stSkeleton"], .stSkeleton, [data-testid="stDecoration"], .element-container [data-testid="stMarkdownContainer"]::after {display:none !important;}
        h1,h2,h3,h4,h5,h6,label,p,span,li,small,strong,em {color: #f5f7fb !important;}
        .stTabs [data-baseweb="tab"] {color: #f5f7fb !important;}
        .stTextInput input, .stNumberInput input, .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            background-color: #1d3148 !important;
            color: #f5f7fb !important;
            border-color: #d8e2ee !important;
            caret-color: #ffffff !important;
        }
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {color:#c6d2e1 !important;}
        .stSelectbox div[data-baseweb="select"] span,
        .stSelectbox div[data-baseweb="select"] input,
        .stMultiSelect div[data-baseweb="select"] span,
        .stMultiSelect div[data-baseweb="select"] input {
            color: #f5f7fb !important;
            caret-color: #ffffff !important;
        }
        /* Dropdown menu / options should stay white with navy text for readability */
        div[data-baseweb="popover"], div[data-baseweb="popover"] * ,
        ul[role="listbox"], ul[role="listbox"] *,
        div[role="listbox"], div[role="listbox"] *,
        li[role="option"], li[role="option"] * {
            color: #122235 !important;
        }
        div[data-baseweb="popover"], ul[role="listbox"], div[role="listbox"], li[role="option"] {
            background: #ffffff !important;
        }
        /* White cards need navy text */
        .primary-card, .secondary-card, .chart-card {
            background: #ffffff !important;
            color: #122235 !important;
        }
        .primary-card .label, .primary-card .value,
        .secondary-card .label, .secondary-card .value,
        .chart-card, .chart-card *, .recent-sample-wrap, .recent-sample-wrap * {
            color: #122235 !important;
        }
        div[data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #d8e2ee33;
            border-radius: 18px;
            padding: 0.5rem 0.75rem;
            color: #122235 !important;
        }
        div[data-testid="stMetric"] * {color:#122235 !important;}
        .stDataFrame, .stTable {background-color: #122235 !important;}
        .stButton > button,
        [data-testid="stFormSubmitButton"] > button,
        button[kind="primary"],
        button[kind="secondary"],
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"] {
            background: #ffffff !important;
            color: #122235 !important;
            border: 1px solid #d8e2ee66 !important;
            -webkit-text-fill-color: #122235 !important;
        }
        .stButton > button *,
        [data-testid="stFormSubmitButton"] > button *,
        button[kind="primary"] *,
        button[kind="secondary"] * {
            color: #122235 !important;
            -webkit-text-fill-color: #122235 !important;
        }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover, button[kind="primary"]:hover, button[kind="secondary"]:hover {
            background: #1d3148 !important;
            color: #ffffff !important;
            border-color: #d8e2ee !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        .stButton > button:hover *, [data-testid="stFormSubmitButton"] > button:hover *, button[kind="primary"]:hover *, button[kind="secondary"]:hover * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        .stButton > button:disabled,
        [data-testid="stFormSubmitButton"] > button:disabled,
        button[kind="primary"]:disabled,
        button[kind="secondary"]:disabled {
            opacity: 1 !important;
            background: #f3f4f6 !important;
            color:#122235 !important;
            -webkit-text-fill-color: #122235 !important;
            border: 1px solid #d8e2ee !important;
        }
        .stButton > button:disabled *, [data-testid="stFormSubmitButton"] > button:disabled * {
            color:#122235 !important;
            -webkit-text-fill-color: #122235 !important;
        }
        [data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #d8e2ee !important;
            border-radius: 18px !important;
        }
        [data-testid="stExpander"] > div, [data-testid="stExpander"] > div * {
            color: #122235 !important;
        }
        [data-testid="stExpander"] .stTextInput input,
        [data-testid="stExpander"] .stNumberInput input,
        [data-testid="stExpander"] .stTextArea textarea,
        [data-testid="stExpander"] .stSelectbox div[data-baseweb="select"] > div,
        [data-testid="stExpander"] .stMultiSelect div[data-baseweb="select"] > div {
            background-color: #1d3148 !important;
            color: #f5f7fb !important;
            border-color: #d8e2ee !important;
            caret-color: #ffffff !important;
        }
        [data-testid="stExpander"] .stSelectbox div[data-baseweb="select"] span,
        [data-testid="stExpander"] .stSelectbox div[data-baseweb="select"] input,
        [data-testid="stExpander"] .stMultiSelect div[data-baseweb="select"] span,
        [data-testid="stExpander"] .stMultiSelect div[data-baseweb="select"] input,
        [data-testid="stExpander"] .stTextInput input,
        [data-testid="stExpander"] .stNumberInput input,
        [data-testid="stExpander"] .stTextArea textarea {
            color: #f5f7fb !important;
        }
        [data-testid="stExpander"] .stButton > button,
        [data-testid="stExpander"] button[kind="primary"],
        [data-testid="stExpander"] button[kind="secondary"] {
            background: #ffffff !important;
            color: #122235 !important;
        }
        [aria-modal="true"] {
            background: #ffffff !important;
        }
        [aria-modal="true"] * {
            color: #122235 !important;
        }
        [aria-modal="true"] input,
        [aria-modal="true"] textarea,
        [aria-modal="true"] .stSelectbox div[data-baseweb="select"] > div,
        [aria-modal="true"] .stMultiSelect div[data-baseweb="select"] > div {
            background-color: #1d3148 !important;
            color: #f5f7fb !important;
            caret-color: #ffffff !important;
        }
        [aria-modal="true"] .stButton > button {background:#ffffff !important; color:#122235 !important; -webkit-text-fill-color:#122235 !important;}
        [aria-modal="true"] label,
        [aria-modal="true"] p,
        [aria-modal="true"] span,
        [aria-modal="true"] div,
        [data-testid="stExpanderDetails"] label,
        [data-testid="stExpanderDetails"] p,
        [data-testid="stExpanderDetails"] span,
        [data-testid="stExpanderDetails"] div {
            color:#122235 !important;
        }
        [data-testid="stExpanderDetails"] .stSelectbox div[data-baseweb="select"] > div,
        [data-testid="stExpanderDetails"] .stTextInput input,
        [data-testid="stExpanderDetails"] .stNumberInput input,
        [data-testid="stExpanderDetails"] .stTextArea textarea,
        [data-testid="stExpanderDetails"] button,
        [data-testid="stExpanderDetails"] button * {
            color:#122235 !important;
            -webkit-text-fill-color:#122235 !important;
        }
        /* Edit Saved Entries dropdown: black title on white card, white selected text on navy field */
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] details summary,
        [data-testid="stExpander"] details summary *,
        [data-testid="stExpanderToggleIcon"],
        [data-testid="stExpanderToggleIcon"] * {
            color:#122235 !important;
            -webkit-text-fill-color:#122235 !important;
            fill:#122235 !important;
            stroke:#122235 !important;
        }
        [data-testid="stExpanderDetails"] label,
        [data-testid="stExpanderDetails"] .stSelectbox label,
        [data-testid="stExpanderDetails"] .stNumberInput label,
        [data-testid="stExpanderDetails"] .stTextArea label {
            color:#122235 !important;
            -webkit-text-fill-color:#122235 !important;
        }
        [data-testid="stExpanderDetails"] .stSelectbox div[data-baseweb="select"] > div {
            background-color:#1d3148 !important;
            color:#ffffff !important;
            -webkit-text-fill-color:#ffffff !important;
            border-color:#d8e2ee !important;
        }
        [data-testid="stExpanderDetails"] .stSelectbox div[data-baseweb="select"] span,
        [data-testid="stExpanderDetails"] .stSelectbox div[data-baseweb="select"] input,
        [data-testid="stExpanderDetails"] .stSelectbox div[data-baseweb="select"] div {
            color:#ffffff !important;
            -webkit-text-fill-color:#ffffff !important;
        }
        </style>
        """, unsafe_allow_html=True)

def render_about_section():


    render_section_title("How to Use Propify")
    st.markdown(
        """
        The **Propify** is designed to help you evaluate NBA player props and parlays using a structured,
        continuously refreshed, data-driven workflow. The engine combines current-season game logs, recent player performance,
        matchup context, projected minutes, volatility, and hit-rate trends to create projections, probability estimates,
        confidence classifications, and tracking outputs that can be used to make more disciplined decisions.

        ### What the engine does
        - Analyzes **single props** and **2-leg / 3-leg / 4-leg parlays**
        - Pulls and refreshes player/team data from live NBA data sources as the app reruns
        - Estimates **projection, over %, under %, hit rates, confidence, recent form, matchup score, and risk**
        - Lets you **save picks**, track results, record cash-outs, and review long-term performance
        - Stores saved picks per user account so each user sees only their own entries

        ### How the data stays accurate and continuously updated
        The engine is built to rerun and refresh from current data sources whenever users interact with the app.
        It pulls player and team information directly from live NBA endpoints and current-season game logs, which means:
        - player/team selection lists stay current
        - opponent auto-fill stays tied to the player's current matchup until that game is over
        - recent averages, hit rates, and matchup context reflect the latest available games
        - saved entries and dashboard metrics refresh from your account-based database

        The model is intentionally designed around **current-season behavior and recent form**, rather than treating all historical data equally.
        That helps the engine stay more responsive to role changes, injuries, usage shifts, and player trends over time.

        ### Step-by-step: how to use the engine
        #### 1. Log in
        Create an account or log in to access the engine. Your saved picks, edits, outcomes, and tracking data are tied to your account.

        #### 2. Analyze a single prop
        Go to **Single Prop Analyzer** and enter:
        - player
        - stat type
        - line

        The opponent is auto-generated from the player's current team matchup and updates only after that game is final.
        You can still manually override the opponent if needed.

        After analyzing, review:
        - Projection
        - Over %
        - Under %
        - Lean
        - Confidence
        - Median / Floor / Ceiling
        - EV %
        - Hit Rate
        - Model Scores
        - Context / Reasons / Risk Flags

        #### 3. Build parlays
        Go to the **2-Leg**, **3-Leg**, or **4-Leg** tabs.
        For each leg:
        - choose player
        - choose stat
        - choose side
        - enter line

        Each leg auto-generates the current valid opponent the same way the single-prop analyzer does.
        After that, analyze the parlay and review:
        - Hit Probability
        - Miss Probability
        - EV %
        - Confidence
        - Individual leg summaries

        #### 4. Save your picks
        Once you identify a play you want to track:
        - save the pick
        - enter bet size
        - enter multiplier
        - optionally include notes

        This stores the entry in your account so you can grade it later and evaluate long-term profitability.

        #### 5. Track outcomes correctly
        In **Pick Tracker**, you can:
        - grade pending picks
        - edit saved entries
        - record wins, losses, pushes, and cash-outs
        - enter a cash-out price to calculate net profit correctly

        For cash-outs:
        - Outcome = **Cash Out**
        - Net Profit = **Cash Out Price - Bet Size**

        #### 6. Use the dashboard to improve decision-making
        The tracker dashboard is where the engine becomes most useful over time.
        Review:
        - Win rate
        - Net Profit
        - ROI
        - Entry-level history
        - Which types of picks are performing best or worst

        ### How to use the tool profitably
        No model guarantees outcomes, and no tool removes variance. The best use of this engine is to make decisions with discipline:
        - prioritize higher-confidence plays
        - compare projection vs line instead of betting emotionally
        - use the tracker to identify what actually performs well over time
        - avoid forcing low-edge plays just to create volume
        - focus on consistency, EV, and data-backed selection quality

        The real advantage of this engine is not just one pick at a time — it is the ability to analyze props systematically,
        track them honestly, and improve your process over repeated use.
        """
    )




def render_metric_guide_section():
    render_section_title("Metric Guide")
    st.markdown(
        """
        This section explains every major widget, value, label, and calculation shown throughout Propify so the full UI is transparent and easy to use correctly.
        """
    )

    guide_sections = {
        "Core Result Values": [
            ("Projection", "The engine's final expected outcome for the selected stat after the current method is applied.", "Use it first against the sportsbook line. If the projection is clearly above the line, that usually supports the over. If clearly below, it usually supports the under."),
            ("Over %", "Estimated chance the player clears the line.", "Higher values indicate a stronger over case. Compare it directly against Under % to see which side the engine actually prefers."),
            ("Under %", "Estimated chance the player stays under the line.", "Use it exactly like Over %. If this is higher than Over %, the under is the stronger side."),
            ("Push %", "Estimated chance the player lands exactly on the line.", "Most relevant on whole-number lines where a push is possible. It helps explain why Over % and Under % may not sum to exactly 100%."),
            ("Confidence", "A summary label that reflects how strong the current play looks after combining projection edge, hit rates, variance, and context.", "Use it to prioritize cleaner spots instead of forcing marginal plays."),
            ("EV %", "Expected value estimate using the multiplier or odds you entered.", "Positive EV means the payout offered may be better than the engine's implied fair price."),
            ("Fair Line", "The line the engine would consider more fairly priced for this prop.", "Use it to compare your book's number to the engine's internal expectation."),
            ("Fair Odds O / U", "The implied fair American odds for the over or under from the engine probabilities.", "Compare these against sportsbook odds to judge whether a side is underpriced or overpriced."),
            ("Season Hit Rate", "How often the player has cleared this exact line this season using the current season window of October 2025 through today.", "Use it as a context check, not as a standalone decision-maker."),
            ("Edge vs Line", "Difference between the final projection and the sportsbook line.", "Positive values help overs. Negative values help unders."),
        ],
        "Projection Method Details": [
            ("Final Projection", "The exact number used in the main Projection card.", "This should always match the Projection widget in the main result cards."),
            ("Rules Projection", "The deterministic projection produced from the rules-based side of the engine.", "Use it to understand what the non-ML side of the engine thinks."),
            ("ML Projection", "The machine-learning model's standalone projection when an ML output is available.", "Compare it to the Rules Projection to see whether the two methods agree or diverge."),
            ("ML Blend %", "How much of the final projection comes from the ML side versus the rules side.", "A 36% ML blend means the final number is 64% rules and 36% ML."),
            ("Train Rows", "How many rows of training data were available for the ML model.", "Higher row counts usually mean the model had more information to learn from."),
            ("Train R²", "How well the model fit the training data.", "Use it as a rough reference only. A strong train fit alone does not guarantee good out-of-sample performance."),
            ("Residual Std", "The typical size of the model's remaining error after fitting.", "Lower values generally indicate a tighter and less noisy model fit."),
            ("Validation MAE", "Mean absolute error on validation data.", "Lower validation MAE is better because it means the model missed by less on held-out data."),
            ("Method", "The active projection method label, such as Rules only, ML only, or ML + Rules blend.", "Use it to understand exactly how the final projection was built."),
        ],
        "Distribution and Range Metrics": [
            ("Median", "Central expected outcome after the model blend and distribution logic.", "Usually similar to the projection and useful as a stable midpoint."),
            ("Floor (20th)", "Lower-end expected result around the 20th percentile.", "Helps you judge downside risk."),
            ("Ceiling (80th)", "Upper-end expected result around the 80th percentile.", "Helps you judge upside."),
            ("Recent Form Score", "How strong the player's recent trend looks.", "Higher values usually support better current form."),
            ("Matchup Score", "How favorable the opponent matchup looks for that stat.", "Higher means the matchup environment is more favorable."),
            ("Volatility Score", "How swingy the stat is from game to game.", "Higher volatility means wider outcomes and more uncertainty."),
            ("Minutes Risk Score", "Risk tied to unstable or uncertain playing time.", "Lower minutes risk usually supports more reliable projections."),
        ],
        "Hit Rates and Context": [
            ("Last 5 / Last 10 / Last 15", "How often the player cleared the line over those rolling windows.", "Use these to compare short-term form across different recent samples."),
            ("Season", "Current-season hit rate for the same line using only October 2025 through today.", "Use it to compare short-term hit rates to longer-term current-season consistency."),
            ("Projected Minutes", "Expected playing time for the next game.", "Minutes are one of the biggest drivers of prop performance."),
            ("Rest Days", "Days since the player's last game.", "Useful context for fatigue, rest, and scheduling effects."),
            ("Home/Away", "Whether the upcoming matchup is at home or on the road.", "Can help explain venue-based splits for some players."),
            ("Pace Proxy", "Approximate game-speed estimate for the matchup.", "Higher pace usually means more possessions and more stat opportunity."),
            ("Team Pace", "Estimated pace for the player's team.", "Use it with Opponent Pace to understand expected possession volume."),
            ("Opponent Pace", "Estimated pace for the opposing team.", "Faster opponents can raise game volume."),
            ("Role", "Snapshot of the player's role stability and usage context.", "Stable roles generally create more dependable projections."),
            ("Opp B2B", "Whether the opponent is on the second night of a back-to-back.", "Can slightly shift pace, efficiency, and matchup quality."),
        ],
        "Why This Lean, Risk, and Recent Sample": [
            ("Why This Lean", "Short bullets describing the top reasons the engine prefers the current side.", "This is the fastest human-readable explanation of the pick."),
            ("Risk Flags", "Warnings for volatility, unstable minutes, difficult matchups, or other caution signals.", "Use it as the last quality-control check before saving or betting a play."),
            ("Recent Sample", "Recent game log table filtered to the selected window.", "Use it to visually inspect trend, variance, and how often the player actually cleared the line."),
            ("Window", "The selected view for the Recent Sample table: Last 10, Season, or All-Time.", "Use Season for October 2025 through today, and All-Time for the broader historical view."),
            ("Game Date", "Date of the game in the Recent Sample table.", "Useful for checking recency and streak context."),
            ("Pick Result", "Whether the player went over, under, or pushed relative to the selected line.", "Use it to quickly audit hit-rate behavior row by row."),
        ],
        "Parlay Summary and Leg Metrics": [
            ("Hit Probability", "Estimated chance the full parlay hits.", "Use it to compare the engine's probability to the payout being offered."),
            ("Miss Probability", "Estimated chance the entry misses.", "Helpful for understanding total ticket risk."),
            ("Entry Type", "The type of ticket being analyzed.", "Used for quick reference across 2-leg, 3-leg, and 4-leg entries."),
            ("Flex Probability", "Chance a flex entry lands enough legs to cash partially.", "Most relevant for 3-leg and 4-leg flex formats."),
            ("Projection", "The final projection for an individual leg.", "Every parlay leg should make sense on its own before being combined."),
            ("Last 5 Average", "Average result for that stat over the player's last 5 games.", "Use it as a quick short-term form check."),
            ("Last 15 Average", "Average result over the player's last 15 games.", "Useful for balancing short-term and medium-term form."),
            ("Season Average", "Current-season average for that stat.", "Helps anchor the leg in larger-sample context."),
            ("Selected Side", "The side chosen for that leg, over or under.", "Confirms exactly what is being modeled."),
            ("Selected Side %", "Estimated hit probability for the chosen side only.", "This is the key percentage that feeds the overall parlay probability."),
            ("Confidence", "Confidence label for the individual leg.", "Use it to avoid forcing weak legs into a larger ticket."),
        ],
        "Pick Tracker and Saved Pick Metrics": [
            ("Outcome", "Final graded result of a saved pick such as win, loss, push, cash out, or pending.", "Use it to track actual historical performance."),
            ("Bet Size", "Stake amount entered for a saved play.", "Needed for profit and ROI tracking."),
            ("Multiplier", "Payout multiplier or equivalent ticket price entered when the pick was saved.", "Used to estimate EV and calculate final profit."),
            ("Profit", "Net money result for the saved pick after grading.", "Use it for bankroll tracking over time."),
            ("Win Rate", "Percentage of graded picks that won.", "Useful for performance review, but should be interpreted alongside ROI."),
            ("Total Profit", "Total money won or lost across saved picks.", "A key long-term performance measure."),
            ("ROI", "Return on investment based on stake and profit.", "Helps evaluate efficiency, not just total dollars won."),
            ("Pending", "Saved picks that have not been graded yet.", "Useful for understanding current open exposure."),
            ("Cash Out", "A saved pick that was exited early for a partial result.", "Lets you capture and review non-standard outcomes in the tracker."),
        ],
    }

    for section_title, items in guide_sections.items():
        render_section_header(section_title)
        for name, meaning, use_case in items:
            st.markdown(
                f"""
                <div class="propify-guide-item">
                    <div class="propify-guide-title">{name}</div>
                    <div class="propify-guide-line">
                        <span class="propify-guide-label">Meaning:</span>
                        <span class="propify-guide-meaning"> {meaning}</span>
                    </div>
                    <div class="propify-guide-line">
                        <span class="propify-guide-label">Use case:</span>
                        <span class="propify-guide-usecase"> {use_case}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def render_faq_section():
    render_section_title("FAQs")
    faqs = [
        (
            "How do I know what picks to analyze?",
            "Browse any sportsbook to find plays that you are interested in potentially placing a bet on, that you think might be mispriced (the multiplier/odds are either too low or too high), or plays that you are just curious about."
        ),
        (
            "What are good percentages and how do you know when a pick is worth making?",
            "Any individual picks with data that shows consistency, a solid difference between the projection and the line, and a hit rate of over 80% is an example of a great pick. A more technical answer would be: if you enter in a play's multiplier/odds, and the engine gives you a positive expected value (EV) of over 10%, there is significant value in that play. Positive expected value measures the percentage of perceived value that a pick has, using the engine's calculations vs the multiplier/odds given by a sportsbook. Positive EV over 10% = value!"
        ),
        (
            "What should I look at first after I analyze a pick?",
            "Start with the projection versus the line, then check the hit rates, confidence, and EV if you entered a multiplier. If the projection is barely above or below the line and the recent data is inconsistent, the play is usually less attractive than one with a clearer statistical edge."
        ),
        (
            "Why can a pick still lose even if the percentages look strong?",
            "Because sports props are still probabilistic. A strong edge does not mean certainty. Even a great play can lose because of foul trouble, poor shooting efficiency, role changes, injuries, game script, or simple variance. The purpose of the engine is to help you make better long-term decisions, not guarantee any individual outcome."
        ),
        (
            "When should I use a parlay instead of a single prop?",
            "Use parlays when multiple legs each show value on their own and still make sense together as one entry. If you are stretching to include weaker legs just to build a bigger payout, that usually lowers quality. Strong singles are often better than forcing low-edge parlays."
        ),
        (
            "Why should I enter the sportsbook multiplier or odds?",
            "Because that allows Propify to compare the model's probability to the payout being offered. Once you enter the multiplier or odds, the engine can estimate expected value (EV), which helps show whether the book may be underpricing or overpricing the play."
        ),
        (
            "What is the best way to use Pick Tracker?",
            "Track every serious play you make, including bet size, multiplier, notes, and final result. Over time, Pick Tracker helps you see whether your process is actually profitable, which types of bets perform best, and whether you are improving or just remembering the wins more than the losses."
        ),
    ]
    for i, (q, a) in enumerate(faqs):
        st.markdown(f"<div style='font-size:1.14rem;font-weight:800;margin:0 0 0.35rem 0;'>{q}</div>", unsafe_allow_html=True)
        st.markdown(a)
        if i != len(faqs) - 1:
            render_thin_section_line()


def section_divider():

    st.markdown("<div class='section-space'></div><hr style='border:none;border-top:1px solid #e5e7eb;margin:16px 0 22px 0;'>", unsafe_allow_html=True)


def render_footer():
    st.markdown(
        """
        <div style="
            margin-top:56px;
            padding:18px 16px;
            border-top:1px solid #e5e7eb;
            color:#6b7280;
            font-size:0.92rem;
            line-height:1.6;
            text-align:center;
        ">
            This engine does not guarantee that any bets placed will cash and is not responsible for any money lost,
            wagering outcomes, or decisions made using its outputs. All projections and probabilities are informational
            tools only and should be treated as model-based estimates rather than certainty. Please bet responsibly and
            understand that sports outcomes are inherently uncertain. This model was created by Davis Higgins, a data
            analyst with experience leveraging artificial intelligence, developing executive financial dashboards, engineering 
            high-level machine learning models, and ulitlizing business analytics to support intelligent, data-driven decision making.
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Save Single Pick")
def save_single_pick_dialog():
    result = st.session_state.get("last_analysis")
    pick_meta = st.session_state.get("last_pick_meta")

    if not result or result.get("error") or not pick_meta:
        st.error("No valid single pick available to save.")
        return

    lean_prob = result["over_prob"] if result["lean"] == "OVER" else result["under_prob"]

    bet_size = st.number_input("Bet Size ($)", min_value=0.0, step=1.0, value=0.0, key="single_save_bet_size")
    multiplier = st.number_input("Multiplier", min_value=0.0, step=0.1, value=1.2, key="single_save_multiplier")
    notes = st.text_area("Notes (optional)", value=pick_meta.get("user_notes", ""), key="single_save_notes")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save Pick", use_container_width=True):
            pick_id = str(uuid.uuid4())[:8]
            payload = {col: "" for col in PICK_HISTORY_COLUMNS}
            payload.update({
                "pick_id": pick_id,
                "entry_type": "single",
                "legs_count": 1,
                "saved_at": format_timestamp(),
                "label": f"{pick_meta['player']} {result['lean']} {stat_display_name(pick_meta['stat'])} {pick_meta['line']}",
                "confidence": result["confidence"],
                "confidence_score_numeric": result.get("confidence_score_numeric", ""),
                "selected_prob": round(lean_prob, 1),
                "bet_size": safe_float(bet_size, 0.0),
                "multiplier": safe_float(multiplier, 0.0),
                "player": pick_meta["player"],
                "stat": pick_meta["stat"],
                "line": pick_meta["line"],
                "opponent": pick_meta["opponent"],
                "selected_side": result["lean"],
                "projection": result["projection"],
                "season_average": result["season_average"],
                "last10_average": result["last10_average"],
                "last15_average": result["last15_average"],
                "last5_average": result["last5_average"],
                "recent_form_score": result["recent_form_score"],
                "matchup_score": result["matchup_score"],
                "volatility_score": result["volatility_score"],
                "minutes_risk_score": result["minutes_risk_score"],
                "user_notes": notes,
            })
            if save_pick_to_history(payload):
                st.session_state["open_single_save_dialog"] = False
                st.rerun()
    with c2:
        if st.button("Cancel", use_container_width=True):
            st.session_state["open_single_save_dialog"] = False
            st.rerun()


@st.dialog("Save Parlay")
def save_parlay_dialog(section_key: str, legs_count: int):
    data = st.session_state.get(section_key)
    if not data:
        st.error("No valid parlay available to save.")
        return

    results = data["legs"]
    meta = data["meta"]
    probs = [selected_side_prob(r, meta["sides"][i]) / 100 for i, r in enumerate(results)]
    parlay_hit_prob = 1.0
    for p in probs:
        parlay_hit_prob *= p
    parlay_miss_prob = 1 - parlay_hit_prob
    combined_conf_label, combined_conf_num = combine_confidence_labels([r["confidence"] for r in results])

    bet_size = st.number_input("Bet Size ($)", min_value=0.0, step=1.0, value=0.0, key=f"{section_key}_save_bet_size")

    allow_flex = legs_count in (3, 4)
    if allow_flex:
        mult_col, flex_col = st.columns([3, 1])
        with flex_col:
            flex_pick_checked = st.checkbox("Flex Pick", key=f"{section_key}_is_flex_pick")
        with mult_col:
            multiplier = st.number_input(
                "Multiplier",
                min_value=0.0,
                step=0.1,
                value=1.2,
                key=f"{section_key}_save_multiplier",
                disabled=flex_pick_checked,
            )
    else:
        flex_pick_checked = False
        multiplier = st.number_input("Multiplier", min_value=0.0, step=0.1, value=1.2, key=f"{section_key}_save_multiplier")

    flex_full_hit_multiplier = 0.0
    flex_one_miss_multiplier = 0.0
    if flex_pick_checked:
        full_label, partial_label = flex_field_labels(legs_count)
        flex_a, flex_b = st.columns(2)
        with flex_a:
            flex_full_hit_multiplier = st.number_input(
                full_label,
                min_value=0.0,
                step=0.1,
                value=0.0,
                key=f"{section_key}_flex_full_multiplier",
            )
        with flex_b:
            flex_one_miss_multiplier = st.number_input(
                partial_label,
                min_value=0.0,
                step=0.1,
                value=0.0,
                key=f"{section_key}_flex_partial_multiplier",
            )

    notes = st.text_area("Notes (optional)", value=meta.get("notes", ""), key=f"{section_key}_save_notes")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save Parlay", use_container_width=True, key=f"{section_key}_confirm"):
            if flex_pick_checked:
                if flex_full_hit_multiplier <= 0 or flex_one_miss_multiplier <= 0:
                    st.error("Enter both flex multipliers before saving this flex pick.")
                    return
                selected_multiplier = safe_float(flex_full_hit_multiplier, 0.0)
            else:
                if multiplier <= 0:
                    st.error("Enter a multiplier greater than 0 before saving this parlay.")
                    return
                selected_multiplier = safe_float(multiplier, 0.0)

            payload = {col: "" for col in PICK_HISTORY_COLUMNS}
            payload.update({
                "pick_id": str(uuid.uuid4())[:8],
                "entry_type": entry_type_label(legs_count),
                "legs_count": legs_count,
                "saved_at": format_timestamp(),
                "label": " + ".join(
                    [f"{meta['players'][i]} {meta['sides'][i]} {stat_display_name(meta['stats'][i])} {meta['lines'][i]}" for i in range(legs_count)]
                ),
                "confidence": combined_conf_label,
                "confidence_score_numeric": combined_conf_num,
                "parlay_hit_prob": round(parlay_hit_prob * 100, 1),
                "parlay_miss_prob": round(parlay_miss_prob * 100, 1),
                "bet_size": safe_float(bet_size, 0.0),
                "multiplier": selected_multiplier,
                "is_flex_pick": flex_pick_checked,
                "flex_full_hit_multiplier": safe_float(flex_full_hit_multiplier, 0.0) if flex_pick_checked else "",
                "flex_one_miss_multiplier": safe_float(flex_one_miss_multiplier, 0.0) if flex_pick_checked else "",
                "flex_full_hit_legs": legs_count if flex_pick_checked else "",
                "flex_one_miss_hit_legs": legs_count - 1 if flex_pick_checked else "",
                "user_notes": notes,
                "stat": entry_type_label(legs_count),
            })
            for i in range(legs_count):
                payload[f"leg{i+1}_player"] = meta["players"][i]
                payload[f"leg{i+1}_stat"] = meta["stats"][i]
                payload[f"leg{i+1}_line"] = meta["lines"][i]
                payload[f"leg{i+1}_opponent"] = meta["opponents"][i]
                payload[f"leg{i+1}_side"] = meta["sides"][i]
                payload[f"leg{i+1}_projection"] = results[i]["projection"]
                payload[f"leg{i+1}_selected_prob"] = round(probs[i] * 100, 1)
                payload[f"leg{i+1}_confidence"] = results[i]["confidence"]
                payload[f"leg{i+1}_confidence_score_numeric"] = results[i].get("confidence_score_numeric", "")
                payload[f"leg{i+1}_season_average"] = results[i]["season_average"]
                payload[f"leg{i+1}_last15_average"] = results[i]["last15_average"]
                payload[f"leg{i+1}_last10_average"] = results[i]["last10_average"]
                payload[f"leg{i+1}_last5_average"] = results[i]["last5_average"]
            if save_pick_to_history(payload):
                st.session_state[f"open_{section_key}_save_dialog"] = False
                st.rerun()
    with c2:
        if st.button("Cancel", use_container_width=True, key=f"{section_key}_cancel"):
            st.session_state[f"open_{section_key}_save_dialog"] = False
            st.rerun()


def settle_single_pick_result(df: pd.DataFrame, idx: int, actual_result: float):
    line = pd.to_numeric(df.at[idx, "line"], errors="coerce")
    side = str(df.at[idx, "selected_side"]).upper()

    if pd.isna(line):
        return False, "Saved line is missing or invalid."

    if actual_result == line:
        outcome = "push"
    elif side == "OVER":
        outcome = "win" if actual_result > line else "loss"
    else:
        outcome = "win" if actual_result < line else "loss"

    amount_won, amount_lost, net_result = calculate_money_from_outcome(outcome, df.at[idx, "bet_size"], df.at[idx, "multiplier"])
    df.at[idx, "actual_result"] = actual_result
    df.at[idx, "outcome"] = outcome
    df.at[idx, "amount_won"] = amount_won
    df.at[idx, "amount_lost"] = amount_lost
    df.at[idx, "net_result"] = net_result
    df.at[idx, "settled_at"] = format_timestamp()
    return True, outcome


def settle_parlay_result(df: pd.DataFrame, idx: int, actual_results: list[float]):
    legs_count = int(pd.to_numeric(df.at[idx, "legs_count"], errors="coerce") or 0)

    def leg_grade(actual, line, side):
        if actual == line:
            return "push"
        if side == "OVER":
            return "win" if actual > line else "loss"
        return "win" if actual < line else "loss"

    leg_outcomes = []

    for i in range(1, legs_count + 1):
        line = pd.to_numeric(df.at[idx, f"leg{i}_line"], errors="coerce")
        side = str(df.at[idx, f"leg{i}_side"]).upper()
        actual = float(actual_results[i - 1])

        if pd.isna(line):
            return False, "Saved parlay lines are missing or invalid."

        df.at[idx, f"actual_result_leg{i}"] = actual
        leg_outcomes.append(leg_grade(actual, line, side))

    flex_pick = bool_from_value(df.at[idx, "is_flex_pick"]) and legs_count in (3, 4)
    applied_multiplier = safe_float(pd.to_numeric(df.at[idx, "multiplier"], errors="coerce"), 0.0)

    if flex_pick:
        win_count = sum(1 for outcome in leg_outcomes if outcome == "win")
        full_hit_target = int(pd.to_numeric(df.at[idx, "flex_full_hit_legs"], errors="coerce") or legs_count)
        one_miss_target = int(pd.to_numeric(df.at[idx, "flex_one_miss_hit_legs"], errors="coerce") or max(legs_count - 1, 0))
        full_hit_multiplier = safe_float(pd.to_numeric(df.at[idx, "flex_full_hit_multiplier"], errors="coerce"), 0.0)
        one_miss_multiplier = safe_float(pd.to_numeric(df.at[idx, "flex_one_miss_multiplier"], errors="coerce"), 0.0)

        if win_count >= full_hit_target:
            outcome = "win"
            applied_multiplier = full_hit_multiplier
        elif win_count >= one_miss_target:
            outcome = "win"
            applied_multiplier = one_miss_multiplier
        else:
            outcome = "loss"
            applied_multiplier = max(full_hit_multiplier, one_miss_multiplier)
    else:
        if "loss" in leg_outcomes:
            outcome = "loss"
        elif "push" in leg_outcomes:
            outcome = "push"
        else:
            outcome = "win"

    amount_won, amount_lost, net_result = calculate_money_from_outcome(outcome, df.at[idx, "bet_size"], applied_multiplier)
    df.at[idx, "multiplier"] = applied_multiplier
    df.at[idx, "outcome"] = outcome
    df.at[idx, "amount_won"] = amount_won
    df.at[idx, "amount_lost"] = amount_lost
    df.at[idx, "net_result"] = net_result
    df.at[idx, "settled_at"] = format_timestamp()
    return True, outcome


def settle_pick_result(pick_id: str, actual_result=None, actual_leg_results=None):
    df = load_pick_history()
    match_idx = df.index[df["pick_id"].astype(str) == str(pick_id)].tolist()
    if not match_idx:
        return False, "Pick not found."
    idx = match_idx[0]
    entry_type = str(df.at[idx, "entry_type"]).lower()
    if entry_type == "single":
        ok, msg = settle_single_pick_result(df, idx, float(actual_result))
    else:
        ok, msg = settle_parlay_result(df, idx, actual_leg_results or [])
    if ok:
        save_pick_to_history(df.iloc[idx].to_dict())
    return ok, msg


def render_table_clean(df: pd.DataFrame):
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_hit_rate_section(result, show_explanations: bool = False):
    render_section_header("Hit Rate")
    h1, h2, h3, h4 = st.columns(4)
    for col, label, value in [
        (h1, "Last 5", result["last5_hit_rate"]),
        (h2, "Last 10", result["last10_hit_rate"]),
        (h3, "Last 15", result["last15_hit_rate"]),
        (h4, "Season", result["season_hit_rate"]),
    ]:
        with col:
            st.metric(label, value)


def apply_pending_single_history_entry():
    entry = st.session_state.pop("pending_single_history_entry", None)
    if not entry:
        return
    st.session_state["single_player"] = entry.get("player_display")
    st.session_state["single_stat"] = entry.get("stat", "points")
    st.session_state["single_line"] = safe_float(entry.get("line"), 24.5)
    st.session_state["single_opponent"] = entry.get("opponent_display")
    st.session_state["single_notes"] = entry.get("notes", "")
    st.session_state["single_ev_multiplier"] = safe_float(entry.get("ev_multiplier"), 0.0)
    st.session_state["_single_last_auto_player"] = entry.get("player_display")
    st.session_state["_single_auto_opponent"] = entry.get("opponent_display")



def render_single_section():
    render_section_title("Single Prop Analyzer")
    align_section_inputs()
    apply_pending_single_history_entry()
    sync_single_auto_opponent()

    cA, cB, cC = st.columns([7, 1, 1])
    with cB:
        if st.button("Clear", key=fresh_widget_key("clear_single_btn"), use_container_width=True):
            clear_single_inputs()
            st.rerun()
    with cC:
        with st.popover("History", use_container_width=True):
            render_single_history_panel()

    c1, c2 = st.columns(2)
    with c1:
        player = searchable_selectbox("Player Name", get_player_options(), key="single_player", placeholder="Type to search players", on_change=update_auto_opponent, args=("single_player", "single_opponent", "_single_last_auto_player", "_single_auto_opponent"))
        stat = st.selectbox("Stat Type", STAT_OPTIONS, key="single_stat", format_func=stat_display_name)
        line = line_input_widget("Line", "single_line", default=24.5)
        opponent = searchable_selectbox("Opponent Team", get_team_options(), key="single_opponent", placeholder="Type to search opponents")
        auto_opp = st.session_state.get("_single_auto_opponent")
        if player and auto_opp:
            st.caption(f"Auto-filled next opponent: {auto_opp}")
    with c2:
        ev_multiplier = st.number_input("Multiplier (optional)", min_value=0.0, step=0.1, value=0.0, key="single_ev_multiplier")
        notes = st.text_area("Optional Notes", placeholder="Examples: teammate out, back-to-back, starter tonight...", key="single_notes")

    st.markdown("<div class='propify-action-spacer-top'></div>", unsafe_allow_html=True)
    submitted = st.button("Analyze Single Prop", use_container_width=True, key="analyze_single_prop_btn")
    st.markdown("<div class='propify-action-spacer-bottom'></div>", unsafe_allow_html=True)

    if submitted:
        player_name, _ = parse_player_display(player)
        if not player_name or not opponent:
            st.warning("Enter both player and opponent.")
        else:
            result = analyze_prop_with_ml_priority(
                player_name=str(player_name).strip(),
                stat=stat,
                line=safe_float(line, 0.0),
                opponent=str(opponent).strip(),
                user_notes=notes.strip(),
            )
            result["line"] = safe_float(line, 0.0)
            result["stat"] = stat
            result = normalize_projection_fields(result)
            result = apply_app_layer_projection_upgrade(result, stat)
            result = recalculate_current_season_hit_rate(result)
            st.session_state["last_analysis"] = result
            st.session_state["last_pick_meta"] = {
                "player": str(player_name).strip(),
                "stat": stat,
                "line": safe_float(line, 0.0),
                "opponent": str(opponent).strip(),
                "user_notes": notes.strip(),
                "ev_multiplier": safe_float(ev_multiplier, 0.0),
            }
            save_single_history_entry(
                {
                    "player_display": player,
                    "player_name": str(player_name).strip(),
                    "stat": stat,
                    "line": safe_float(line, 0.0),
                    "opponent_display": opponent,
                    "notes": notes.strip(),
                    "ev_multiplier": safe_float(ev_multiplier, 0.0),
                    "analyzed_at": datetime.now(ZoneInfo("America/New_York")).strftime("%-I:%M %p"),
                }
            )

    result = st.session_state.get("last_analysis")
    pick_meta = st.session_state.get("last_pick_meta")
    if result and not result.get("error"):
        result = normalize_projection_fields(result)
        result = recalculate_current_season_hit_rate(result)
        st.session_state["last_analysis"] = result

    if result and not result.get("error"):
        over_display, under_display, push_display = adjusted_display_probs(result, result.get("lean"))
        lean_prob = over_display if result["lean"] == "OVER" else under_display
        ev_percent = calculate_ev_percent(lean_prob, pick_meta.get("ev_multiplier", 0)) if pick_meta else None

        st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.82rem; opacity:0.8; margin:0 0 0.45rem 0;'>{format_model_summary(result)}</div>", unsafe_allow_html=True)

        render_play_summary_strip(result, pick_meta)
        render_top_risk_box(result)

        if get_analysis_mode("single") == "Quick Mode":
            render_quick_mode_single(result, pick_meta)
            st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        else:
            top1, top2, top3, top4, top5 = st.columns(5)
            with top1:
                render_primary_card("Projection", f"{safe_float(result.get('projection'), 0.0):.2f}")
            with top2:
                render_primary_card("Over %", f"{over_display:.1f}%")
            with top3:
                render_primary_card("Under %", f"{under_display:.1f}%")
            with top4:
                render_primary_card("Edge vs Line", f"{safe_float(result.get('edge_vs_line'), 0.0):+.2f}")
            with top5:
                render_primary_card("Season Hit Rate", str(result.get("season_hit_rate", "N/A")))

            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
            sec1, sec2, sec3, sec4, sec5 = st.columns(5)
            with sec1:
                render_secondary_card("Fair Line", str(result.get("fair_line", result.get("projection", "N/A"))))
            with sec2:
                render_secondary_card("Fair Odds O", str(result.get("fair_odds_over", "N/A")))
            with sec3:
                render_secondary_card("Fair Odds U", str(result.get("fair_odds_under", "N/A")))
            with sec4:
                render_secondary_card("Confidence", result["confidence"])
            with sec5:
                render_secondary_card("EV %", f"{ev_percent:.1f}%" if ev_percent is not None else "N/A")

            st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
            render_model_details(result)
            st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
            render_hit_rate_section(result)
            st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
            render_comparable_spots(result, pick_meta)
            st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
            render_why_this_lean(result)
            st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)

            tabs = st.tabs(["Projection & Scores", "Context & Risk", "Recent Sample"])

            with tabs[0]:
                secA, secB, secC = st.columns(3)
                with secA:
                    render_secondary_card("Median", str(result["median_projection"]))
                with secB:
                    render_secondary_card("Floor (20th)", str(result["floor_projection"]))
                with secC:
                    render_secondary_card("Ceiling (80th)", str(result["ceiling_projection"]))

                render_section_header("Model Scores")
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    render_score_card("Recent Form Score", result["recent_form_score"], True)
                with s2:
                    render_score_card("Matchup Score", result["matchup_score"], True)
                with s3:
                    render_score_card("Volatility Score", result["volatility_score"], False)
                with s4:
                    render_score_card("Minutes Risk Score", result["minutes_risk_score"], False)

                render_section_header("Key Statistics")
                reasons = [
                    f"Season average: {safe_float(result.get('season_average'), 0.0):.2f}",
                    f"Season median: {safe_float(result.get('season_median'), 0.0):.2f}",
                    f"Weighted recent average: {safe_float(result.get('weighted_recent_average'), 0.0):.2f}",
                    f"Last 5 average: {safe_float(result.get('last5_average'), 0.0):.2f}",
                    f"Last 10 average: {safe_float(result.get('last10_average'), 0.0):.2f}",
                    f"Last 15 average: {safe_float(result.get('last15_average'), 0.0):.2f}",
                    f"Base projected minutes: {safe_float(result.get('projected_minutes'), 0.0):.1f}",
                    f"Enhanced minutes: {safe_float(result.get('enhanced_projected_minutes'), 0.0):.1f}",
                    f"Rotation role: {result.get('rotation_role_label', 'N/A')}",
                    f"Archetype: {result.get('stat_archetype', 'neutral')}",
                    f"Enhanced matchup multiplier: {safe_float(result.get('matchup_multiplier_enhanced'), 1.0):.3f}",
                    f"Season hit rate over line: {result['season_hit_rate']}",
                    f"Last 10 hit rate over line: {result['last10_hit_rate']}",
                    f"Last 5 hit rate over line: {result['last5_hit_rate']}",
                ]
                reason_cols = st.columns(4)
                for idx, item in enumerate(reasons):
                    with reason_cols[idx % 4]:
                        st.write(f"• {item}")

                render_section_header("Distribution View")
                render_projected_outcome_distribution(result, safe_float((pick_meta or {}).get("line", 0.0), 0.0), stat_display_name((pick_meta or {}).get("stat", "")))

            with tabs[1]:
                if str(pick_meta["stat"]).lower() in {"fga", "2pa", "3pa", "fta"}:
                    st.subheader("Shot-Volume Visual")
                    recent_col_map = {"fga": "FGA", "2pa": "2PA", "3pa": "FG3A", "fta": "FTA"}
                    recent_col = recent_col_map.get(str(pick_meta["stat"]).lower())
                    recent_df = result["recent_games_df"].copy()
                    if recent_col in recent_df.columns:
                        st.line_chart(recent_df.iloc[::-1][recent_col], use_container_width=True)

                render_section_header("Context")
                _, single_team_abbr = parse_player_display(pick_meta.get("player") if pick_meta else "")
                _, upcoming_home_away = get_next_matchup_context(single_team_abbr)
                cx1, cx2, cx3, cx4 = st.columns(4)
                with cx1:
                    st.metric("Projected Minutes", safe_float(result.get("enhanced_projected_minutes"), result.get("projected_minutes")))
                with cx2:
                    st.metric("Rest Days", result["rest_days"])
                with cx3:
                    st.metric("Home/Away", upcoming_home_away if single_team_abbr else result.get("home_away", "Neutral"))
                with cx4:
                    st.metric("Pace Proxy", result["pace_proxy"])

                cx5, cx6, cx7, cx8 = st.columns(4)
                with cx5:
                    st.metric("Team Pace", result.get("team_pace_proxy", "N/A"))
                with cx6:
                    st.metric("Opponent Pace", result.get("opp_pace_proxy", "N/A"))
                with cx7:
                    st.metric("Role", result.get("rotation_role_label", result.get("role_stability_label", "N/A")))
                with cx8:
                    st.metric("Opp B2B", result.get("opp_back_to_back", 0))

                render_thin_section_line()
                render_section_header("Risk Flags")
                risk_items = [get_top_risk_reason(result)] + [x for x in (result.get("risks") or []) if str(x).strip()]
                risk_cols = st.columns(4)
                for idx, item in enumerate(risk_items[:8]):
                    with risk_cols[idx % 4]:
                        st.write(f"• {item}")

            with tabs[2]:
                render_section_header("Recent Sample")
                sample_df = result["recent_games_df"].copy()
                if not sample_df.empty:
                    render_recent_sample_table(
                        sample_df,
                        stat_key=pick_meta.get("stat") if pick_meta else None,
                        line_value=pick_meta.get("line") if pick_meta else None,
                        target_opponent=pick_meta.get("opponent") if pick_meta else None,
                    )
                else:
                    st.info("No recent sample data available.")

        st.markdown("<div style='height:2.6rem;'></div>", unsafe_allow_html=True)
        if st.button("Save Single Pick", use_container_width=True):
            st.session_state["open_single_save_dialog"] = True
            st.rerun()
        st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif result and result.get("error"):
        st.error(result["error"])



def render_leg_inputs(section_key, legs_count):
    inputs = []
    cols = st.columns(2)
    player_options = get_player_options()
    team_options = get_team_options()
    for i in range(legs_count):
        with cols[i % 2]:
            sync_parlay_auto_opponent(section_key, i)
            st.markdown(f'<div class="propify-leg-label">Leg {i + 1}</div>', unsafe_allow_html=True)
            player = searchable_selectbox("Player Name", player_options, key=f"{section_key}_player_{i}", placeholder=f"Type to search player {i + 1}", on_change=update_auto_opponent, args=(f"{section_key}_player_{i}", f"{section_key}_opp_{i}", f"_{section_key}_last_auto_player_{i}", f"_{section_key}_auto_opponent_{i}"))
            stat = st.selectbox(
                "Stat Type",
                STAT_OPTIONS,
                key=f"{section_key}_stat_{i}",
                format_func=stat_display_name,
            )
            side = st.selectbox("Pick Side", ["OVER", "UNDER"], key=f"{section_key}_side_{i}")
            line = line_input_widget("Line", f"{section_key}_line_{i}", default=10.5)
            opponent = searchable_selectbox("Opponent Team", team_options, key=f"{section_key}_opp_{i}", placeholder="Type to search opponents")
            auto_opp = st.session_state.get(f"_{section_key}_auto_opponent_{i}")
            if player and auto_opp:
                st.caption(f"Auto-filled next opponent: {auto_opp}")
            inputs.append({"player": player, "stat": stat, "side": side, "line": safe_float(line, 0.0), "opponent": opponent})
    return inputs



def analyze_parlay_legs(legs):
    results = []
    for leg in legs:
        player_name, _ = parse_player_display(leg["player"])
        leg_result = analyze_prop_with_ml_priority(
            player_name=str(player_name).strip(),
            stat=leg["stat"],
            line=safe_float(leg["line"], 0.0),
            opponent=str(leg["opponent"]).strip(),
        )
        leg_result["line"] = safe_float(leg["line"], 0.0)
        leg_result = normalize_projection_fields(leg_result)
        leg_result = recalculate_current_season_hit_rate(leg_result)
        results.append(leg_result)
    return results



def render_leg_card(leg_result, leg_meta, leg_index, show_explanations: bool = False):
    summary_line = get_parlay_leg_summary(leg_result, leg_meta)
    st.markdown(
        f"**Leg {leg_index}: {leg_meta['player']} {leg_meta['side']} {leg_meta['line']} {stat_display_name(leg_meta['stat'])}**"
    )
    st.caption(summary_line)
    m1, m2, m3, m4 = st.columns(4)
    metric_rows = [
        (m1, "Projection", f"{safe_float(leg_result.get('projection'), 0.0):.2f}"),
        (m2, "Last 5 Average", f"{safe_float(leg_result.get('last5_average'), 0.0):.2f}"),
        (m3, "Last 15 Average", f"{safe_float(leg_result.get('last15_average'), 0.0):.2f}"),
        (m4, "Season Average", f"{safe_float(leg_result.get('season_average'), 0.0):.2f}"),
    ]
    for col, label, value in metric_rows:
        with col:
            st.metric(label, value)

    m5, m6, m7, m8 = st.columns(4)
    leg_prob = selected_side_prob(leg_result, leg_meta["side"])
    season_hit_rate = leg_result.get("season_hit_rate", "N/A")
    if season_hit_rate in (None, ""):
        season_hit_rate = "N/A"
    metric_rows2 = [
        (m5, "Selected Side", leg_meta["side"]),
        (m6, "Selected Side %", f"{leg_prob:.1f}%"),
        (m7, "Confidence", leg_result["confidence"]),
        (m8, "Season Hit Rate", season_hit_rate),
    ]
    for col, label, value in metric_rows2:
        with col:
            st.metric(label, value)




def render_parlay_section(legs_count):
    render_section_title(f"{legs_count}-Leg Parlay Builder")
    align_section_inputs()
    section_key = f"parlay_{legs_count}"

    _, header_right = st.columns([8.2, 1.0])
    with header_right:
        if st.button("Clear", key=fresh_widget_key(f"clear_{section_key}_btn"), use_container_width=True):
            clear_parlay_inputs(section_key, legs_count)
            st.rerun()

    legs = render_leg_inputs(section_key, legs_count)
    extra1, extra2 = st.columns(2)
    with extra1:
        ev_multiplier = st.number_input("Multiplier (optional)", min_value=0.0, step=0.1, value=0.0, key=f"{section_key}_ev_multiplier")
    with extra2:
        notes = st.text_input("Notes", key=f"{section_key}_notes")

    st.markdown("<div class='propify-action-spacer-top'></div>", unsafe_allow_html=True)
    submitted = st.button(f"Analyze {legs_count}-Leg Parlay", use_container_width=True, key=f"analyze_{section_key}_btn")
    st.markdown("<div class='propify-action-spacer-bottom'></div>", unsafe_allow_html=True)

    if submitted:
        resolved_legs = []
        for i, leg in enumerate(legs):
            player_display = leg["player"]
            player_name, team_abbr = parse_player_display(player_display)
            final_opp = leg["opponent"]
            if team_abbr:
                next_opp, _ = get_next_matchup_context(team_abbr)
                if next_opp:
                    final_opp = next_opp
            resolved_legs.append({
                "player": player_display,
                "stat": leg["stat"],
                "side": leg["side"],
                "line": float(leg["line"]),
                "opponent": final_opp,
            })

        valid = all(leg["player"] and leg["opponent"] for leg in resolved_legs)
        if not valid:
            st.warning(f"Enter all players and opponents for the {legs_count}-leg parlay.")
        else:
            results = analyze_parlay_legs(resolved_legs)
            upgraded = []
            for i, leg_result in enumerate(results):
                leg_result["stat"] = resolved_legs[i]["stat"]
                upgraded.append(apply_app_layer_projection_upgrade(leg_result, resolved_legs[i]["stat"]))
            st.session_state[section_key] = {
                "legs": upgraded,
                "meta": {
                    "players": [parse_player_display(x["player"])[0] for x in resolved_legs],
                    "stats": [x["stat"] for x in resolved_legs],
                    "sides": [x["side"] for x in resolved_legs],
                    "lines": [safe_float(x["line"], 0.0) for x in resolved_legs],
                    "opponents": [str(x["opponent"]).strip() for x in resolved_legs],
                    "notes": notes.strip(),
                    "ev_multiplier": safe_float(ev_multiplier, 0.0),
                }
            }

    data = st.session_state.get(section_key)
    if not data:
        return

    results = data["legs"]
    meta = data["meta"]
    if any(r.get("error") for r in results):
        for i, r in enumerate(results, start=1):
            if r.get("error"):
                st.error(f"Leg {i} error: {r['error']}")
        return

    probs = [selected_side_prob(r, meta["sides"][i]) / 100 for i, r in enumerate(results)]
    parlay_hit_prob = 1.0
    for p in probs:
        parlay_hit_prob *= p
    parlay_miss_prob = 1 - parlay_hit_prob
    combined_conf, _ = combine_confidence_labels([r["confidence"] for r in results])
    ev_percent = calculate_ev_percent(parlay_hit_prob * 100, meta.get("ev_multiplier", 0)) if meta.get("ev_multiplier", 0) > 0 else None

    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    mode = get_analysis_mode(section_key)

    flex_probability = None
    flex_label = None
    if legs_count == 3:
        flex_probability = calculate_flex_probability(probs, 2)
        flex_label = "Flex Probability (chance 2+ picks hit)"
    elif legs_count == 4:
        flex_probability = calculate_flex_probability(probs, 3)
        flex_label = "Flex Probability (chance 3+ picks hit)"

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        render_primary_card("Hit Probability", f"{parlay_hit_prob * 100:.1f}%", None)
    with p2:
        render_primary_card("Miss Probability", f"{parlay_miss_prob * 100:.1f}%", None)
    with p3:
        render_primary_card("Confidence", combined_conf, None)
    with p4:
        if flex_probability is not None:
            render_primary_card(flex_label, f"{flex_probability * 100:.1f}%", None)
        else:
            render_primary_card("Entry Type", entry_type_label(legs_count).title(), None)
    with p5:
        render_primary_card("EV %", f"{ev_percent:.1f}%" if ev_percent is not None else "N/A", None)

    render_section_header("Sticky Context")
    sticky_cols = st.columns(legs_count)
    for i, col in enumerate(sticky_cols):
        with col:
            st.caption(get_parlay_leg_summary(results[i], {
                "side": meta["sides"][i],
                "line": meta["lines"][i],
                "stat": meta["stats"][i],
                "player": meta["players"][i],
            }))

    if get_analysis_mode("single") == "Quick Mode":
        render_section_header("Leg Summaries")
        for start_idx in range(0, legs_count, 2):
            cols = st.columns(min(2, legs_count - start_idx))
            for j, col in enumerate(cols):
                idx = start_idx + j
                with col:
                    st.markdown(f"**Leg {idx+1}: {meta['players'][idx]}**")
                    st.write(get_parlay_leg_summary(results[idx], {
                        "side": meta["sides"][idx],
                        "line": meta["lines"][idx],
                        "stat": meta["stats"][idx],
                        "player": meta["players"][idx],
                    }))
    else:
        render_section_header("Leg Summaries")
        for start_idx in range(0, legs_count, 2):
            cols = st.columns(min(2, legs_count - start_idx))
            for j, col in enumerate(cols):
                idx = start_idx + j
                with col:
                    render_leg_card(
                        results[idx],
                        {
                            "side": meta["sides"][idx],
                            "player": meta["players"][idx],
                            "stat": meta["stats"][idx],
                            "line": meta["lines"][idx],
                            "opponent": meta["opponents"][idx],
                        },
                        idx + 1,
                        False,
                    )

    st.markdown("<div style='height:2.6rem;'></div>", unsafe_allow_html=True)
    if st.button(f"Save {legs_count}-Leg Parlay", use_container_width=True, key=f"{section_key}_save"):
        st.session_state[f"open_{section_key}_save_dialog"] = True
        st.rerun()
    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



def render_grading_section():
    st.subheader("Track Saved Picks")
    history_df = load_pick_history()
    pending_df = history_df[history_df["outcome"].fillna("").astype(str).str.strip() == ""].copy()

    if pending_df.empty:
        st.info("No pending saved picks.")
        return

    pending_options = {f"{format_saved_display(row.get('saved_at'))} | {row.get('label', '')}": row["pick_id"] for _, row in pending_df.iterrows()}
    selected_label = st.selectbox("Pending Pick", list(pending_options.keys()))
    selected_pick_id = pending_options[selected_label]
    selected_row = pending_df[pending_df["pick_id"].astype(str) == str(selected_pick_id)].iloc[0]
    selected_type = str(selected_row["entry_type"]).lower()
    legs_count = int(pd.to_numeric(selected_row["legs_count"], errors="coerce") or 1)

    if selected_type == "single":
        with st.form("settle_single_form"):
            actual_result = st.number_input("Actual Final Stat Result", min_value=0.0, step=0.5, value=0.0)
            settle_submitted = st.form_submit_button("Mark Single Result")
        if settle_submitted:
            ok, message = settle_pick_result(selected_pick_id, actual_result=safe_float(actual_result, 0.0))
            if ok:
                st.success(f"Pick graded as: {message.upper()}")
            else:
                st.error(message)
    else:
        with st.form("settle_parlay_form"):
            actual_results = []
            for i in range(1, legs_count + 1):
                actual = st.number_input(
                    f"Leg {i} Result — {selected_row[f'leg{i}_player']} {stat_display_name(selected_row[f'leg{i}_stat'])}",
                    min_value=0.0,
                    step=0.5,
                    value=0.0,
                    key=f"settle_leg_{i}",
                )
                actual_results.append(safe_float(actual, 0.0))

            settle_submitted = st.form_submit_button(f"Mark {entry_type_label(legs_count).title()} Result")
        if settle_submitted:
            ok, message = settle_pick_result(selected_pick_id, actual_leg_results=actual_results)
            if ok:
                st.success(f"{entry_type_label(legs_count).title()} graded as: {message.upper()}")
            else:
                st.error(message)


def render_summary_section(title: str, df: pd.DataFrame):
    st.markdown(f"### {title}")
    summary = compute_record_summary(df)
    cols = st.columns(4)
    cols[0].metric("Record", f"{summary['wins']}-{summary['losses']}")
    cols[1].metric("Win Rate", summary["win_rate"])
    cols[2].metric("Net Profit", f"${summary['net_profit']:,.2f}")
    cols[3].metric("ROI", summary["roi"])


def compute_record_summary(df: pd.DataFrame):
    if df.empty:
        return {
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "pending": 0,
            "win_rate": "N/A",
            "net_profit": 0.0,
            "roi": "N/A",
        }

    outcome_series = df["outcome"].fillna("").str.lower()
    graded = df[outcome_series.isin(["win", "loss", "push", "cash out"])].copy()
    pending = df[~outcome_series.isin(["win", "loss", "push", "cash out"])].copy()

    wins = int((graded["outcome"].str.lower() == "win").sum()) if not graded.empty else 0
    losses = int((graded["outcome"].str.lower() == "loss").sum()) if not graded.empty else 0
    pushes = int((graded["outcome"].str.lower() == "push").sum()) if not graded.empty else 0
    decision_count = wins + losses
    win_rate = f"{(wins / decision_count) * 100:.1f}%" if decision_count > 0 else "N/A"

    net_profit = df.apply(
        lambda row: compute_net_profit_value(
            row.get("outcome", ""),
            pd.to_numeric(row.get("bet_size"), errors="coerce"),
            pd.to_numeric(row.get("multiplier"), errors="coerce"),
            pd.to_numeric(row.get("cash_out_price"), errors="coerce"),
        ),
        axis=1,
    ).sum()
    total_risked = pd.to_numeric(df["bet_size"], errors="coerce").fillna(0).sum()
    roi = f"{(net_profit / total_risked) * 100:.1f}%" if total_risked > 0 else "N/A"

    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "pending": len(pending),
        "win_rate": win_rate,
        "net_profit": net_profit,
        "roi": roi,
    }




def _safe_event_datetime(series):
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return parsed.dt.tz_convert(None)
    except Exception:
        try:
            return parsed.dt.tz_localize(None)
        except Exception:
            return pd.to_datetime(parsed, errors="coerce")



def render_recent_sample_table(
    display_df: pd.DataFrame,
    stat_key: str | None = None,
    line_value: float | None = None,
    target_opponent: str | None = None,
):
    if display_df.empty:
        st.info("No recent sample data available.")
        return

    controls = st.columns([1.12, 1.02, 1.18, 1.08])
    with controls[0]:
        st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)
        show_season_avg = st.toggle("Season Avg Line", value=True, key="single_rs_season_avg")
    with controls[1]:
        st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)
        show_prop_line = st.toggle("Prop Line", value=True, key="single_rs_prop_line")
    with controls[2]:
        st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)
        opp_only = st.toggle("Opponent-Specific", value=False, key="single_rs_opp_only")
    with controls[3]:
        sample_window = st.selectbox(
            "Window",
            ["Last 10", "Season", "All-Time"],
            index=1,
            key="single_recent_sample_window",
        )

    table_df, stat_col = prepare_recent_stat_column(display_df, str(stat_key or ""))
    if table_df.empty:
        st.info("No recent sample data available.")
        return

    if "Game Date" in table_df.columns:
        table_df["Game Date"] = pd.to_datetime(table_df["Game Date"], errors="coerce")
        if sample_window == "Last 10":
            table_df = table_df.sort_values("Game Date", ascending=False).head(10).copy()
        elif sample_window == "Season":
            table_df = table_df[table_df["Game Date"] >= pd.Timestamp("2025-10-01")].copy()
            table_df = table_df.sort_values("Game Date", ascending=False)
        elif sample_window == "All-Time":
            table_df = table_df[table_df["Game Date"] >= pd.Timestamp("2023-10-01")].copy()
            table_df = table_df.sort_values("Game Date", ascending=False)

    if opp_only:
        team_variants = expand_team_filter_variants(target_opponent)
        filtered = table_df.iloc[0:0].copy()

        if team_variants:
            opp_col = None
            matchup_col = None
            for cand in ["Opponent", "Opp"]:
                if cand in table_df.columns:
                    opp_col = cand
                    break
            for cand in ["Matchup", "MATCHUP"]:
                if cand in table_df.columns:
                    matchup_col = cand
                    break

            if opp_col:
                filtered = table_df[
                    table_df[opp_col].apply(lambda x: text_contains_team_variant(x, team_variants))
                ].copy()

            if matchup_col:
                matchup_filtered = table_df[
                    table_df[matchup_col].apply(lambda x: text_contains_team_variant(x, team_variants))
                ].copy()
                if filtered.empty:
                    filtered = matchup_filtered
                elif not matchup_filtered.empty:
                    filtered = pd.concat([filtered, matchup_filtered], ignore_index=True).drop_duplicates()

        table_df = filtered

    if table_df.empty:
        st.info("No recent sample rows match the selected filters.")
        return

    if "Game Date" in table_df.columns:
        table_df["Game Date"] = pd.to_datetime(table_df["Game Date"], errors="coerce").dt.strftime("%m/%d/%Y")

    desired_order = [
        "Game Date", "Matchup", "W/L", "MIN", "Home/Away", "Opponent",
        "PTS", "REB", "AST", "PRA", "PR", "PA", "RA",
        "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "STL", "BLK", "TOV", "PF",
        "Pick Result"
    ]
    display_cols = [c for c in desired_order if c in table_df.columns]
    for c in table_df.columns:
        if c not in display_cols and c != "__row_type__":
            display_cols.append(c)

    footer_values = {}
    if stat_col and stat_col in table_df.columns:
        footer_values[stat_col] = pd.to_numeric(table_df[stat_col], errors="coerce").mean()
    if "MIN" in table_df.columns:
        footer_values["MIN"] = pd.to_numeric(table_df["MIN"], errors="coerce").mean()

    avg_row = {col: "" for col in display_cols}
    if display_cols:
        avg_row[display_cols[0]] = "AVG"
    for col, val in footer_values.items():
        avg_row[col] = round(float(val), 1) if pd.notna(val) else ""

    def fmt_value(val):
        if pd.isna(val) or val == "":
            return ""
        if isinstance(val, (int, float)):
            return str(int(val)) if float(val).is_integer() else f"{val:.1f}"
        return str(val)

    def row_result_style(row):
        outcome = str(row.get("Pick Result", "")).strip().lower()
        if outcome == "over":
            return "background:#e7f6ea;"
        if outcome == "under":
            return "background:#fdecec;"
        if outcome == "push":
            return "background:#fff7d6;"
        return ""

    def cell_style(col, row):
        styles = []
        if col == stat_col:
            styles.append("font-weight:700;")
        if col == "Game Date":
            styles.append("background:#ffffff;")
        else:
            styles.append(row_result_style(row))
        return ''.join(styles)

    header_html = ''.join(f'<th>{c}</th>' for c in display_cols)
    body_rows = []
    for _, row in table_df.iterrows():
        cells = ''.join(
            f'<td style="{cell_style(col, row)}">{fmt_value(row.get(col, ""))}</td>'
            for col in display_cols
        )
        body_rows.append(f'<tr>{cells}</tr>')

    overlay_html = ""
    if stat_col and stat_col in table_df.columns:
        stat_series = pd.to_numeric(table_df[stat_col], errors="coerce").dropna()
        info_bits = []
        if show_season_avg and not stat_series.empty:
            info_bits.append(f"Season Avg: {stat_series.mean():.1f}")
        if show_prop_line and line_value is not None:
            info_bits.append(f"Prop Line: {safe_float(line_value, 0.0):g}")
        if info_bits:
            overlay_html = f'<div style="font-size:0.82rem;color:#c9d6e5;margin:0.35rem 0 0.4rem 0;">{" | ".join(info_bits)}</div>'

    footer_cells = ''.join(
        f'<td>{"AVG" if col == display_cols[0] else fmt_value(avg_row.get(col, ""))}</td>'
        for col in display_cols
    )
    footer_html = f'<tr style="background:#e5e7eb; font-weight:600;">{footer_cells}</tr>'

    table_html = f"""
    <style>
    .recent-sample-wrap {{
        width:100%;
        overflow-x:auto;
        border-radius:18px;
        background:#ffffff;
        border:1px solid rgba(18,34,53,0.10);
        box-shadow:0 8px 22px rgba(15,23,42,0.08);
        padding:12px 12px 6px 12px;
    }}
    .recent-sample-table {{
        width:100%;
        border-collapse:collapse;
        font-size:0.92rem;
        color:#122235;
    }}
    .recent-sample-table th {{
        position:sticky; top:0; background:#eff4f8; z-index:1;
        padding:9px 8px; border-bottom:1px solid #d6dee8; text-align:center; font-weight:700;
    }}
    .recent-sample-table td {{
        padding:8px 8px; border-bottom:1px solid #eef2f7; text-align:center;
    }}
    </style>
    <div class="recent-sample-wrap">
      {overlay_html}
      <table class="recent-sample-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
        <tfoot>{footer_html}</tfoot>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

def render_all_time_profit_chart(df: pd.DataFrame):
    if df.empty:
        st.info("No saved picks yet.")
        return

    filter_col_left, filter_col_right = st.columns([5, 1])
    with filter_col_right:
        range_filter = st.selectbox(
            "Range",
            ["All-Time", "Past Month", "Past 14 Days", "Past Week"],
            index=0,
            key="all_time_profit_range_filter",
        )

    chart_df = df.copy()
    settled_series = _safe_event_datetime(chart_df.get("settled_at"))
    saved_series = _safe_event_datetime(chart_df.get("saved_at"))
    if len(settled_series) != len(chart_df):
        settled_series = pd.Series([pd.NaT] * len(chart_df), index=chart_df.index, dtype="datetime64[ns]")
    if len(saved_series) != len(chart_df):
        saved_series = pd.Series([pd.NaT] * len(chart_df), index=chart_df.index, dtype="datetime64[ns]")
    chart_df["event_dt"] = settled_series.combine_first(saved_series)
    chart_df = chart_df.dropna(subset=["event_dt"]).copy()
    if chart_df.empty:
        st.info("No dated picks available yet.")
        return

    chart_df["profit_change"] = chart_df.apply(
        lambda row: compute_net_profit_value(
            row.get("outcome", ""),
            pd.to_numeric(row.get("bet_size"), errors="coerce"),
            pd.to_numeric(row.get("multiplier"), errors="coerce"),
            pd.to_numeric(row.get("cash_out_price"), errors="coerce"),
        ),
        axis=1,
    )
    chart_df = chart_df.sort_values("event_dt", kind="mergesort")
    end_dt = chart_df["event_dt"].max()
    if range_filter == "Past Week":
        chart_df = chart_df[chart_df["event_dt"] >= end_dt - pd.Timedelta(days=7)]
    elif range_filter == "Past 14 Days":
        chart_df = chart_df[chart_df["event_dt"] >= end_dt - pd.Timedelta(days=14)]
    elif range_filter == "Past Month":
        chart_df = chart_df[chart_df["event_dt"] >= end_dt - pd.Timedelta(days=30)]

    if chart_df.empty:
        st.info("No picks available in this date range.")
        return

    # One point per day: sum each day's net change, then build cumulative profit
    chart_df["event_date"] = pd.to_datetime(chart_df["event_dt"]).dt.normalize()
    daily_df = (
        chart_df.groupby("event_date", as_index=False)["profit_change"]
        .sum()
        .sort_values("event_date", kind="mergesort")
    )
    daily_df["cum_profit"] = daily_df["profit_change"].cumsum()

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    main_color = "#2f3b52"
    trend_color = "#b7efc5" if float(daily_df["cum_profit"].iloc[-1]) >= 0 else "#f4b7b7"

    ax.plot(daily_df["event_date"], daily_df["cum_profit"], linewidth=1.7, color=main_color, zorder=3)
    ax.scatter(daily_df["event_date"], daily_df["cum_profit"], s=18, color=main_color, zorder=4)

    for x_val, y_val in zip(daily_df["event_date"], daily_df["cum_profit"]):
        ax.text(
            x_val,
            y_val + 4,
            f"${y_val:.2f}",
            fontsize=8,
            ha="center",
            va="bottom",
            color=main_color,
            zorder=5,
        )

    if len(daily_df) >= 2:
        x = mdates.date2num(daily_df["event_date"])
        z = np.polyfit(x, daily_df["cum_profit"], 1)
        p = np.poly1d(z)
        ax.plot(daily_df["event_date"], p(x), linewidth=1.1, color=trend_color, alpha=0.95, zorder=2)

    y_min = float(daily_df["cum_profit"].min())
    y_max = float(daily_df["cum_profit"].max())
    spread = y_max - y_min
    padding = max(5.0, spread * 0.10 if spread > 0 else max(abs(y_max), 10.0) * 0.10)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Net Profit", fontsize=11)

    ax.grid(which="major", color="#dfe4ea", linewidth=0.6, alpha=0.9)
    ax.grid(which="minor", color="#eef2f6", linewidth=0.35, alpha=1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d/%y"))
    except Exception:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color("#aab4c3")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_tracker_dashboard():
    render_section_title("Pick Tracker")

    history_df = load_pick_history()
    working_df = history_df.copy()
    if not working_df.empty:
        working_df["type_label"] = working_df.apply(get_pick_type_label, axis=1)

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        stat_choices = ["All"] + sorted([str(x) for x in working_df["stat"].dropna().astype(str).unique().tolist()]) if not working_df.empty else ["All"]
        filter_stat = st.selectbox("Stat Filter", stat_choices, format_func=lambda x: "All" if x == "All" else stat_display_name(x))
    with f2:
        filter_type = st.selectbox("Type Filter", ["All", "Power", "Flex"])
    with f3:
        filter_conf = st.selectbox("Confidence Filter", ["All", "Low", "Low-Medium", "Medium", "Medium-High", "High"])
    with f4:
        filter_outcome = st.selectbox("Outcome Filter", ["All", "win", "loss", "push", "cash out"])
    with f5:
        filter_search = st.text_input("Search Saved Picks", placeholder="Player, label, stat...")

    dashboard_df = working_df.copy()
    if filter_stat != "All":
        dashboard_df = dashboard_df[dashboard_df["stat"].astype(str) == filter_stat]
    if filter_type != "All":
        dashboard_df = dashboard_df[dashboard_df["type_label"] == filter_type]
    if filter_conf != "All":
        dashboard_df = dashboard_df[dashboard_df["confidence"].astype(str) == filter_conf]
    if filter_outcome != "All":
        dashboard_df = dashboard_df[dashboard_df["outcome"].fillna("").astype(str).str.lower() == filter_outcome]
    if filter_search.strip():
        q = filter_search.strip().lower()
        dashboard_df = dashboard_df[
            dashboard_df["label"].astype(str).str.lower().str.contains(q)
            | dashboard_df["player"].astype(str).str.lower().str.contains(q)
            | dashboard_df["stat"].astype(str).str.lower().str.contains(q)
        ]

    summary = compute_record_summary(dashboard_df)

    top_row_1 = st.columns(4)
    top_row_1[0].metric("Wins", summary["wins"])
    top_row_1[1].metric("Losses", summary["losses"])
    top_row_1[2].metric("Pushes", summary["pushes"])
    top_row_1[3].metric("Pending", summary["pending"])

    top_row_2 = st.columns(3)
    top_row_2[0].metric("Win Rate", summary["win_rate"])
    top_row_2[1].metric("Net Profit", f"${summary['net_profit']:,.2f}")
    top_row_2[2].metric("ROI", summary["roi"])

    outcome_series = dashboard_df["outcome"].fillna("").astype(str).str.lower() if not dashboard_df.empty else pd.Series(dtype=str)
    completed_df = dashboard_df[outcome_series.isin(["win", "loss", "push", "cash out"])].copy() if not dashboard_df.empty else dashboard_df.copy()
    if not completed_df.empty and "saved_at" in completed_df.columns:
        completed_df["_saved_dt"] = pd.to_datetime(completed_df["saved_at"], errors="coerce")
        completed_df = completed_df.sort_values("_saved_dt", ascending=False)
        last_five_df = completed_df.head(5).drop(columns=["_saved_dt"], errors="ignore")
        cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=30)
        month_df = completed_df[completed_df["_saved_dt"] >= cutoff].drop(columns=["_saved_dt"], errors="ignore")
    else:
        last_five_df = completed_df.copy()
        month_df = completed_df.copy()

    render_summary_section("Last 5 Entries", last_five_df)
    render_summary_section("Last Month", month_df)

    render_saved_entries_editor(history_df, dashboard_df)
    st.markdown("<div style='height:34px;'></div>", unsafe_allow_html=True)
    st.subheader("Profit Over Time")
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    render_all_time_profit_chart(dashboard_df)



inject_css()
ensure_pick_history_file()
ensure_json_file(FAVORITES_FILE, [])
ensure_json_file(RECENT_SEARCHES_FILE, [])

if "last_analysis" not in st.session_state:
    st.session_state["last_analysis"] = None
if "last_pick_meta" not in st.session_state:
    st.session_state["last_pick_meta"] = None
if "open_single_save_dialog" not in st.session_state:
    st.session_state["open_single_save_dialog"] = False

if "editing_pick_id" not in st.session_state:
    st.session_state["editing_pick_id"] = None
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True
if "app_view" not in st.session_state:
    st.session_state["app_view"] = "home"
if "propify_force_tab" not in st.session_state:
    st.session_state["propify_force_tab"] = None
if "nav_open" not in st.session_state:
    st.session_state["nav_open"] = False
if "current_section" not in st.session_state:
    st.session_state["current_section"] = "Overview"


for key in ["parlay_2", "parlay_3", "parlay_4"]:
    if key not in st.session_state:
        st.session_state[key] = None
    if f"open_{key}_save_dialog" not in st.session_state:
        st.session_state[f"open_{key}_save_dialog"] = False

user = render_auth_gate()
if user is None:
    st.stop()
st.session_state["auth_user"] = user

render_mode_toggle()

if st.session_state.get("app_view") == "home":
    render_home_screen()
else:
    render_main_navigation_layout()
    render_footer()

if st.session_state.get("open_single_save_dialog"):
    save_single_pick_dialog()
if st.session_state.get("open_parlay_2_save_dialog"):
    save_parlay_dialog("parlay_2", 2)
if st.session_state.get("open_parlay_3_save_dialog"):
    save_parlay_dialog("parlay_3", 3)
if st.session_state.get("open_parlay_4_save_dialog"):
    save_parlay_dialog("parlay_4", 4)

