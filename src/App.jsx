import React, { useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowRight,
  Briefcase,
  Code2,
  Dumbbell,
  ExternalLink,
  FileText,
  GraduationCap,
  HandHelping,
  Home,
  LineChart,
  Linkedin,
  Mail,
  Medal,
  NotebookText,
  Phone,
  ShieldCheck,
  Sparkles,
  Trophy,
} from 'lucide-react'

import headshot from './assets/cleanerheadshot.jpeg'
import propifyLogo from './assets/proptrans.png'
import dhLogo from './assets/dhtrans.png'
import anthropologyAward from './assets/anthropologycert.png'
import thecreek from './assets/thecreek.png'
import unccLogo from './assets/uncclogotrans.png'
import cvccLogo from './assets/cvcctrans2!.png'

const NAV_ITEMS = [
  { label: 'Home', key: 'home', icon: Home },
  { label: 'Experience', key: 'experience', icon: Briefcase },
  { label: 'Projects', key: 'projects', icon: Code2 },
  { label: 'Impact', key: 'skills-proof', icon: ShieldCheck },
  { label: 'Education', key: 'education', icon: GraduationCap },
  { label: 'Resume', key: 'resume', icon: NotebookText },
  { label: 'Interests', key: 'interests', icon: Trophy },
  { label: 'Contact', key: 'contact', icon: Mail },
]

const STATS = [
  { value: '20+', label: 'Executive Dashboards' },
  { value: '$3,000+', label: 'Philanthropy Funds Raised' },
  { value: '3.84', label: 'Current UNCC GPA' },
  { value: '1', label: 'Trademarked Product' },
]

const PROJECTS = [
  {
    tag: 'Product Platform',
    title: 'Propify',
    subtitle: 'Full-stack NBA player prop analytics platform with a public-facing demo and private production system.',
    description:
      'Propify is a trademarked sports analytics platform built from scratch to turn sports betting into a more disciplined, data-driven decision process through modeling, user-specific tracking, and product-minded design.',
    bullets: [
      'The engine combines a rules-based statistical system with a trained Random Forest Regressor, pulling NBA game log data, weighting recent performance across rolling windows, and blending ML and statistical projections to generate hit probabilities, confidence indicators, and fair-value outputs for each prop.',
      'The application includes a polished single-prop workflow, multi-leg parlay analysis, authenticated user access, protected pick tracking, grading workflows, and ROI and profit monitoring so performance can be measured over time.',
      'The private production version is positioned like a real software product, with clear subscription potential built around repeat usage, premium analytics access, and a scalable decision-support experience.',
    ],
    stack: ['Python', 'Streamlit', 'Supabase', 'Postgres', 'Auth', 'nba_api', 'Pandas', 'NumPy', 'Matplotlib', 'scikit-learn'],
    primary: { label: 'Live Demo', href: 'https://propifydemo.streamlit.app/' },
    secondary: { label: 'GitHub Repository', href: 'https://github.com/DavisHiggins/PropifyDemo' },
    accent: 'from-sky-500/25 via-cyan-400/10 to-indigo-500/20',
    logo: propifyLogo,
  },
  {
    tag: 'Statistical Analysis',
    title: 'Hate Crime Data Analysis',
    subtitle: 'In-depth analysis of social issues and their impact across different time periods, offense categories, and demographic patterns.',
    description:
      'This project used Python and SQL to analyze NYPD hate crime datasets across pre-COVID, during-COVID, and post-COVID periods in order to understand how major external events affected crime patterns and bias-related incident behavior.',
    bullets: [
      'Cleaned and structured raw data so the analysis would be reliable, consistent, and usable.',
      'Built visualizations to identify changes in frequency, offense categories, and bias patterns across time periods.',
      'Applied Chi-Square testing and logistic regression to extend the project beyond description into statistical and predictive insight.',
    ],
    stack: ['Python', 'Pandas', 'SQL', 'Matplotlib', 'Chi-Square', 'Logistic Regression'],
    primary: { label: 'GitHub Repository', href: 'https://github.com/DavisHiggins/HateCrimeDataAnalysis' },
    accent: 'from-violet-500/25 via-fuchsia-400/10 to-sky-500/20',
    logo: dhLogo,
  },
  {
    tag: 'Research Project',
    title: 'Census Data Analysis',
    subtitle: 'Demographic and socioeconomic analysis of salary outcomes using education, age, race, and sex as predictors.',
    description:
      'This project analyzed U.S. Census data to identify which demographic and socioeconomic factors most strongly influence salary outcomes, with education emerging as the strongest predictor.',
    bullets: [
      'Applied correlation analysis, ANOVA, and Chi-Square techniques to evaluate relationships among variables.',
      'Built visualizations that made income patterns and disparities easier to interpret.',
      'Connected findings to broader real-world discussion around inequality, opportunity, and socioeconomic outcomes.',
    ],
    stack: ['Python', 'Visualization', 'Correlation', 'ANOVA', 'Chi-Square'],
    primary: { label: 'Final Paper', href: '/docs/Census_Data_Analysis_Paper.pdf' },
    accent: 'from-emerald-500/25 via-teal-400/10 to-sky-500/20',
    logo: dhLogo,
  },
]

const EXPERIENCE = [
  {
    period: 'Jun 2025 – Present',
    title: 'Data Analyst Intern',
    company: 'Kewaunee Scientific Corp.',
    detail: 'Dashboard Development · KPI Reporting · Data Automation',
    bullets: [
      'Engineer and maintain 20+ executive dashboards and 100+ visualizations that improve reporting quality, readability, and decision-making for leadership.',
      'Audit 2,000+ project and quote records through CRM systems to strengthen data accuracy, integrity, and reporting reliability across enterprise datasets.',
      'Build automated KPI reporting workflows and a centralized data dictionary to improve data governance, cross-team usability, and long-term reporting consistency.',
      'Meet directly with executives and stakeholders to understand what reports, dashboards, and automations would make their jobs easier, then translate those business requests into practical solutions.',
      'Support cleaner communication between raw data, business questions, and final executive output by building reporting assets that are both technically useful and professionally polished.',
    ],
  },
  {
    period: 'Jan 2026 – Present',
    title: 'Vice President of Philanthropy',
    company: 'Phi Delta Theta',
    detail: 'Leadership · Commitment · NC Epsilon',
    bullets: [
      'Lead fundraising strategy and execution, generating $3,000+ for the Live Like Lou Foundation.',
      'Coordinate philanthropy initiatives that improve member engagement and community impact.',
    ],
  },
  {
    period: 'May 2024 – Aug 2024',
    title: 'Assistant to Project Manager',
    company: 'Higgins Building Group, Inc.',
    detail: 'Operational Oversight · Timeline Documentation',
    bullets: [
      'Coordinated project documentation and cross-functional workflows to support construction operations.',
      'Monitored timelines, materials, and logistics to improve project execution and efficiency.',
    ],
  },
  {
    period: 'Jan 2023 – May 2024',
    title: 'Co-Founder & Operations Lead',
    company: 'Mobile Custom Apparel Business',
    detail: 'Self-employed · Retail Sales · Business Ownership',
    bullets: [
      'Ran a mobile retail operation across regional AAU basketball tournaments, optimizing pricing, revenue splits, and product mix using live sales data.',
      'Managed end-to-end operations including inventory, transactions, on-demand production, and fulfillment for 100+ custom orders per event.',
      'Adjusted offerings in real time based on customer demand, event context, and buying behavior to maximize revenue and improve execution under pressure.',
      'Operated the business with an owner mindset by balancing speed, service quality, logistics, and profitability in live-event environments.',
    ],
  },
  {
    period: 'Dec 2022 – May 2024',
    title: 'Warehouse Specialist',
    company: 'Touch-Up Solutions',
    detail: 'Order Fulfillment · Production · Shipment Handling',
    bullets: [
      'Executed high-volume inventory handling and order fulfillment with accuracy and efficiency.',
      'Ensured on-time shipment processing, supporting productive and reliable warehouse operations.',
    ],
  },
]

const PROOF_OF_SKILLS = [
  {
    eyebrow: 'TECHNICAL EXECUTION',
    title: 'Problem-Solving and Commitment to Success',
    traits: ['Dedication', 'Initiative', 'Adaptability'],
    text:
      'During my time at Kewaunee, I regularly met with the sales team to understand what data, reports, and dashboards would best support their decision-making or resolve any present issues. In one of these discussions, the VP of Sales proposed a complex dashboard that would track projects set to expire across multiple timelines - within the next day, week, month, and year - along with many additional components. From a technical standpoint, I immediately recognized a major challenge: the structure he wanted conflicted with limitations in the analytics platform we were using. While it appeared unlikely that the dashboard could be built as requested, I chose not to push back. I took full ownership of the problem and committed to finding a solution. After extensive trial and error, deep research, and experimentation, I eventually identified a workaround that allowed the required data to be structured and displayed correctly. By leveraging available tools, collaborating with colleagues and supervisors, and using AI to accelerate problem-solving, I was able to successfully develop and deliver the dashboard. What began as a seemingly infeasible request became a fully functional executive tool, reinforcing my approach to problem-solving: persistence, adaptability, and a refusal to accept defeat.',
  },
  {
    eyebrow: 'PRESSURE PERFORMANCE',
    title: 'Perseverance Through Adversity',
    traits: ['Self-Reliance', 'Work Ethic', 'Accountability'],
    text:
      'It was May 2023, and I was heading to Rock Hill, SC to promote my apparel business and sell t-shirts. My friend who usually ran the table with me was out of town, so I brought in another friend with sales experience to help. The tournament ran Friday through Sunday from 7 a.m. to 8 p.m. each day. While Friday went well, on Saturday morning, my partner woke up with a high fever and had to sit out the rest of the weekend. I considered going back, packing everything up, and leaving, but I decided to keep the table running. I knew it would be less efficient with very few breaks, since I would be handling everything myself: tracking orders, managing inventory, attracting customers, printing designs, processing transactions, and packing all equipment into my Honda Accord. I stayed and ran the table the rest of the weekend to the best of my ability. After paying the tournament owner and deducting costs, I generated just over $1,500 across Saturday and Sunday. I could have taken the easy route and left, but I chose to stay and see how well I could perform under pressure.',
  },
  {
    eyebrow: 'SYSTEM ARCHITECTURE',
    title: 'Engineering a Decision Engine',
    traits: ['Innovation', 'Product Thinking', 'Technical Execution', 'System Design'],
    text:
      'Projects like Propify reflect the way I like to work: build with structure, think through user experience, protect what is proprietary, and create systems that can realistically scale into something bigger than a class assignment or demo. The idea for Propify came from my interest in AI and sports. As I was learning more about AI and what it is capable of, I started thinking about its potential for product creation. As a big sports fan with friends and family members who place wagers on games, I thought to myself, there has to be a better way to sports bet. Instead of relying on emotion or biased decisions with poor odds, I believed data could be used to estimate outcomes more accurately. I researched existing tools and found that while many platforms provide statistics and insights, none offer a clear, engine-generated probability for whether an event will occur. That gap led me to build my own solution. I created a Streamlit app using the nba_api to pull player data and developed an internal engine to calculate probabilities based on key performance factors and distributions. The goal was to turn raw data into a structured, usable percentage that supports more logical decision-making. From a curious thought, came an analytics platform and decision engine from scratch. It is incredible how far your mind can take you in a short amount of time.',
  },
]

const INTERESTS = [
  {
    icon: Dumbbell,
    title: 'Weightlifting',
    text:
      'Discipline, consistency, and long-term self-improvement are a major part of how I approach both life and work. Optimizing physical health is a blessing and not only improves appearance and confidence, but supports mental health and longevity as well.',
  },
  {
    icon: Trophy,
    title: 'Sports',
    text:
      'I was a part of the football, basketball, and soccer team in high school, earning a total of 1x All-State, 3x All-Conference, and 1x Honorable Mention across all of those sports. I still play each of these sports weekly with friends, or in UNC Charlotte fraternity intramurals.',
  },
  {
    icon: Code2,
    title: 'Building Products',
    text:
      'I enjoy creating tools, interfaces, and systems that make data easier to use and more valuable to people. I understand the growing importance of data science and AI in the corporate world, and I truly love implementing the concepts I have learned to discover what all I am capable of creating.',
  },
  {
    icon: HandHelping,
    title: 'Volunteer Work',
    text:
      'My volunteer interests include philanthropy work with the Live Like Lou Foundation and service through Habitat for Humanity. I have always dedicated time to help those who are not as fortunate, and spending time helping and supporting those in-need is one of my favorite things to do.',
  },
]

const CONTACT_LINKS = [
  {
    label: 'Email',
    value: 'davishiggins@icloud.com',
    href: 'mailto:davishiggins@icloud.com',
    icon: Mail,
    glow: 'from-sky-500/25 to-cyan-400/10',
  },
  {
    label: 'LinkedIn',
    value: 'linkedin.com/in/davishiggins',
    href: 'https://www.linkedin.com/in/davishiggins',
    icon: Linkedin,
    glow: 'from-blue-500/25 to-sky-400/10',
    external: true,
  },
  {
    label: 'Phone',
    value: '+1 (828) 578-0262',
    href: 'tel:+18285780262',
    icon: Phone,
    glow: 'from-violet-500/25 to-fuchsia-400/10',
  },
]

function SectionHeader({ eyebrow, title, subtitle, showSubtitle = true }) {
  return (
    <div className="mb-10 max-w-4xl">
      <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium uppercase tracking-[0.22em] text-sky-300">
        <Sparkles className="h-3.5 w-3.5" />
        {eyebrow}
      </div>
      <h2 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">{title}</h2>
      {subtitle && showSubtitle ? <p className="mt-4 text-base leading-7 text-slate-300 sm:text-lg">{subtitle}</p> : null}
    </div>
  )
}

function Pill({ children }) {
  return <span className="chip">{children}</span>
}

function InfoCard({ icon: Icon, title, text }) {
  return (
    <div className="glass rounded-[1.75rem] p-6 shadow-soft">
      <div className="inline-flex rounded-2xl bg-white/5 p-3 text-sky-300">
        <Icon className="h-5 w-5" />
      </div>
      <h3 className="mt-5 text-xl font-semibold text-white">{title}</h3>
      <p className="mt-3 text-sm leading-7 text-slate-400">{text}</p>
    </div>
  )
}

function PageShell({ children }) {
  return <div className="section-shell px-6 py-14 sm:px-8 lg:px-12">{children}</div>
}

function SchoolLogo({ src, alt, sizeClass = 'h-20 w-20', imgClass = 'h-20 w-20' }) {
  return (
    <div className={`flex ${sizeClass} shrink-0 items-center justify-center`}>
      <img src={src} alt={alt} className={`${imgClass} object-contain`} />
    </div>
  )
}

export default function App() {
  const [activePage, setActivePage] = useState('home')

  const activeTitle = useMemo(() => {
    const item = NAV_ITEMS.find((nav) => nav.key === activePage)
    return item?.label ?? 'Home'
  }, [activePage])

  const pageAnimation = {
    initial: { opacity: 0, y: 14 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -12 },
    transition: { duration: 0.28 },
  }

  const renderPage = () => {
    switch (activePage) {
      case 'home':
        return (
          <motion.div key="home" {...pageAnimation}>
            <PageShell>
              <div className="relative overflow-hidden rounded-[2.5rem] border border-white/10 bg-white/[0.03] px-6 py-8 sm:px-8 lg:px-10">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(125,211,252,0.12),transparent_32%),radial-gradient(circle_at_top_right,rgba(99,102,241,0.12),transparent_28%),linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.01))]" />
                <div className="absolute -left-12 top-10 h-44 w-44 rounded-full bg-sky-400/10 blur-3xl" />
                <div className="absolute right-0 top-0 h-52 w-52 rounded-full bg-indigo-500/10 blur-3xl" />
                <div className="absolute bottom-0 left-1/3 h-40 w-40 rounded-full bg-cyan-400/10 blur-3xl" />

                <div className="relative grid items-center gap-10 lg:grid-cols-[1.1fr_0.9fr] lg:gap-16">
                  <div>
                    <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-sky-400/20 bg-sky-400/10 px-4 py-2 text-sm font-medium text-sky-200">
                      <LineChart className="h-4 w-4" />
                      Data Science • Artificial Intelligence • Charlotte, NC
                    </div>

                    <h1 className="max-w-4xl text-5xl font-semibold tracking-tight text-white sm:text-6xl lg:text-7xl">
                      Hi, I'm <span className="bg-gradient-to-r from-sky-300 to-cyan-200 bg-clip-text text-transparent">Davis Higgins</span>
                    </h1>

                    <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300 sm:text-xl">
                      Data Science student at UNC Charlotte building dashboards, reporting systems, and product-oriented tools that make information clearer, faster, and more useful.
                    </p>

                    <div className="mt-6 flex flex-wrap gap-3">
                      <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-[11px] font-medium uppercase tracking-[0.2em] text-slate-300">
                        REAL-WORLD DATA EXPERIENCE
                      </div>
                      <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-[11px] font-medium uppercase tracking-[0.2em] text-slate-300">
                        END-TO-END SYSTEM CREATION
                      </div>
                      <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-[11px] font-medium uppercase tracking-[0.2em] text-slate-300">
                        BUSINESS-DRIVEN INSIGHTS
                      </div>
                      <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-[11px] font-medium uppercase tracking-[0.2em] text-slate-300">
                        DECISION-FOCUSED MODELING
                      </div>
                    </div>

                    <p className="mt-5 max-w-2xl text-base leading-8 text-slate-400">
                      Through coursework, personal projects, and internship experience, I have developed skills in data analytics, data cleaning and preprocessing, statistical modeling, machine learning, data visualization, and data governance.
                    </p>

                    <div className="mt-8 flex flex-wrap gap-4">
                      <button type="button" onClick={() => setActivePage('projects')} className="primary-btn">
                        See Projects
                        <ArrowRight className="h-4 w-4" />
                      </button>
                      <button type="button" onClick={() => setActivePage('experience')} className="secondary-btn">
                        Professional Experience
                      </button>
                    </div>

                    <div className="mt-10 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                      {STATS.map((item) => (
                        <div key={item.label} className="glass rounded-3xl p-5 shadow-soft">
                          <div className="text-2xl font-bold text-white">{item.value}</div>
                          <div className="mt-1 text-sm text-slate-400">{item.label}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="mx-auto w-full max-w-md">
                    <div className="rounded-[2rem] border border-white/10 bg-[#5f7ea4]/95 p-5 shadow-glow">
                      <div className="overflow-hidden rounded-[1.6rem] border border-white/10 bg-[#a8b9d5]">
                        <div className="aspect-[4/5] bg-[#a8b9d5]">
                          <img src={headshot} alt="Professional headshot of Davis Higgins" className="h-full w-full object-cover" />
                        </div>
                      </div>

                      <div className="mt-5 rounded-[1.5rem] border border-white/10 bg-[#12203b] p-5">
                        <div className="flex items-start gap-4">
                          <div className="flex h-14 w-28 items-center justify-center overflow-hidden rounded-2xl bg-white">
                            <img src={dhLogo} alt="DH logo" className="h-12 w-12 object-contain" />
                          </div>
                          <div>
                            <div className="text-sm uppercase tracking-[0.35em] text-sky-300">Focus</div>
                            <div className="mt-2 text-base leading-7 text-slate-300">
                              Motivated student looking for opportunities to apply my skills, develop professionally, grow personally, and learn as much as possible.
                            </div>
                          </div>
                        </div>

                        <div className="mt-5 rounded-[1.5rem] border border-white/10 bg-white/[0.05] p-5">
                          <div className="flex items-center gap-4">
                            <div className="rounded-2xl bg-sky-400/10 p-4 text-sky-300">
                              <Briefcase className="h-6 w-6" />
                            </div>
                            <div>
                              <div className="text-sm text-slate-400">Current Role</div>
                              <div className="mt-1 text-2xl font-semibold text-white">Data Analyst Intern at Kewaunee Scientific Corporation</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </PageShell>
          </motion.div>
        )

      case 'experience':
        return (
          <motion.div key="experience" {...pageAnimation}>
            <PageShell>
              <SectionHeader
                eyebrow="EXECUTION → BUSINESS IMPACT"
                title="Experience"
                subtitle="Hands-on experience across analytics, operations, and business-facing execution."
              />

              <div className="relative space-y-6 before:absolute before:left-[1.1rem] before:top-2 before:h-[calc(100%-1rem)] before:w-px before:bg-white/10 sm:before:left-6">
                {EXPERIENCE.map((item, index) => (
                  <motion.div
                    key={`${item.title}-${item.company}`}
                    initial={{ opacity: 0, x: 12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.04 }}
                    className="relative pl-12 sm:pl-16"
                  >
                    <div className="absolute left-0 top-5 flex h-9 w-9 items-center justify-center rounded-full border border-sky-400/30 bg-sky-400/10 text-sky-300 sm:left-1.5">
                      <Briefcase className="h-4 w-4" />
                    </div>

                    <div className="glass rounded-[2rem] p-6 shadow-soft">
                      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                        <div>
                          <div className="text-sm font-medium uppercase tracking-[0.18em] text-sky-300">{item.period}</div>
                          <h3 className="mt-3 text-2xl font-semibold text-white">{item.title}</h3>
                          <div className="mt-1 text-base text-slate-300">{item.company}</div>
                          <div className="mt-1 text-sm text-slate-500">{item.detail}</div>
                        </div>
                      </div>

                      <ul className="mt-6 space-y-3 text-sm leading-7 text-slate-300">
                        {item.bullets.map((bullet) => (
                          <li key={bullet} className="flex gap-3">
                            <span className="mt-2 h-1.5 w-1.5 rounded-full bg-cyan-300" />
                            <span>{bullet}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </motion.div>
                ))}
              </div>
            </PageShell>
          </motion.div>
        )

      case 'projects':
        return (
          <motion.div key="projects" {...pageAnimation}>
            <PageShell>
              <SectionHeader
                eyebrow="PRODUCTS → ANALYTICAL SOLUTIONS"
                title="Projects"
                subtitle="Selected work demonstrating how I build, analyze, and translate data into practical tools and insights."
              />

              <div className="space-y-6">
                {PROJECTS.map((project, index) => (
                  <motion.div
                    key={project.title}
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05, duration: 0.32 }}
                    className="group relative overflow-hidden rounded-[2rem] border border-white/10 bg-white/[0.04] p-6 shadow-soft"
                  >
                    <div className={`absolute inset-0 bg-gradient-to-br ${project.accent}`} />
                    <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(8,17,31,0.18),rgba(8,17,31,0.9))]" />
                    <div className="relative">
                      <div className="mb-4 flex items-center gap-4">
                        <img
                          src={project.logo}
                          alt={`${project.title} logo`}
                          className="h-16 w-16 shrink-0 object-contain bg-transparent sm:h-20 sm:w-20"
                        />
                        <div className="rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-sky-200">
                          {project.tag}
                        </div>
                      </div>

                      <h3 className="text-2xl font-semibold text-white">{project.title}</h3>
                      <p className="mt-2 text-sm text-slate-300">{project.subtitle}</p>
                      <p className="mt-5 text-sm leading-7 text-slate-200/90">{project.description}</p>

                      <ul className="mt-6 space-y-3 text-sm leading-7 text-slate-300">
                        {project.bullets.map((bullet) => (
                          <li key={bullet} className="flex gap-3">
                            <span className="mt-2 h-1.5 w-1.5 rounded-full bg-sky-300" />
                            <span>{bullet}</span>
                          </li>
                        ))}
                      </ul>

                      <div className="mt-6 flex flex-wrap gap-2">
                        {project.stack.map((item) => (
                          <Pill key={item}>{item}</Pill>
                        ))}
                      </div>

                      <div className="mt-8 flex flex-wrap gap-3">
                        <a href={project.primary.href} target="_blank" rel="noreferrer" className="primary-btn">
                          {project.primary.label}
                          <ExternalLink className="h-4 w-4" />
                        </a>
                        {project.secondary ? (
                          <a href={project.secondary.href} target="_blank" rel="noreferrer" className="secondary-btn">
                            {project.secondary.label}
                          </a>
                        ) : null}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </PageShell>
          </motion.div>
        )

      case 'skills-proof':
        return (
          <motion.div key="skills-proof" {...pageAnimation}>
            <PageShell>
              <SectionHeader
                eyebrow="PROBLEMS → MEASURABLE RESULTS"
                title="Impact"
                subtitle="Real-world examples of solving complex problems, overcoming constraints, and delivering results."
              />

              <div className="space-y-6">
                {PROOF_OF_SKILLS.map((item, index) => (
                  <motion.div
                    key={item.title}
                    initial={{ opacity: 0, y: 14 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.28, delay: index * 0.05 }}
                    className="glass rounded-[2rem] p-6 shadow-soft"
                  >
                    <div className="mb-4 inline-flex rounded-full bg-white/5 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-sky-200">
                      {item.eyebrow}
                    </div>

                    <h3 className="text-xl font-semibold text-white">{item.title}</h3>

                    <div className="mt-4 flex flex-wrap gap-2">
                      {item.traits.map((trait) => (
                        <Pill key={trait}>{trait}</Pill>
                      ))}
                    </div>

                    <p className="mt-4 text-sm leading-7 text-slate-300">{item.text}</p>
                  </motion.div>
                ))}
              </div>
            </PageShell>
          </motion.div>
        )

      case 'education':
        return (
          <motion.div key="education" {...pageAnimation}>
            <PageShell>
              <SectionHeader
                eyebrow="FOUNDATION → TECHNICAL DEVELOPMENT"
                title="Education"
                subtitle="Classroom efficiency, technical development, and the foundation behind my skill set."
              />

              <div className="space-y-10"/>
                <div>
                  <h3 className="text-2xl font-semibold text-white">Academic Background</h3>

                  <div className="mt-6 space-y-6">
  <div className="glass rounded-[2rem] px-3 py-7 shadow-soft">
    <div className="flex items-start gap-2.5">
      <div className="flex h-[84px] w-[86px] shrink-0 items-start justify-center">
        <img
  src={unccLogo}
  alt="UNC Charlotte logo"
  className="mt-0 h-[55px] w-[85px] object-contain"
  style={{
    filter:
      'brightness(0) saturate(100%) invert(74%) sepia(22%) saturate(1185%) hue-rotate(180deg) brightness(101%) contrast(101%)',
  }}
/>
      </div>

      <div className="min-w-0 flex-1">
        <h4 className="text-2xl font-semibold text-white">University of North Carolina at Charlotte</h4>

        <p className="mt-3 text-sm text-slate-400">
          B.S. in Data Science • Minor in Artificial Intelligence • Aug 2024 – May 2027
        </p>

        <div className="mt-6 flex flex-wrap gap-2">
          <Pill>GPA: 3.84</Pill>
          <Pill>2x Chancellor’s List</Pill>
          <Pill>1x Dean’s List</Pill>
        </div>

        <p className="mt-6 text-sm leading-7 text-slate-300">
          Building a technical foundation in analytics, machine learning, statistical thinking, and applied business problem solving while pairing coursework with internships, product development, and independent projects.
        </p>
      </div>
    </div>
  </div>

  <div className="glass rounded-[2rem] px-4 py-7 shadow-soft">
  <div className="flex items-start gap-2">
    <div className="flex h-[70px] w-[84px] shrink-0 items-start justify-center">
      <img
  src={cvccLogo}
  alt="Catawba Valley Community College logo"
  className="mt-0 h-[56px] w-[55px] object-contain"
  style={{
    filter:
      'hue-rotate(192deg) saturate(135%) brightness(1.18) contrast(1.08) drop-shadow(0 0 2px rgba(125, 211, 252, 0.12))',
  }}
/>
    </div>

      <div className="min-w-0 flex-1">
        <h4 className="text-xl font-semibold text-white">Catawba Valley Community College</h4>

        <p className="mt-3 text-sm text-slate-400">Aug 2022 – May 2024</p>

        <div className="mt-5 flex flex-wrap gap-2">
          <Pill>GPA: 4.0</Pill>
          <Pill>4x President’s List</Pill>
        </div>

        <p className="mt-6 text-sm leading-7 text-slate-300">
          Selected for a dual-enrollment program to gain college credits while completing high school.
        </p>
      </div>
    </div>
  </div>
</div>            

                <div>
                  <h3 className="mt-16 text-2xl font-semibold text-white">Honors</h3>

                  <div className="mt-6 space-y-5">
                    <div className="glass rounded-[2rem] p-7 shadow-soft">
                      <div className="flex items-start gap-7">
                        <div className="rounded-2xl bg-sky-400/10 p-3 text-sky-300">
                          <Medal className="h-8 w-7" />
                        </div>
                        <div className="w-full">
                          <h4 className="text-xl font-semibold text-white">Public Anthropology Award for Excellence in Writing on Public Issues</h4>
                          <p className="mt-2 text-sm text-slate-400">Center for a Public Anthropology • Apr 2025</p>

                          <div className="mt-5 flex flex-wrap gap-2">
                            <Pill>Published Work</Pill>
                            <Pill>Storytelling</Pill>
                          </div>

                          <p className="mt-6 text-sm leading-7 text-slate-300">
                            Recognized for excellence in writing on public issues. The publication was selected for its ability to address a public issue through thoughtful, persuasive, and captivating writing.
                          </p>

                          <div className="mt-6 flex flex-wrap gap-2">
                            <a
                              href={anthropologyAward}
                              target="_blank"
                              rel="noreferrer"
                              className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-5 py-3 text-sm font-semibold text-white transition hover:bg-white/10"
                            >
                              View Certification
                              <ExternalLink className="h-4 w-4" />
                            </a>

                            <a
                              href={thecreek}
                              target="_blank"
                              rel="noreferrer"
                              className="secondary-btn"
                            >
                              Publication
                              <ExternalLink className="h-4 w-4" />
                            </a>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </PageShell>
          </motion.div>
        )

      case 'resume':
        return (
          <motion.div key="resume" {...pageAnimation}>
            <PageShell>
              <SectionHeader
                eyebrow="SKILLS → PROFESSIONAL READINESS"
                title="Resume"
                subtitle="A clear, structured overview of my experience, skills, and technical capabilities."
              />

              <div className="grid items-start gap-6 lg:grid-cols-[1.15fr_0.85fr]">
                <div className="glass rounded-[2rem] p-9 shadow-soft self-start">
                  <div className="flex items-start gap-4">
                    <div className="rounded-2xl bg-white/5 p-3 text-sky-300">
                      <NotebookText className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-semibold text-white">Quick access</h3>
                      <p className="mt-6 max-w-2xl text-sm leading-7 text-slate-300">
                        Click below to view my resume or visit my projects page.
                      </p>

                      <div className="mt-16 flex flex-wrap gap-3">
                        <a href="/docs/DBH-Resume.pdf" target="_blank" rel="noreferrer" className="primary-btn">
                          <FileText className="h-4 w-4" />
                          Open Resume
                        </a>
                        <button type="button" onClick={() => setActivePage('projects')} className="secondary-btn">
                          Projects
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="grid gap-4">
                  <div className="glass rounded-[1.75rem] p-5 shadow-soft">
                    <div className="text-sm text-slate-400">Tools & Platforms</div>
                    <div className="mt-2 text-lg font-semibold text-white">Python, SQL, R, Java, Power BI, Zoho, AI</div>
                  </div>
                  <div className="glass rounded-[1.75rem] p-5 shadow-soft">
                    <div className="text-sm text-slate-400">Technical Strengths</div>
                    <div className="mt-2 text-lg font-semibold text-white">
                    Exploratory Data Analysis, Dashboard Development, KPI Reporting, Reporting Automation, Data Cleaning, Data Validation
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 glass rounded-[2rem] p-7 shadow-soft">
                <h3 className="text-2xl font-semibold text-white">Resume Highlights</h3>

                <ul className="mt-5 space-y-3 text-sm leading-7 text-slate-300 sm:text-base">
                  <li className="flex gap-3">
                    <span className="mt-2 h-1.5 w-1.5 rounded-full bg-cyan-300" />
                    <span>Data Science student at UNC Charlotte pursuing a minor in Artificial Intelligence.</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="mt-2 h-1.5 w-1.5 rounded-full bg-cyan-300" />
                    <span>Hands-on experience in dashboard development, reporting automation, data validation, and business-facing analytics.</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="mt-2 h-1.5 w-1.5 rounded-full bg-cyan-300" />
                    <span>Built and maintained 20+ executive dashboards, created 100+ visualizations, and audited 2,000+ CRM records.</span>
                  </li>
                  <li className="flex gap-3">
                    <span className="mt-2 h-1.5 w-1.5 rounded-full bg-cyan-300" />
                    <span>Strong working foundation in Python, SQL, R, Power BI, Zoho Analytics, data visualization, and product-oriented problem solving.</span>
                  </li>
                </ul>

                <div className="mt-6 flex flex-wrap gap-2">
                  <Pill>Dashboard Development</Pill>
                  <Pill>Reporting Automation</Pill>
                  <Pill>Product Thinking</Pill>
                  <Pill>Data Quality</Pill>
                </div>
              </div>
            </PageShell>
          </motion.div>
        )

      case 'interests':
        return (
          <motion.div key="interests" {...pageAnimation}>
            <PageShell>
              <SectionHeader
                eyebrow="DISCIPLINE → PERSONAL GROWTH"
                title="Interests"
                subtitle="Activities and habits that shape my discipline, mindset, and growth outside of work."
              />

              <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
                {INTERESTS.map((item) => (
                  <InfoCard key={item.title} icon={item.icon} title={item.title} text={item.text} />
                ))}
              </div>
            </PageShell>
          </motion.div>
        )

      case 'contact':
        return (
          <motion.div key="contact" {...pageAnimation}>
            <PageShell>
              <SectionHeader
                eyebrow="CONNECTION → OPPORTUNITY"
                title="Contact"
                subtitle="Direct ways to reach me for opportunities, collaboration, or professional conversations."
              />

              <div className="mx-auto max-w-5xl">
                <motion.div
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35 }}
                  className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-white/[0.04] p-8 shadow-[0_20px_80px_rgba(0,0,0,0.28)]"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-sky-500/12 via-transparent to-indigo-500/10" />
                  <div className="absolute -right-20 top-0 h-56 w-56 rounded-full bg-sky-400/10 blur-3xl" />
                  <div className="absolute -left-16 bottom-0 h-52 w-52 rounded-full bg-violet-500/10 blur-3xl" />

                  <div className="relative">
                    <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] text-sky-300">
                      <Sparkles className="h-3.5 w-3.5" />
                      Let's Connect
                    </div>

                    <h3 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">Direct Links</h3>

                    <p className="mt-4 max-w-3xl text-base leading-8 text-slate-300">
                      I am actively looking for opportunities to apply my skills, continue developing professionally, and keep building meaningful analytics and product-oriented work.
                    </p>

                    <div className="mt-8 grid gap-4 md:grid-cols-3">
                      {CONTACT_LINKS.map((item, index) => {
                        const Icon = item.icon
                        return (
                          <motion.a
                            key={item.label}
                            href={item.href}
                            target={item.external ? '_blank' : undefined}
                            rel={item.external ? 'noreferrer' : undefined}
                            initial={{ opacity: 0, y: 12 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3, delay: index * 0.06 }}
                            whileHover={{ y: -4, scale: 1.015 }}
                            className="group relative overflow-hidden rounded-[1.6rem] border border-white/10 bg-[#0f172a]/70 p-5 transition duration-300 hover:border-sky-300/25 hover:bg-[#13203a]/85 hover:shadow-[0_15px_40px_rgba(56,189,248,0.12)]"
                          >
                            <div className={`absolute inset-0 bg-gradient-to-br ${item.glow} opacity-0 transition duration-300 group-hover:opacity-100`} />
                            <div className="relative">
                              <div className="flex items-center justify-between">
                                <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-sky-300 transition duration-300 group-hover:scale-110 group-hover:bg-white/10">
                                  <Icon className="h-5 w-5" />
                                </div>
                                <ExternalLink className="h-4 w-4 text-slate-500 transition duration-300 group-hover:text-sky-300" />
                              </div>

                              <div className="mt-5 text-sm uppercase tracking-[0.22em] text-slate-400">{item.label}</div>
                              <div className="mt-2 break-words text-lg font-semibold text-white">{item.value}</div>
                            </div>
                          </motion.a>
                        )
                      })}
                    </div>
                  </div>
                </motion.div>
              </div>
            </PageShell>
          </motion.div>
        )

      default:
        return null
    }
  }

  return (
    <div className="min-h-screen overflow-x-hidden bg-ink text-slate-100">
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute inset-0 bg-grid bg-[size:52px_52px] opacity-[0.04]" />
        <div className="absolute left-1/2 top-0 h-[34rem] w-[34rem] -translate-x-1/2 rounded-full bg-sky-500/20 blur-3xl" />
        <div className="absolute right-0 top-40 h-[28rem] w-[28rem] rounded-full bg-indigo-500/15 blur-3xl" />
        <div className="absolute bottom-0 left-0 h-[24rem] w-[24rem] rounded-full bg-cyan-400/10 blur-3xl" />
      </div>

      <header className="sticky top-0 z-50 border-b border-white/10 bg-[#08111f]/85 backdrop-blur-xl">
        <div className="section-shell flex items-center justify-between gap-4 py-4">
          <div className="flex items-center gap-3">
            <div className="hidden items-center gap-3.5 sm:flex">
              <div className="flex h-14 w-12 items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-white">
                <img src={dhLogo} alt="DH logo" className="h-10 w-10 object-contain" />
              </div>

              <div className="leading-none">
                <div className="text-sm font-semibold uppercase tracking-[0.24em] text-sky-300">DAVIS HIGGINS</div>

                <div className="whitespace-nowrap text-sm text-slate-400">
                  Data Analyst | Data Science &amp; AI |
                </div>

                <div className="whitespace-nowrap text-sm text-slate-400">
                  Statistical Modeling | Predictive Analytics
                </div>
              </div>
            </div>

            <div className="flex h-14 w-14 items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-white sm:hidden">
              <img src={dhLogo} alt="DH logo" className="h-10 w-10 object-contain" />
            </div>
          </div>

          <nav className="hidden items-center justify-center gap-2 rounded-full border border-white/10 bg-white/5 px-6 py-3 shadow-glow lg:flex">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon
              const active = activePage === item.key

              return (
                <button
                  key={item.key}
                  type="button"
                  onClick={() => setActivePage(item.key)}
                  className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-4 py-2 text-sm font-medium transition ${
                    active
                      ? 'bg-white text-slate-950'
                      : 'text-slate-300 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  {item.label}
                </button>
              )
            })}
          </nav>

          <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-300 lg:hidden">
            {activeTitle}
          </div>
        </div>

        <div className="section-shell pb-4 lg:hidden">
          <div className="flex snap-x gap-2 overflow-x-auto rounded-full border border-white/10 bg-white/5 p-2">
            {NAV_ITEMS.map((item) => {
              const active = activePage === item.key
              return (
                <button
                  key={item.key}
                  type="button"
                  onClick={() => setActivePage(item.key)}
                  className={`whitespace-nowrap rounded-full px-3.5 py-2 text-sm font-medium transition ${
                    active ? 'bg-white text-slate-950' : 'text-slate-300'
                  }`}
                >
                  {item.label}
                </button>
              )
            })}
          </div>
        </div>
      </header>

      <main>
        <AnimatePresence mode="wait">{renderPage()}</AnimatePresence>
      </main>
    </div>
  )
}
