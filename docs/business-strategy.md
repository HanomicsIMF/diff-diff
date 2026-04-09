# Strategic Analysis: diff-diff for Business Data Science

*April 2026*

## Context

diff-diff is the most comprehensive Difference-in-Differences library in Python — 16 estimators, unique survey design support, HonestDiD sensitivity analysis, and a practitioner workflow. But its entire framing speaks to academic econometricians. There's a large, underserved market of business data scientists who need DiD for real-world problems (campaign measurement, product launches, pricing changes) but are currently using fragmented tools or manual approaches. This analysis assesses the opportunity, competitive positioning, and what we need to do.

---

## 1. The Market Opportunity

### The Direction
- The causal inference market is growing rapidly — analyst estimates vary widely by scope, but all directionally agree on strong double-digit growth
- Enterprise adoption is accelerating: Microsoft (DoWhy, EconML), Meta (GeoLift, Robyn), Google (Meridian, CausalImpact), Uber (CausalML) have all invested heavily in open-source causal inference tooling in the past 2 years
- Privacy changes (cookie deprecation, tracking restrictions) are forcing marketing teams toward causal measurement methods — away from tracking-based attribution

### The Shift Happening Now
Marketing measurement is undergoing a structural shift. The old model (track users, attribute conversions, optimize) is breaking due to privacy regulation and platform restrictions. The new model requires **causal inference**: geo-experiments, DiD, synthetic control, and MMM. This shift is why Google built Meridian (Jan 2025), Meta built GeoLift and Robyn, and Uber invested in CausalML.

Companies actively using causal inference in production: Uber, DoorDash, Airbnb, Netflix, Meta, Spotify, Booking.com, Mercado Libre (20+ geo-experiments), among many others.

### Why This Matters for diff-diff
The demand for DiD in business is real and growing. But the supply side in Python is fragmented and academic. No one owns "DiD for business data scientists" in Python. This is our lane.

---

## 2. Our Current Position

### What We've Built (Strengths)

| Capability | Competitive Position |
|---|---|
| 16 estimators (CS, SA, BJS, ETWFE, SDiD, TROP, etc.) | **Unmatched** — nearest competitor has 3-4 |
| Survey design support (strata, PSU, FPC, replicate weights) | **Unique in Python** — no competitor offers this |
| HonestDiD sensitivity analysis | **Unique in Python** — critical for credibility |
| Baker et al. (2025) practitioner workflow | **Unique** — no other library embeds methodological guardrails |
| Power analysis & pre-trends power | **Unique** — essential for study design |
| Bacon decomposition, parallel trends tests, placebo tests | **Most complete** diagnostic suite |
| Rust backend (5-50x speedup) | **Unique** performance advantage |
| 16 tutorials, real datasets, rich visualization | Strong, but academic framing |

### Who We Serve Today
Applied econometricians and academic researchers who:
- Know what "ATT(g,t)" means
- Read Callaway & Sant'Anna (2021) and Rambachan & Roth (2023)
- Work in R-like workflows with Python
- Need publication-ready statistical output

### What's Missing for Business
Our technical foundation is strong. The gap is not in methodology — it's in **packaging, language, workflows, and examples**. A marketing data scientist looking at our README sees Card-Krueger minimum wage studies and "forbidden comparisons." They need to see campaign lift measurement and "is this result trustworthy?"

---

## 3. Target Personas

### Persona A: Brand & Market Research
**Role**: Marketing analytics lead at CPG, retail, or agency
**Problem**: "We ran an awareness campaign in 5 markets. Did it actually move consideration?"
**Current tools**: Qualtrics/Dynata for survey data, Excel/manual for analysis, no formal causal framework
**What they need**: Survey data → DiD → stakeholder report. Plain-English validity assessment. Design effect handling.
**Our advantage**: Survey design support is unique and directly relevant. No competitor can do design-based variance with modern DiD estimators.

### Persona B: Growth & Performance Marketing
**Role**: Marketing data scientist at tech company or e-commerce
**Problem**: "We launched a campaign in some geos. What was the incremental lift?"
**Current tools**: GeoLift (synthetic control only), CausalImpact (time-series only), manual DiD in pandas
**What they need**: Geo-experiment → DiD with staggered rollout → confidence intervals → ROI calculation
**Our advantage**: Staggered estimators handle the reality that campaigns roll out in waves, not all at once. GeoLift can't do this.

### Persona C: Product & Operations DS
**Role**: Data scientist at tech/SaaS company
**Problem**: "We rolled out a new feature/pricing/process in some regions. What was the impact?"
**Current tools**: A/B testing platforms (Optimizely, Statsig), manual DiD when randomization isn't feasible
**What they need**: Quick setup → estimation → diagnostics → presentation to PM/VP
**Our advantage**: Comprehensive estimator suite handles any design pattern. Sensitivity analysis answers "how robust is this?"

### Common Needs Across All Personas
1. **Business language**: "lift", "incremental impact", "confidence level" — not "ATT", "parallel trends assumption"
2. **Speed to insight**: Minutes from data to answer, not hours learning methodology
3. **Stakeholder communication**: Output a VP can read, not a statistics table
4. **Validity without PhD**: "Is this analysis trustworthy?" answered in plain English
5. **Real business examples**: Campaigns, launches, pricing — not minimum wage studies

---

## 4. Competitive Landscape

### Direct Competitors (Python DiD)

| Package | Estimators | Business-Ready? | Weakness vs Us |
|---|---|---|---|
| **pyfixest** | TWFE, SA, did2s | No — academic framing | No CS, no HonestDiD, no survey. Has wildboottest integration for bootstrap inference. |
| **differences** | CS | No — maintenance issues | Removed TWFE + plotting in v0.2.0, limited scope |
| **CausalPy** | Basic DiD, SC | Partially — Bayesian framing | No staggered, no sensitivity, no survey |
| **linearmodels** | PanelOLS (manual) | No — building block | Requires manual DiD implementation |
| **statsmodels OLS** | Manual 2x2 | "Good enough" for many | Many business DS do DiD manually with OLS + interaction terms. No diagnostics, no staggered, no sensitivity — but low friction. |

**The bilingual R/Python angle**: Many data science teams have R capability. A business DS who needs serious DiD might reach for R's `did` package rather than learn a Python library. Our pitch must be stronger than "it's in Python" — it needs to be "it's better than chaining 3 R packages together, and it has survey support no R package matches."

**Assessment**: No Python DiD library targets business users. All are academic-oriented. We're the most complete, but we're also academic-oriented. The real competitor for many business DS is not another library — it's manual OLS in statsmodels or switching to R.

### Adjacent Platforms (Causal ML)

| Platform | Focus | DiD Support | Business Positioning |
|---|---|---|---|
| **DoWhy** (Microsoft) | DAG-based causal inference | Minimal — no modern DiD | Strong — "democratizing causal inference" |
| **CausalML** (Uber) | Uplift/CATE | None | Strong — "personalization + targeting" |
| **EconML** (Microsoft) | HTE estimation | None | Strong — "causal ML for decisions" |

**Assessment**: These platforms are well-positioned for business but don't do DiD. They're not competitors for our core use case — they're adjacent. We could potentially integrate rather than compete.

### Narrow Tools (Marketing-Specific)

| Tool | Method | Scope |
|---|---|---|
| **GeoLift** (Meta) | Synthetic control | Geo-experiments only, no staggered, no panel |
| **CausalImpact** (Google) | Bayesian structural time-series | Single intervention, time-series only |
| **Robyn** (Meta) | MMM (ridge regression) | Marketing mix, not DiD |
| **Meridian** (Google) | Bayesian MMM | Marketing mix, not DiD |

**Assessment**: These are point solutions for specific marketing problems. We're broader and more rigorous, but they own the "marketing causal inference" mindshare. We need to explicitly show how diff-diff handles the same problems — and more.

### Competitive Positioning Map (DiD-Specific)

```
                    Academic ←————————————→ Business
                         |                      |
    Narrow (1-2 methods) |  pyfixest            |  GeoLift (synth control only)
                         |  differences          |  CausalImpact (time-series only)
                         |  linearmodels         |  statsmodels OLS (manual)
                         |                      |  
                         |                      |  
  Broad DiD suite        |  diff-diff ←(today)  |  ← (opportunity)
                         |                      |
```

Note: DoWhy and CausalML are broad causal inference platforms but don't specialize in DiD — they occupy a different map entirely. The open quadrant is specifically **broad DiD + business framing** in Python. This is a real but narrower opportunity than a generic "causal AI" framing would suggest.

---

## 5. Gap Analysis

### Gap 1: Language & Terminology
**Current**: "ATT", "parallel trends assumption", "forbidden comparisons", "no-anticipation"
**Business needs**: "lift", "incremental impact", "is this result valid?", "how confident are we?"
**Impact**: Business DS bounces off the README. The methodology is powerful but the words are foreign.

### Gap 2: Examples & Use Cases
**Current**: Card-Krueger (1994), Castle Doctrine, unilateral divorce laws
**Business needs**: Marketing campaign lift, product launch rollout, pricing experiment, brand tracking survey, loyalty program evaluation
**Impact**: No "I see myself in this" moment. Business DS can't map their problem to our examples.

### Gap 3: Stakeholder Communication
**Current**: Statistical tables with t-stats, p-values, significance stars
**Business needs**: "The campaign increased awareness by 4.2 percentage points (95% CI: 1.8 to 6.6). This result is robust to violations of the parallel trends assumption up to 1.5x the pre-treatment variation."
**Impact**: Results can't be dropped into a deck or email to leadership without manual translation.

### Gap 4: Automated Validity Assessment
**Current**: 8-step Baker et al. workflow requiring statistical knowledge at each step
**Business needs**: "Run diagnostics → get a traffic-light assessment (green/yellow/red) with plain-English explanation"
**Impact**: Diagnostics are skipped because they're hard to interpret, producing less credible analyses.

### Gap 5: Business Workflow Integration
**Current**: Standalone analysis, academic notebook style
**Business needs**: Integration with common data patterns — survey exports from Qualtrics, geo-level marketing data, event logs from experimentation platforms
**Impact**: Significant data wrangling before analysis can begin. No guidance on common transformations.

### Gap 6: Decision-Oriented Output
**Current**: Estimate → inference → done
**Business needs**: Estimate → "what does this mean?" → "what should we do?" → "how confident should we be?"
**Impact**: Analysis produces a number but not a decision recommendation.

---

## 6. Strategic Recommendations

### Tier 1: Reframe & Reach (Documentation + Positioning)
*Effort: Low-Medium. Impact: High. No code changes required.*

**1a. Business-oriented "Getting Started" guide**
A new entry point alongside the academic quickstart. Frame DiD in business terms:
- "Measuring the impact of interventions when A/B tests aren't possible"
- "Did the campaign/launch/change actually work?"
- Walk through a business scenario end-to-end
- Use business terminology with parenthetical academic equivalents: "lift (average treatment effect on the treated)"

**1b. Terminology bridge**
A reference mapping business ↔ academic language:
| Business Term | Statistical Term |
|---|---|
| Lift / incremental impact | ATT (Average Treatment Effect on the Treated) |
| Test vs. control markets | Treated vs. untreated units |
| Pre-campaign / post-campaign | Pre-treatment / post-treatment |
| "Would the trend have continued?" | Parallel trends assumption |
| Confidence level | Confidence interval |
| "How robust is this?" | Sensitivity analysis |
| Staggered rollout | Staggered adoption |
| Campaign intensity / dose | Continuous treatment |

**1c. README positioning update**
Add a "For Data Scientists" section alongside "For Academics" and "For AI Agents". Highlight business use cases, survey support, and the automated workflow.

**1d. Comparison with business tools**
New docs page: "diff-diff vs GeoLift vs CausalImpact" — showing how we handle the same problems (and more) with greater rigor and flexibility.

### Tier 2: Business Tutorials (Content)
*Effort: Medium. Impact: High.*

Six new tutorial notebooks, each telling a complete business story:

**2a. Marketing Campaign Lift Measurement**
Scenario: E-commerce company runs brand campaign in 8 of 20 DMAs. Measures sales lift.
Estimator: CallawaySantAnna (staggered rollout across DMAs)
Unique value: Shows why GeoLift's synthetic control is insufficient for staggered launches.

**2b. Brand Awareness Survey DiD** *(primary use case)*
Scenario: CPG company runs awareness campaign. Surveys track aided awareness, consideration, purchase intent in test vs. control markets before and after.
Estimator: DifferenceInDifferences + SurveyDesign (strata, PSU, weights)
Unique value: Full survey methodology — design effects, replicate weights, subpopulation analysis. No other Python tool can do this.

**2c. Product Launch Regional Rollout**
Scenario: SaaS company rolls out new pricing in waves across regions. Measures revenue impact.
Estimator: CallawaySantAnna or EfficientDiD (staggered by region)
Unique value: Handles the reality that launches aren't simultaneous.

**2d. Pricing/Promotion Impact**
Scenario: Retailer changes pricing in some stores. Measures unit sales and revenue.
Estimator: ContinuousDiD (varying discount levels as dose)
Unique value: Dose-response curves for different discount levels.

**2e. Loyalty Program Evaluation**
Scenario: Company launches loyalty program in some markets. Measures retention and LTV.
Estimator: TripleDifference (market × eligible × post)
Unique value: DDD handles the fact that only eligible customers can enroll.

**2f. Geo-Experiment with Few Markets**
Scenario: Brand runs campaign in 3 test markets with 15 control markets.
Estimator: SyntheticDiD (few treated units)
Unique value: Direct comparison with GeoLift/CausalImpact, showing when each is appropriate.

### Tier 3: Convenience Layer (API Additions)
*Effort: Medium-High. Impact: High for adoption.*

**3a. `BusinessReport` class**
Generates stakeholder-ready output from any results object. Uses only existing dependencies (numpy/pandas/scipy for computation, string formatting for output). Rich export formats (PowerPoint, HTML) would be optional extras via `pip install diff-diff[reporting]` to preserve the core dependency policy.

```python
from diff_diff import BusinessReport

report = BusinessReport(results)
report.summary()
# "The campaign increased awareness by 4.2 pp (95% CI: 1.8-6.6, p=0.003).
#  This is statistically significant at the 99% level.
#  Robustness: The result holds under parallel trends violations up to 1.5x
#  the observed pre-period variation."

report.export_markdown() # Always available -- plain text/markdown for Notion/Confluence/email
report.export_slide()    # Requires diff-diff[reporting] extra (python-pptx)
```

**3b. `DiagnosticReport` -- automated validity assessment**
Wraps existing diagnostic functions into a unified runner with plain-English interpretation. The check battery maps to existing capabilities:
- Parallel trends -> `check_parallel_trends()` existing function
- Sensitivity -> `HonestDiD` with default M grid
- Placebo -> `run_all_placebo_tests()` existing function
- Effect stability -> coefficient of variation across cohort effects

Traffic-light thresholds (green/yellow/red) are a design decision that needs careful thought -- naive thresholds risk false confidence. The initial version should present results descriptively with plain-English interpretation rather than hard pass/fail gates. Example:

```python
from diff_diff import DiagnosticReport

diag = DiagnosticReport(results)
diag.run_all()
# Parallel trends: No significant pre-trends detected (joint p=0.42)
# Sensitivity: ATT sign stable through M=1.5; CI includes zero at M=2.0
# Placebo: Pre-period placebo ATT = 0.003 (p=0.91), consistent with no effect
# Cohort heterogeneity: ATT ranges from 2.1 to 5.8 across cohorts (CV=0.38)
#
# Interpretation: Results appear credible. The main caveat is moderate
# heterogeneity across cohorts -- consider reporting group-specific effects.
```

**3c. Business data generators**
These would be thin wrappers around existing generators (`generate_did_data`, `generate_staggered_data`, `generate_survey_did_data`) with business-friendly parameter names and defaults -- not new DGPs. The value is discoverability and narrative framing, not new statistical machinery.

```python
from diff_diff import generate_campaign_data  # wraps generate_staggered_data

data = generate_campaign_data(
    n_markets=20, n_treated_markets=8, n_months=12,
    lift=0.05, noise=0.02
)
# Returns DataFrame with columns: market, month, sales, campaign_active, campaign_start_month
# (vs. unit, time, outcome, treatment, first_treat)
```

**3d. Deferred: `QuickDiD` simplified entry point**
Originally proposed as an auto-selecting estimator, but auto-selection risks encouraging methodologically unsound analysis -- the very problem Baker et al. (2025) warns against. Defer this until the business tutorial content validates whether users actually need it, or whether good documentation + `practitioner_next_steps()` is sufficient guidance.

### Tier 4: Ecosystem & Integration (Longer-term)
*Effort: High. Impact: Medium-High (broadens reach).*

**4a. Integration guides**
- "Using diff-diff with Databricks/Spark" -- handling large datasets
- "diff-diff in Jupyter dashboards" -- interactive analysis templates
- "Connecting survey platforms (Qualtrics, SurveyMonkey) to diff-diff" -- data pipeline guides

**4b. Decision framework documentation**
"Which method should I use?" framed for business contexts:
- "I ran a campaign in some markets" -> CallawaySantAnna
- "I have only 3 test markets" -> SyntheticDiD
- "Campaign rolled out at different times" -> Staggered estimators
- "I varied the spending level" -> ContinuousDiD
- "I have survey data with complex sampling" -> Any estimator + SurveyDesign
Not the academic flowchart -- a business decision tree.

**4c. Presentation/export templates**
- PowerPoint slide generator from results
- Markdown report for Notion/Confluence
- HTML dashboard widget

---

## 7. Interaction with Existing Roadmap

The project has an existing ROADMAP.md covering Phase 10 (survey academic credibility), future estimators, and research directions. This strategy supplements rather than replaces it:

**Directly subsumed items:**
- **10g. "Practitioner guidance: when does survey design matter?"** -- this becomes part of the business tutorials and Getting Started guide. No longer a standalone item.
- **survey_aggregate() helper** -- the microdata-to-panel workflow helper is directly relevant for Persona A (survey data from BRFSS/ACS -> geographic panel). Should be prioritized alongside business tutorials.

**Reprioritized by business use cases:**
- **de Chaisemartin-D'Haultfouille (reversible treatments)** -- marketing interventions frequently switch on/off (seasonal campaigns, promotions). This estimator becomes higher priority for business DS than for academics. Should move up in the roadmap.
- **10e. Position paper / arXiv preprint** -- still valuable for academic credibility but not on the critical path for business DS adoption.

**Unchanged:**
- Future estimators (Local Projections DiD, Causal Duration, etc.) and long-term research directions remain academic-oriented and unaffected by this strategy.

---

## 8. Prioritized Roadmap

### Phase 1: Foundation
*Goal: Make diff-diff discoverable and approachable for business DS*

1. Business "Getting Started" guide (1a)
2. Terminology bridge as supplement within business docs, not standalone (1b)
3. README "For Data Scientists" section (1c)
4. Business decision tree -- "which method should I use?" (4b)
5. Brand Awareness Survey DiD tutorial -- the lead use case (2b)

**Why start here**: Zero code changes. Maximum positioning impact. The survey tutorial showcases our unique capability (survey design support) in the context that matters most to the user.

**Validation gate before Phase 2**: After Phase 1 ships, look for adoption signals -- tutorial page views, GitHub issues from business users, PyPI download trajectory. These signals determine how aggressively to invest in Phases 2-3.

### Phase 2: Business Content
*Goal: Provide end-to-end examples for each major persona*

Tutorials in priority order (ship incrementally, not all at once):

6. Marketing Campaign Lift tutorial (2a) -- **highest priority after survey**
7. Geo-Experiment tutorial (2f) -- captures GeoLift/CausalImpact search traffic
8. Comparison page: diff-diff vs GeoLift vs CausalImpact (1d)
9. Product Launch Rollout tutorial (2c)
10. Pricing/Promotion Impact tutorial (2d)
11. Loyalty Program tutorial using DDD (2e)

### Phase 3: Convenience Layer
*Goal: Reduce time-to-insight and enable stakeholder communication*

12. `BusinessReport` class (3a) -- core uses only numpy/pandas/scipy; rich export via optional `[reporting]` extra
13. `DiagnosticReport` descriptive assessment (3b)
14. Business data generator wrappers (3c)
15. `survey_aggregate()` helper from existing roadmap -- directly enables the survey tutorial workflow

### Phase 4: Platform (Longer-term)
*Goal: Integrate into business DS workflows*

16. Integration guides (4a)
17. Export templates (4c)
18. AI agent integration -- position DiagnosticReport and BusinessReport as tools AI agents can invoke on behalf of business DS (leveraging existing `practitioner_next_steps()` infrastructure)

---

## 9. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Oversimplifying may undermine credibility with academic users | Keep business layer additive -- don't change existing academic interface. Business tools translate, not replace. |
| Business tutorials may encourage methodologically unsound analysis | Embed guardrails: DiagnosticReport flags issues, tutorials emphasize assumption checking in business language |
| Scope creep | Phase 1 is documentation-only. Validate adoption signals before investing in code (Phase 3+). |
| Maintaining two audiences | Shared codebase, separate entry points. Like scikit-learn serving both ML engineers and researchers. |

---

## 10. Success Metrics

**Leading indicators (measurable after Phase 1):**
- Tutorial notebook page views / nbviewer hits for business tutorials
- GitHub issues or discussions mentioning business use cases (campaigns, surveys, geo-experiments)
- Search console impressions for business-oriented queries ("python campaign lift", "python geo experiment", "python survey did")

**Lagging indicators (Phases 2-3):**
- PyPI download trajectory (month-over-month growth rate, not absolute)
- GitHub stars from non-academic profiles
- External blog posts or talks using diff-diff for business analysis

**Phase 1 -> Phase 2 gate**: At least one of: (a) 3+ GitHub issues from business users, (b) measurable search impression growth for business queries, (c) qualitative signal that the business framing is resonating (social media, conference mentions). If none after 8 weeks, revisit the strategy before investing in code changes.

---

## 11. Bottom Line

We have the best DiD engine in Python. What we don't have is the business packaging. The methodology is sound, the survey support is unique, the diagnostic suite is unmatched. But a marketing data scientist looking at our docs sees academic econometrics, not their problem.

The fix is mostly about **framing, examples, and a thin convenience layer** -- not rebuilding the core. Phase 1 requires zero code changes. Phases 2-3 add content and lightweight APIs. The competitive window is open because no one else is targeting this intersection: comprehensive DiD + business data science + Python.

The survey use case is the sharpest wedge. No other tool in any language combines complex survey design with modern heterogeneity-robust DiD estimators. Lead with that, then broaden.
