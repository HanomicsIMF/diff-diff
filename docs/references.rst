References
==========

This library implements methods from the following scholarly works.

Difference-in-Differences
-------------------------

- **Ashenfelter, O., & Card, D. (1985).** "Using the Longitudinal Structure of Earnings to Estimate the Effect of Training Programs." *The Review of Economics and Statistics*, 67(4), 648-660. https://doi.org/10.2307/1924810

- **Card, D., & Krueger, A. B. (1994).** "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania." *The American Economic Review*, 84(4), 772-793. https://www.jstor.org/stable/2118030

- **Angrist, J. D., & Pischke, J.-S. (2009).** *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press. Chapter 5: Differences-in-Differences.

Two-Way Fixed Effects
---------------------

- **Wooldridge, J. M. (2010).** *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

- **Imai, K., & Kim, I. S. (2021).** "On the Use of Two-Way Fixed Effects Regression Models for Causal Inference with Panel Data." *Political Analysis*, 29(3), 405-415. https://doi.org/10.1017/pan.2020.33

Wooldridge ETWFE
----------------

- **Wooldridge, J. M. (2025).** "Two-Way Fixed Effects, the Two-Way Mundlak Regression, and Difference-in-Differences Estimators." *Empirical Economics*, 69(5), 2545-2587. (Published version of NBER Working Paper 29154.)

  Primary source for the saturated OLS ETWFE design implemented in our ``WooldridgeDiD`` class.

- **Wooldridge, J. M. (2023).** "Simple Approaches to Nonlinear Difference-in-Differences with Panel Data." *The Econometrics Journal*, 26(3), C31-C66. https://doi.org/10.1093/ectj/utad016

  Secondary source for the logit/Poisson QMLE (ASF-based ATT) extensions in ``WooldridgeDiD``.

Robust Standard Errors
----------------------

- **White, H. (1980).** "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity." *Econometrica*, 48(4), 817-838. https://doi.org/10.2307/1912934

- **MacKinnon, J. G., & White, H. (1985).** "Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties." *Journal of Econometrics*, 29(3), 305-325. https://doi.org/10.1016/0304-4076(85)90158-7

- **Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).** "Robust Inference With Multiway Clustering." *Journal of Business & Economic Statistics*, 29(2), 238-249. https://doi.org/10.1198/jbes.2010.07136

Wild Cluster Bootstrap
----------------------

- **Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008).** "Bootstrap-Based Improvements for Inference with Clustered Errors." *The Review of Economics and Statistics*, 90(3), 414-427. https://doi.org/10.1162/rest.90.3.414

- **Webb, M. D. (2014).** "Reworking Wild Bootstrap Based Inference for Clustered Errors." Queen's Economics Department Working Paper No. 1315. https://www.econ.queensu.ca/sites/econ.queensu.ca/files/qed_wp_1315.pdf

- **MacKinnon, J. G., & Webb, M. D. (2018).** "The Wild Bootstrap for Few (Treated) Clusters." *The Econometrics Journal*, 21(2), 114-135. https://doi.org/10.1111/ectj.12107

Placebo Tests and DiD Diagnostics
---------------------------------

- **Bertrand, M., Duflo, E., & Mullainathan, S. (2004).** "How Much Should We Trust Differences-in-Differences Estimates?" *The Quarterly Journal of Economics*, 119(1), 249-275. https://doi.org/10.1162/003355304772839588

Synthetic Control Method
------------------------

- **Abadie, A., & Gardeazabal, J. (2003).** "The Economic Costs of Conflict: A Case Study of the Basque Country." *The American Economic Review*, 93(1), 113-132. https://doi.org/10.1257/000282803321455188

- **Abadie, A., Diamond, A., & Hainmueller, J. (2010).** "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association*, 105(490), 493-505. https://doi.org/10.1198/jasa.2009.ap08746

- **Abadie, A., Diamond, A., & Hainmueller, J. (2015).** "Comparative Politics and the Synthetic Control Method." *American Journal of Political Science*, 59(2), 495-510. https://doi.org/10.1111/ajps.12116

Synthetic Difference-in-Differences
-----------------------------------

- **Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).** "Synthetic Difference-in-Differences." *American Economic Review*, 111(12), 4088-4118. https://doi.org/10.1257/aer.20190159

Triply Robust Panel (TROP)
--------------------------

- **Athey, S., Imbens, G. W., Qu, Z., & Viviano, D. (2025).** "Triply Robust Panel Estimators." *Working Paper*. https://arxiv.org/abs/2508.21536

  This paper introduces the TROP estimator which combines three robustness components:

  - **Factor model adjustment**: Low-rank factor structure via SVD removes unobserved confounders
  - **Unit weights**: Synthetic control style weighting for optimal comparison
  - **Time weights**: SDID style time weighting for informative pre-periods

  TROP is particularly useful when there are unobserved time-varying confounders with a factor structure that affect different units differently over time.

Triple Difference (DDD)
-----------------------

- **Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).** "Better Understanding Triple Differences Estimators." *Working Paper*. https://arxiv.org/abs/2505.09942

  This paper shows that common DDD implementations (taking the difference between two DiDs, or applying three-way fixed effects regressions) are generally invalid when identification requires conditioning on covariates. The ``TripleDifference`` class implements their regression adjustment, inverse probability weighting, and doubly robust estimators.

- **Gruber, J. (1994).** "The Incidence of Mandated Maternity Benefits." *American Economic Review*, 84(3), 622-641. https://www.jstor.org/stable/2118071

  Classic paper introducing the Triple Difference design for policy evaluation.

- **Olden, A., & Møen, J. (2022).** "The Triple Difference Estimator." *The Econometrics Journal*, 25(3), 531-553. https://doi.org/10.1093/ectj/utac010

Parallel Trends and Pre-Trend Testing
-------------------------------------

- **Roth, J. (2022).** "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." *American Economic Review: Insights*, 4(3), 305-322. https://doi.org/10.1257/aeri.20210236

- **Lakens, D. (2017).** "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses." *Social Psychological and Personality Science*, 8(4), 355-362. https://doi.org/10.1177/1948550617697177

Honest DiD / Sensitivity Analysis
---------------------------------

The ``HonestDiD`` module implements sensitivity analysis methods for relaxing the parallel trends assumption.

- **Rambachan, A., & Roth, J. (2023).** "A More Credible Approach to Parallel Trends." *The Review of Economic Studies*, 90(5), 2555-2591. https://doi.org/10.1093/restud/rdad018

  This paper introduces the "Honest DiD" framework implemented in our ``HonestDiD`` class:

  - **Relative Magnitudes (ΔRM)**: Bounds post-treatment violations by a multiple of observed pre-treatment violations
  - **Smoothness (ΔSD)**: Bounds on second differences of trend violations, allowing for linear extrapolation of pre-trends
  - **Breakdown Analysis**: Finding the smallest violation magnitude that would overturn conclusions
  - **Robust Confidence Intervals**: Valid inference under partial identification

- **Roth, J., & Sant'Anna, P. H. C. (2023).** "When Is Parallel Trends Sensitive to Functional Form?" *Econometrica*, 91(2), 737-747. https://doi.org/10.3982/ECTA19402

  Discusses functional form sensitivity in parallel trends assumptions, relevant to understanding when smoothness restrictions are appropriate.

Multi-Period and Staggered Adoption
-----------------------------------

- **Borusyak, K., Jaravel, X., & Spiess, J. (2024).** "Revisiting Event-Study Designs: Robust and Efficient Estimation." *Review of Economic Studies*, 91(6), 3253-3285. https://doi.org/10.1093/restud/rdae007

  This paper introduces the imputation estimator implemented in our ``ImputationDiD`` class:

  - **Efficient imputation**: OLS on untreated observations, impute counterfactuals, aggregate
  - **Conservative variance**: Theorem 3 clustered variance estimator with auxiliary model
  - **Pre-trend test**: Independent of treatment effect estimation (Proposition 9)
  - **Efficiency gains**: ~50% shorter CIs than Callaway-Sant'Anna under homogeneous effects

- **Callaway, B., & Sant'Anna, P. H. C. (2021).** "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230. https://doi.org/10.1016/j.jeconom.2020.12.001

- **Sant'Anna, P. H. C., & Zhao, J. (2020).** "Doubly Robust Difference-in-Differences Estimators." *Journal of Econometrics*, 219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003

- **Sun, L., & Abraham, S. (2021).** "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*, 225(2), 175-199. https://doi.org/10.1016/j.jeconom.2020.09.006

- **Gardner, J. (2022).** "Two-stage differences in differences." *arXiv preprint arXiv:2207.05943*. https://arxiv.org/abs/2207.05943

- **Butts, K., & Gardner, J. (2022).** "did2s: Two-Stage Difference-in-Differences." *The R Journal*, 14(1), 162-173. https://doi.org/10.32614/RJ-2022-048

- **de Chaisemartin, C., & D'Haultfœuille, X. (2020).** "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects." *American Economic Review*, 110(9), 2964-2996. https://doi.org/10.1257/aer.20181169

- **de Chaisemartin, C., & D'Haultfœuille, X. (2022, revised 2024).** "Difference-in-Differences Estimators of Intertemporal Treatment Effects." *NBER Working Paper* 29873. https://www.nber.org/papers/w29873

  Dynamic companion to the 2020 paper. Web Appendix Section 3.7.3 contains the cohort-recentered plug-in variance formula implemented in our ``ChaisemartinDHaultfoeuille`` class.

- **Goodman-Bacon, A. (2021).** "Difference-in-Differences with Variation in Treatment Timing." *Journal of Econometrics*, 225(2), 254-277. https://doi.org/10.1016/j.jeconom.2021.03.014

- **Wing, C., Freedman, S. M., & Hollingsworth, A. (2024).** "Stacked Difference-in-Differences." *NBER Working Paper* 32054. https://www.nber.org/papers/w32054

- **Chen, X., Sant'Anna, P. H. C., & Xie, H. (2025).** "Efficient Difference-in-Differences and Event Study Estimators." *Working Paper*.

  Primary source for the optimal-weighting / PT-All / PT-Post efficient DiD implemented in our ``EfficientDiD`` class.

- **Baker, A., Callaway, B., Cunningham, S., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2025).** "Difference-in-Differences Designs: A Practitioner's Guide." *arXiv preprint* arXiv:2503.13323. https://arxiv.org/abs/2503.13323

  Source for the 8-step practitioner workflow surfaced via ``diff_diff.get_llm_guide("practitioner")`` and the README ``## Practitioner Workflow`` section. See ``docs/methodology/REGISTRY.md`` for the diff-diff renumbering and per-step deviations.

Continuous Treatment DiD
------------------------

- **Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2024).** "Difference-in-Differences with a Continuous Treatment." *NBER Working Paper* 32117. https://www.nber.org/papers/w32117

  Primary source for ATT(d), ACRT, dose-response curves, and B-spline flexibility implemented in our ``ContinuousDiD`` class.

Power Analysis
--------------

- **Bloom, H. S. (1995).** "Minimum Detectable Effects: A Simple Way to Report the Statistical Power of Experimental Designs." *Evaluation Review*, 19(5), 547-556. https://doi.org/10.1177/0193841X9501900504

- **Burlig, F., Preonas, L., & Woerman, M. (2020).** "Panel Data and Experimental Design." *Journal of Development Economics*, 144, 102458. https://doi.org/10.1016/j.jdeveco.2020.102458

  Essential reference for power analysis in panel DiD designs. Discusses how serial correlation (ICC) affects power and provides formulas for panel data settings.

- **Djimeu, E. W., & Houndolo, D.-G. (2016).** "Power Calculation for Causal Inference in Social Science: Sample Size and Minimum Detectable Effect Determination." *Journal of Development Effectiveness*, 8(4), 508-527. https://doi.org/10.1080/19439342.2016.1244555

General Causal Inference
------------------------

- **Imbens, G. W., & Rubin, D. B. (2015).** *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.

- **Cunningham, S. (2021).** *Causal Inference: The Mixtape*. Yale University Press. https://mixtape.scunning.com/
