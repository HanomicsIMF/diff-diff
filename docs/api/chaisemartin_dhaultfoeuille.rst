de Chaisemartin-D'Haultfœuille (dCDH) DiD
============================================

The only modern staggered DiD estimator in diff-diff that handles
**non-absorbing (reversible) treatments** — treatment may switch on AND
off over time.

This module implements the methodology from de Chaisemartin & D'Haultfœuille
(2020), "Two-Way Fixed Effects Estimators with Heterogeneous Treatment
Effects", *American Economic Review*. Phase 1 ships the contemporaneous-
switch estimator ``DID_M`` from the AER 2020 paper, which is mathematically
identical to ``DID_1`` (horizon ``l = 1``) of the dynamic companion paper
(de Chaisemartin & D'Haultfœuille, 2024, NBER WP 29873). The Phase 1 class
is forward-compatible with later phases — Phase 2 will add multi-horizon
event-study output ``DID_l`` for ``l > 1`` on the same class, and Phase 3
will add covariate adjustment.

The estimator:

1. Aggregates individual-level panel data to ``(group, time)`` cells
2. Drops multi-switch groups by default (matches R ``DIDmultiplegtDYN``)
3. Filters singleton-baseline groups (footnote 15 of the dynamic paper)
4. Computes per-period joiner (``DID_{+,t}``) and leaver (``DID_{-,t}``)
   contributions via Theorem 3 of the AER 2020 paper
5. Aggregates them into ``DID_M``, the joiners-only ``DID_+``, and the
   leavers-only ``DID_-``
6. Computes the single-lag placebo ``DID_M^pl`` (Theorem 4)
7. Optionally computes the TWFE decomposition diagnostic from Theorem 1
   (per-cell weights, fraction negative, ``sigma_fe``)
8. Inference uses the cohort-recentered analytical plug-in variance from
   Web Appendix Section 3.7.3 of the dynamic paper, optionally
   complemented by a multiplier bootstrap clustered at the group level

**When to use ChaisemartinDHaultfoeuille:**

- Treatment can switch on **and** off over time (e.g., marketing campaigns,
  seasonal promotions, on/off policy cycles)
- You need separate joiners (``DID_+``) and leavers (``DID_-``) views, plus
  the aggregate ``DID_M``
- You want a built-in placebo and a TWFE decomposition diagnostic from the
  same fit
- You want a Python implementation that matches R ``DIDmultiplegtDYN`` at
  ``l = 1``

All other staggered estimators in diff-diff (:class:`~diff_diff.CallawaySantAnna`,
:class:`~diff_diff.SunAbraham`, :class:`~diff_diff.ImputationDiD`,
:class:`~diff_diff.TwoStageDiD`, :class:`~diff_diff.EfficientDiD`,
:class:`~diff_diff.WooldridgeDiD`) assume treatment is **absorbing** —
once treated, stays treated. ``ChaisemartinDHaultfoeuille`` is the only
library option for non-absorbing treatments.

**References:**

- de Chaisemartin, C. & D'Haultfœuille, X. (2020). Two-Way Fixed Effects
  Estimators with Heterogeneous Treatment Effects. *American Economic
  Review*, 110(9), 2964-2996.
- de Chaisemartin, C. & D'Haultfœuille, X. (2022, revised 2024).
  Difference-in-Differences Estimators of Intertemporal Treatment
  Effects. NBER Working Paper 29873.

.. module:: diff_diff.chaisemartin_dhaultfoeuille

ChaisemartinDHaultfoeuille
--------------------------

Main estimator class for de Chaisemartin-D'Haultfœuille (dCDH) DiD estimation.
The alias :class:`~diff_diff.DCDH` is also available.

.. autoclass:: diff_diff.ChaisemartinDHaultfoeuille
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~ChaisemartinDHaultfoeuille.fit
      ~ChaisemartinDHaultfoeuille.get_params
      ~ChaisemartinDHaultfoeuille.set_params

ChaisemartinDHaultfoeuilleResults
---------------------------------

Results container for dCDH estimation.

.. autoclass:: diff_diff.ChaisemartinDHaultfoeuilleResults
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~ChaisemartinDHaultfoeuilleResults.summary
      ~ChaisemartinDHaultfoeuilleResults.print_summary
      ~ChaisemartinDHaultfoeuilleResults.to_dataframe

DCDHBootstrapResults
--------------------

Multiplier-bootstrap inference results, populated when ``n_bootstrap > 0``.

.. autoclass:: diff_diff.DCDHBootstrapResults
   :members:
   :undoc-members:
   :show-inheritance:

Convenience Function
--------------------

.. autofunction:: diff_diff.chaisemartin_dhaultfoeuille

Standalone TWFE Decomposition Diagnostic
----------------------------------------

The TWFE decomposition diagnostic from Theorem 1 of de Chaisemartin &
D'Haultfœuille (2020) is also available as a standalone function for
users who want the diagnostic without fitting the full estimator. It
returns per-cell weights, the fraction of treated cells with negative
weights, and ``sigma_fe`` — the smallest standard deviation of per-cell
treatment effects that could flip the sign of the plain TWFE coefficient.

.. autofunction:: diff_diff.twowayfeweights

.. autoclass:: diff_diff.chaisemartin_dhaultfoeuille.TWFEWeightsResult
   :members:

Example Usage
-------------

Basic usage with reversible treatment::

    from diff_diff import ChaisemartinDHaultfoeuille
    from diff_diff.prep import generate_reversible_did_data

    data = generate_reversible_did_data(
        n_groups=80, n_periods=6, pattern="single_switch", seed=42,
    )

    est = ChaisemartinDHaultfoeuille()
    results = est.fit(
        data,
        outcome="outcome",
        group="group",
        time="period",
        treatment="treatment",
    )
    results.print_summary()

Joiners and leavers views::

    print(f"DID_M (overall):  {results.overall_att:.3f}")
    print(f"DID_+ (joiners):  {results.joiners_att:.3f}")
    print(f"DID_- (leavers):  {results.leavers_att:.3f}")
    print(f"Placebo (DID^pl): {results.placebo_effect:.3f}")

Per-period decomposition::

    for t, cell in results.per_period_effects.items():
        print(
            f"t={t}: DID+={cell['did_plus_t']:.3f} "
            f"({cell['n_10_t']} joiners, {cell['n_00_t']} stable_0 controls)"
        )

Multiplier bootstrap inference::

    est = ChaisemartinDHaultfoeuille(
        n_bootstrap=999, bootstrap_weights="rademacher", seed=42,
    )
    results = est.fit(
        data, outcome="outcome", group="group",
        time="period", treatment="treatment",
    )
    print(f"Bootstrap SE: {results.bootstrap_results.overall_se:.3f}")
    print(f"Bootstrap CI: {results.bootstrap_results.overall_ci}")

Standalone TWFE diagnostic (without fitting the full estimator)::

    from diff_diff import twowayfeweights

    diagnostic = twowayfeweights(
        data, group="group", time="period", treatment="treatment",
    )
    print(f"Plain TWFE coefficient: {diagnostic.beta_fe:.3f}")
    print(f"Fraction of negative weights: {diagnostic.fraction_negative:.3f}")
    print(f"sigma_fe (sign-flipping threshold): {diagnostic.sigma_fe:.3f}")

The ``DCDH`` alias::

    from diff_diff import DCDH

    est = DCDH()  # equivalent to ChaisemartinDHaultfoeuille()
