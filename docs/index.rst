diff-diff: Difference-in-Differences in Python
==============================================

**diff-diff** is a Python library for Difference-in-Differences (DiD) causal inference analysis.
It provides sklearn-like estimators with statsmodels-style output for econometric analysis.

.. code-block:: python

   from diff_diff import DifferenceInDifferences

   # Fit a basic DiD model
   did = DifferenceInDifferences()
   results = did.fit(data, outcome='y', treatment='treated', time='post')
   print(results.summary())

Key Features
------------

- **13+ Estimators**: Basic DiD, TWFE, Event Study, Synthetic DiD, plus modern staggered estimators (Callaway-Sant'Anna, Sun-Abraham, Imputation, Two-Stage, Stacked DiD), advanced methods (TROP, Continuous DiD, Efficient DiD, Triple Difference), and Bacon Decomposition diagnostics
- **Modern Inference**: Robust standard errors, cluster-robust SEs, wild cluster bootstrap, and multiplier bootstrap
- **Assumption Testing**: Parallel trends tests, placebo tests, Bacon decomposition, and comprehensive diagnostics
- **Sensitivity Analysis**: Honest DiD (Rambachan & Roth 2023) for robust inference under parallel trends violations
- **Built-in Datasets**: Real-world datasets from published studies (Card & Krueger, Castle Doctrine, and more)
- **High Performance**: Optional Rust backend for compute-intensive estimators like Synthetic DiD and TROP
- **Publication-Ready Output**: Summary tables, event study plots, and sensitivity analysis figures

Installation
------------

.. code-block:: bash

   pip install diff-diff

For development:

.. code-block:: bash

   pip install diff-diff[dev]

Quick Links
-----------

- :doc:`quickstart` - Get started with basic examples
- :doc:`choosing_estimator` - Which estimator should I use?
- :doc:`troubleshooting` - Common issues and solutions
- :doc:`r_comparison` - Comparison with R packages
- :doc:`python_comparison` - Comparison with Python packages
- :doc:`benchmarks` - Performance benchmarks vs R packages
- :doc:`api/index` - Full API reference

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   quickstart
   choosing_estimator
   troubleshooting
   r_comparison
   python_comparison
   benchmarks

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
