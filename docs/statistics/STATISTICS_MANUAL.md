# Statistical Foundations for Institutional Data Systems

- Author: Simon Parris
- Date: 2026-02-22

## 1. What probability really means

Probability is a structured way to describe uncertainty. It helps teams express how likely events are, instead of relying on intuition alone.

## 2. What a distribution is (simple explanation)

A distribution describes how values are spread out, including what is common, what is rare, and how extreme values behave. Different distributions match different real-world processes.

## 3. What a p-value actually means (no textbook jargon)

A p-value tells you how surprising your result would be if there were no real effect and only random variation was operating. It does not prove a finding is true or false by itself.

## 4. Why statistical significance can be misleading

- Large datasets can make tiny, unimportant effects look "significant"
- Poor data quality or sampling bias can produce misleading results
- Repeated testing can create false positives if not controlled
- Statistical significance does not equal operational importance

## 5. Why institutions require reproducibility

UN, IATA, and CERN-style environments need repeatable methods because results may influence operations, safety, policy, or scientific claims. Reproducibility supports auditability, peer review, and trust.

## 6. How to rebuild this module without AI

1. Create folders: `src/statistics`, `tests/statistics`, `docs/statistics`
2. Implement distribution functions (Normal, Binomial, Poisson)
3. Add synthetic data generator for flight delays (10,000 records)
4. Implement Welch t-test, chi-square test, and confidence interval functions
5. Implement Monte Carlo simulations (10,000 iterations) and save histogram PNG to `reports/`
6. Add pytest tests for numerical correctness and output files
7. Run `python3 -m pytest -q tests/statistics/test_statistics.py`
8. Commit and push with a descriptive message
