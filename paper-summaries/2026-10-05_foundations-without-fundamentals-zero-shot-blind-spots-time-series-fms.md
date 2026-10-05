# Foundations without Fundamentals: Zero-Shot Blind Spots in Time Series Foundation Models

**arXiv:** 2610.02058  
**Authors:** Nafiseh Ghoroghchian, Haipeng Zhang, Shuyi Han, Alex Labach, George Stein  
**Affiliation:** ServiceNow Research; McGill University  
**Date:** October 1, 2026

---

## One-Line Summary

SimpleTimeBench exposes a striking gap: Chronos-2, Moirai, and Toto—prominent multivariate time series foundation models—frequently fail at forecasting monotonic trends, periodic signals, and leading-indicator scenarios that any well-calibrated model should solve trivially, and these failures persist on real-world sensor data.

---

## Problem

Time Series Foundation Models (TSFMs) trained on massive corpora of diverse time series data have demonstrated strong zero-shot performance on broad downstream benchmarks, suggesting they have internalized general temporal structure. However, broad benchmark performance conflates many factors—dataset coverage, distributional similarity to training data, evaluation metric choice—making it difficult to assess whether these models have genuinely learned basic temporal primitives. If TSFMs fail at simple, deterministic temporal patterns, their benchmark success may reflect data overlap rather than true temporal reasoning, with significant implications for deployment in industrial, scientific, and healthcare settings where such patterns are common.

---

## Why Existing Approaches Fall Short

- **Broad benchmarks conflate capabilities:** Standard TSFT benchmarks (e.g., Monash, ETT, M4) contain thousands of diverse series; good aggregate performance does not guarantee competence on any specific temporal primitive.
- **No unit tests for temporal logic:** There is no existing benchmark analogous to unit tests in software engineering that verifies whether a model handles basic temporal building blocks correctly.
- **Covariate underutilization is invisible:** Existing benchmarks rarely provide systematic tests of whether a model can leverage exogenous covariates (leading indicators), so failures to exploit available causal information go undetected.

---

## Core Method

**SimpleTimeBench.** The authors construct a diagnostic benchmark consisting of three families of "unit tests":

1. **Monotonic trend tests:** Univariate time series that are strictly monotonically increasing or decreasing with added noise. A well-calibrated model should forecast near-continuation of the trend; any systematic reversal or flattening constitutes a failure.

2. **Periodic signal tests:** Univariate and multivariate series with known periods (daily, weekly, annual) and controlled amplitudes. A model with internalized temporal periodicity should closely track the cycle; failure to do so reveals missing periodic inductive bias.

3. **Leading indicator tests:** Multivariate series in which one channel (the leading indicator) Granger-causes another channel with a known lag, so the model has complete information to produce accurate zero-shot forecasts. Failure here reveals inability to exploit exogenous covariates.

Each test family includes a range of noise levels, trend slopes, periods, and lag lengths to characterize the model's sensitivity rather than just binary pass/fail.

---

## Technical Formulation

Let $\mathbf{x}_{1:T}$ be the historical context and $\mathbf{x}_{T+1:T+H}$ the forecast horizon. For a monotonic trend:
$$x_t = \alpha t + \sigma \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,1)$$
with known slope $\alpha$ and noise $\sigma$. The oracle forecast is $\hat{x}_{T+h} = \alpha(T+h)$; a TSFM passes the test if its Mean Absolute Scaled Error (MASE) is within $\delta$ of the oracle.

For leading indicator tests:
$$y_t = \phi(x_{t-k}) + \sigma \epsilon_t$$
where $x_{t-k}$ is the observed leading indicator with lag $k$. The oracle MSE is $\sigma^2$; a model that fully utilizes the leading indicator achieves this bound. Failure is defined as MSE exceeding $2\sigma^2$.

SimpleTimeBench reports per-primitive pass rates and failure mode distributions alongside standard metrics.

---

## Learning or Inference Procedure

SimpleTimeBench is an evaluation-only benchmark; no training is required or performed. The evaluation procedure:

1. Generate controlled synthetic series for each primitive type at multiple difficulty levels.
2. Run each TSFM (Chronos-2, Moirai, Toto) in zero-shot mode on each series, using the model's default context window.
3. Compute MASE, sMAPE, and leading-indicator utilization rate for each test family.
4. Compare against oracle forecasters (linear trend extrapolation, periodicity-aware Fourier baseline) and classical models (ARIMA, Theta).
5. Validate failure modes on real-world sensor datasets (building energy meters, meteorological stations) that contain the tested primitives.

---

## What the Guarantee Says

SimpleTimeBench is a diagnostic tool rather than a performance metric; it does not claim to evaluate general TSFM quality. The paper's central claim is falsifiability: if a model fails on SimpleTimeBench, then by construction there exist natural time series inputs (with no distributional shift or long-tail characteristics) on which the model makes poor forecasts that a trivially simple oracle would get right. This constitutes a concrete failure mode regardless of the model's benchmark rank.

---

## Experimental Findings

- **Monotonic trends:** Chronos-2 reverses the direction of a clearly increasing trend in approximately 18% of test cases at low noise levels; Moirai flattens trends early in about 23% of cases.
- **Periodic signals:** All three models exhibit poor calibration on annual-period signals, with MASE ratios relative to oracle exceeding 2.0 in over 30% of test instances.
- **Leading indicator exploitation:** Average leading-indicator utilization rate is below 40% for all three models, meaning that over 60% of the information in available covariates is discarded. Classical ARIMAX models achieve over 80% utilization on the same series.
- **Real-world sensors:** On building energy meter data where weekly and daily periodicity is prominent, TSFMs consistently underperform a tuned Fourier decomposition model, with the gap correlated with the strength of the periodic signal.
- **Classical model comparison:** ARIMA and Theta match or exceed TSFMs on monotonic trend and leading indicator tests, despite being far simpler models, confirming that scale alone does not guarantee basic temporal inductive biases.

---

## Ablations and Interpretation

- **Context window sensitivity:** Providing a longer historical context that contains multiple complete periods modestly improves periodic signal tracking (5–10% MASE reduction) but does not eliminate failures, suggesting the issue is structural rather than context-length limited.
- **Model size:** Larger variants of Chronos-2 and Moirai do not substantially improve SimpleTimeBench performance, indicating that the blind spots are not simply a function of parameter count.
- **Fine-tuning:** A small amount of domain-specific fine-tuning on examples containing each primitive dramatically closes the gap (pass rates exceed 85%), confirming that the inductive biases are learnable but not present in zero-shot form.

---

## Reference

Nafiseh Ghoroghchian, Haipeng Zhang, Shuyi Han, Alex Labach, George Stein. **Foundations without Fundamentals: Zero-Shot Blind Spots in Time Series FMs**. arXiv:2610.02058, October 2026.  
https://arxiv.org/abs/2610.02058
