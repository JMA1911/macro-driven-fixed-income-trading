# Macro-Driven Fixed Income Trading Strategies

## Independent Quantitative Research Project

This repository contains the Python implementation and research report for an independent quantitative finance project investigating whether macroeconomic information embedded in the US Treasury yield curve can be transformed into systematic fixed-income trading strategies.

The project develops and evaluates both directional duration and relative-value Treasury ETF strategies using economically motivated yield signals, systematic portfolio construction, and comprehensive robustness testing.

---

## Research Overview

US Treasury yields contain valuable information about monetary policy expectations, economic activity, and financial conditions.

This project investigates whether these macroeconomic signals can be exploited to construct systematic fixed-income trading strategies. Multiple yield-curve signals are developed and evaluated before being extended using dynamic threshold selection, continuous position sizing, volatility targeting, and DV01-neutral portfolio construction.

Strategies are tested using walk-forward optimisation, historical backtesting, stress testing, and parametric VAR bootstrap simulation to assess robustness across a wide range of market environments.

---

## Methodology

The project combines:

- Macroeconomic yield-curve signal engineering
- Directional duration and relative-value Treasury ETF strategies
- Walk-forward optimisation
- Dynamic threshold selection
- Continuous position sizing
- Volatility targeting
- DV01-neutral portfolio construction
- Historical stress testing
- Parametric VAR bootstrap robustness analysis

Market data were obtained from Yahoo Finance and the Federal Reserve Economic Data (FRED) database.

---

## Repository Contents

### Research Report

**Macro_Driven_Fixed_Income_Trading_Strategies.pdf**

The report contains:

- Economic motivation
- Exploratory data analysis
- Signal development
- Strategy construction
- Walk-forward optimisation
- Historical backtesting
- Stress testing
- Bootstrap robustness analysis
- Risk analysis and discussion

### Python Implementation

| File | Description |
|------|-------------|
| `macro_fixed_income_trading.py` | Complete end-to-end research implementation including data collection, signal construction, strategy development, backtesting, robustness analysis, and visualisation. |

---

## Key Results

The strongest-performing strategy combined relative-value Treasury spread trading with continuous position sizing and volatility targeting.

| Metric | Strategy |
|--------|---------:|
| Total Return | **21.9%** |
| Annualised Return | **1.10%** |
| Sharpe Ratio | **0.67** |
| Maximum Drawdown | **-3.03%** |

Robustness analysis demonstrated:

- Profitable in **98%** of parametric VAR bootstrap simulations
- Consistent profitability across major historical macroeconomic regimes
- Strong downside protection during the 2022 interest-rate tightening cycle
- Improved risk-adjusted performance relative to passive Treasury benchmarks

---

## Research Abstract

This project investigates whether macroeconomic information embedded in US Treasury yields can be used to develop systematic fixed-income trading strategies. Directional duration and relative-value Treasury ETF strategies are constructed using economically motivated yield signals, then extended through dynamic threshold selection, continuous position sizing, and volatility targeting.

The strategies are evaluated using walk-forward optimisation, historical backtesting, stress testing, and parametric bootstrap simulation. Performance is assessed using both traditional investment metrics and fixed-income-specific risk measures, including duration, DV01, Value at Risk, and Expected Shortfall.

The results indicate that relative-value yield curve strategies provide stronger risk-adjusted performance than directional duration strategies. In particular, a continuously scaled and volatility-targeted TLT–IEF ETF spread strategy achieves the best balance between return generation, volatility control, and downside protection while remaining robust across historical stress periods and simulated yield-curve scenarios.

---

## Technologies

- Python
- NumPy
- Pandas
- SciPy
- Statsmodels
- Matplotlib
- Scikit-learn
- yfinance
- fredapi

---

## Author

**Jack Adams**

MSc Financial Mathematics (Distinction)

University of Leeds

LinkedIn: https://www.linkedin.com/in/jackmadams
