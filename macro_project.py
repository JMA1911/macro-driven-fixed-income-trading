import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy import stats
import numpy as np

# =========================
# 1. SETTINGS
# =========================
start_date = "2007-01-01"
end_date = "2026-05-14"  # latest available data

etf_tickers = ["TLT", "IEF", "SHY"]

# =========================
# 2. DOWNLOAD ETF DATA
# =========================
etf_data = yf.download(
    etf_tickers,
    start=start_date,
    end=end_date,
    auto_adjust=False,
    progress=False
)

# Adjusted close prices
adj_close = etf_data["Adj Close"].copy()
adj_close.columns = [f"{ticker}_AdjClose" for ticker in adj_close.columns]

# Daily returns
etf_returns = adj_close.pct_change()
etf_returns.columns = [col.replace("_AdjClose", "_Ret") for col in adj_close.columns]

# =========================
# 3. DOWNLOAD MACRO DATA FROM FRED
# =========================

# 10Y Treasury yield (daily, in percent)
us10y = pdr.DataReader("DGS10", "fred", start_date, end_date)
us10y.columns = ["US10Y_Yield"]

# 3M Treasury yield / Treasury bill rate (daily, in percent)
us3m = pdr.DataReader("DGS3MO", "fred", start_date, end_date)
us3m.columns = ["US3M_Yield"]

# 2Y Treasury yield
us2y = pdr.DataReader("DGS2", "fred", start_date, end_date)
us2y.columns = ["US2Y_Yield"]

# CPI (monthly index level)
cpi = pdr.DataReader("CPIAUCSL", "fred", start_date, end_date)
cpi.columns = ["CPI"]

# =========================
# 4. COMPUTE MACRO TRANSFORMATIONS CORRECTLY
# =========================

# 10Y yield changes
us10y["US10Y_Yield"] = us10y["US10Y_Yield"].ffill()
us10y["US10Y_Change_pctpts"] = us10y["US10Y_Yield"].diff()
us10y["US10Y_Change_bps"] = us10y["US10Y_Change_pctpts"] * 100

# 3M yield changes
us3m["US3M_Yield"] = us3m["US3M_Yield"].ffill()
us3m["US3M_Change_pctpts"] = us3m["US3M_Yield"].diff()
us3m["US3M_Change_bps"] = us3m["US3M_Change_pctpts"] * 100

# 2Y yield changes
us2y["US2Y_Yield"] = us2y["US2Y_Yield"].ffill()
us2y["US2Y_Change_pctpts"] = us2y["US2Y_Yield"].diff()
us2y["US2Y_Change_bps"] = us2y["US2Y_Change_pctpts"] * 100

# CPI YoY inflation
# IMPORTANT: compute on monthly CPI first, BEFORE any forward-filling to daily
cpi["CPI_YoY"] = cpi["CPI"].pct_change(12)

# Combine macro data
macro_df = pd.concat([us10y, us3m, us2y, cpi], axis=1)

# =========================
# 5. MERGE ETF + MACRO DATA
# =========================
df = pd.concat([adj_close, etf_returns, macro_df], axis=1)

# Forward-fill macro variables after merging to align monthly/daily frequencies
macro_cols = [
    "US10Y_Yield",
    "US10Y_Change_pctpts",
    "US10Y_Change_bps",
    "US3M_Yield",
    "US3M_Change_pctpts",
    "US3M_Change_bps",
    "US2Y_Yield",
    "US2Y_Change_pctpts",
    "US2Y_Change_bps",
    "CPI",
    "CPI_YoY"
]
df[macro_cols] = df[macro_cols].ffill()

# Drop rows with missing values from initial return/CPI calculations
df = df.dropna().copy()

# 10Y minus 3M yield spread
df["US10Y_3M_Spread"] = df["US10Y_Yield"] - df["US3M_Yield"]

# Daily change in 10Y-3M spread in bps
df["US10Y_3M_Spread_Change_bps"] = df["US10Y_3M_Spread"].diff() * 100

# 10Y minus 2Y yield spread
df["US10Y_2Y_Spread"] = df["US10Y_Yield"] - df["US2Y_Yield"]

# Daily change in 10Y-2Y spread in bps
df["US10Y_2Y_Spread_Change_bps"] = df["US10Y_2Y_Spread"].diff() * 100

# =========================
# 6. ADD RELATIVE RETURN SERIES
# =========================
df["TLT_minus_IEF_Ret"] = df["TLT_Ret"] - df["IEF_Ret"]
df["TLT_minus_SHY_Ret"] = df["TLT_Ret"] - df["SHY_Ret"]
df["IEF_minus_SHY_Ret"] = df["IEF_Ret"] - df["SHY_Ret"]

# =========================
# 7. SAVE DATASET
# =========================
df.to_csv("fixed_income_macro_dataset.csv")

# =========================
# 0. SETUP
# =========================
os.makedirs("figures", exist_ok=True)

# Consistent colours
COLORS = {
    "TLT": "#1f77b4",       # blue
    "IEF": "#ff7f0e",       # orange
    "SHY": "#2ca02c",       # green
    "US10Y": "#d62728",     # red
    "CPI": "#9467bd",       # purple
    "Signal": "#4c72b0"
}

FIG_DPI = 300

BASE_PATH = Path("/Users/jackadams/Documents/Quant Research Preparation/Tests/Macro Project")
FIG_PATH = BASE_PATH / "figures"
FIG_PATH.mkdir(exist_ok=True)
TABLE_PATH = BASE_PATH / "tables"
TABLE_PATH.mkdir(exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_PATH / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.show()

# =========================
# 1. US 10Y YIELD OVER TIME
# =========================
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["US10Y_Yield"], color=COLORS["US10Y"], linewidth=1.5)
plt.title("US 10Y Treasury Yield Over Time")
plt.xlabel("Date")
plt.ylabel("Yield (%)")
plt.grid(alpha=0.3)
savefig("us10y_yield_over_time")

# =========================
# 2. ETF PRICES OVER TIME
# =========================
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["TLT_AdjClose"], label="TLT", color=COLORS["TLT"], linewidth=1.5)
plt.plot(df.index, df["IEF_AdjClose"], label="IEF", color=COLORS["IEF"], linewidth=1.5)
plt.plot(df.index, df["SHY_AdjClose"], label="SHY", color=COLORS["SHY"], linewidth=1.5)
plt.title("Treasury ETF Prices by Duration")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
savefig("etf_prices_duration_comparison")

# =========================
# 3. TLT PRICE VS 10Y YIELD
# =========================
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(df.index, df["TLT_AdjClose"], color=COLORS["TLT"], linewidth=1.5)
ax1.set_xlabel("Date")
ax1.set_ylabel("TLT Adjusted Close Price (USD)", color=COLORS["TLT"])
ax1.tick_params(axis='y', labelcolor=COLORS["TLT"])
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(df.index, df["US10Y_Yield"], color=COLORS["US10Y"], linewidth=1.5)
ax2.set_ylabel("US 10Y Treasury Yield (%)", color=COLORS["US10Y"])
ax2.tick_params(axis='y', labelcolor=COLORS["US10Y"])

plt.title("TLT Price and US 10Y Treasury Yield")
savefig("tlt_vs_us10y_yield")

# =========================
# 4. YIELD CHANGE VS TLT RETURN
# =========================
plt.figure(figsize=(7, 6))
plt.scatter(
    df["US10Y_Change_bps"],
    df["TLT_Ret"] * 100,
    alpha=0.25,
    color=COLORS["TLT"],
    edgecolors="none"
)
plt.axhline(0, color="black", linewidth=0.8, alpha=0.6)
plt.axvline(0, color="black", linewidth=0.8, alpha=0.6)
plt.title("Daily TLT Return vs Change in US 10Y Yield")
plt.xlabel("Daily Change in US 10Y Yield (bps)")
plt.ylabel("Daily TLT Return (%)")
plt.grid(alpha=0.3)
savefig("yield_change_vs_tlt_return")

# =========================
# 5. YIELD SENSITIVITY BY DURATION
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(df["US10Y_Change_bps"], df["TLT_Ret"] * 100, alpha=0.20, label="TLT", color=COLORS["TLT"])
plt.scatter(df["US10Y_Change_bps"], df["IEF_Ret"] * 100, alpha=0.20, label="IEF", color=COLORS["IEF"])
plt.scatter(df["US10Y_Change_bps"], df["SHY_Ret"] * 100, alpha=0.20, label="SHY", color=COLORS["SHY"])
plt.axhline(0, color="black", linewidth=0.8, alpha=0.6)
plt.axvline(0, color="black", linewidth=0.8, alpha=0.6)
plt.title("Daily ETF Return Sensitivity to Changes in US 10Y Yield")
plt.xlabel("Daily Change in US 10Y Yield (bps)")
plt.ylabel("Daily ETF Return (%)")
plt.legend()
plt.grid(alpha=0.3)
savefig("yield_sensitivity_by_duration")

# =========================
# 6. CPI YOY VS US 10Y YIELD OVER TIME
# =========================
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(df.index, df["CPI_YoY"] * 100, color=COLORS["CPI"], linewidth=1.5)
ax1.set_xlabel("Date")
ax1.set_ylabel("CPI Inflation YoY (%)", color=COLORS["CPI"])
ax1.tick_params(axis='y', labelcolor=COLORS["CPI"])
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(df.index, df["US10Y_Yield"], color=COLORS["US10Y"], linewidth=1.5)
ax2.set_ylabel("US 10Y Treasury Yield (%)", color=COLORS["US10Y"])
ax2.tick_params(axis='y', labelcolor=COLORS["US10Y"])

plt.title("US CPI Inflation and US 10Y Treasury Yield")
savefig("cpi_yoy_vs_us10y_yield")

# =====================================================
# SIGNAL MOTIVATION AND PREDICTIVE ANALYSIS
# =====================================================

# =========================
# 1. CREATE FORWARD RETURNS
# =========================

eda = df.copy()

# Forward simple returns from time t to t+h
eda["TLT_Fwd_1D_Ret"] = eda["TLT_AdjClose"].shift(-1) / eda["TLT_AdjClose"] - 1
eda["TLT_Fwd_5D_Ret"] = eda["TLT_AdjClose"].shift(-5) / eda["TLT_AdjClose"] - 1
eda["TLT_Fwd_10D_Ret"] = eda["TLT_AdjClose"].shift(-10) / eda["TLT_AdjClose"] - 1
# Forward returns for SHY (needed for spread signal)
eda["SHY_Fwd_10D_Ret"] = (
    eda["SHY_AdjClose"].shift(-10) / eda["SHY_AdjClose"] - 1
)


# =========================
# 2. CREATE SIGNAL CANDIDATES
# =========================

# Daily change in 10Y yield
eda["Signal_Daily_Change"] = eda["US10Y_Change_bps"]

# Multi-day yield changes
eda["Signal_5D_Yield_Change"] = (eda["US10Y_Yield"] - eda["US10Y_Yield"].shift(5)) * 100
eda["Signal_10D_Yield_Change"] = (eda["US10Y_Yield"] - eda["US10Y_Yield"].shift(10)) * 100

# Smoothed daily yield change
eda["Signal_10D_MA_Change"] = eda["US10Y_Change_bps"].rolling(10).mean()


# =========================
# 3. HELPER FUNCTION FOR TESTS
# =========================

def signal_test(data, signal_col, return_col, signal_name):
    """
    Runs simple predictive tests:
    - correlation
    - OLS beta
    - t-stat
    - p-value
    - R-squared
    - directional hit rate

    Regression:
    future TLT return (%) = alpha + beta * signal (bps) + error
    """

    temp = data[[signal_col, return_col]].dropna().copy()

    x = temp[signal_col].values
    y = temp[return_col].values * 100  # convert return to %

    # OLS regression using scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Correlation
    corr = np.corrcoef(x, y)[0, 1]

    # R-squared
    r_squared = r_value ** 2

    # t-stat for slope
    t_stat = slope / std_err if std_err != 0 else np.nan

    # Directional rule:
    # If signal > 0, yields rising -> expect TLT return < 0
    # If signal < 0, yields falling -> expect TLT return > 0
    predicted_direction = -np.sign(x)
    realised_direction = np.sign(y)

    valid = (predicted_direction != 0) & (realised_direction != 0)
    hit_rate = np.mean(predicted_direction[valid] == realised_direction[valid])

    return {
        "Signal": signal_name,
        "Observations": len(temp),
        "Correlation": corr,
        "Beta (pct return per bp)": slope,
        "t-stat": t_stat,
        "p-value": p_value,
        "R-squared": r_squared,
        "Directional Hit Rate": hit_rate
    }


# =========================
# 4. RUN TESTS
# =========================

results = []

results.append(
    signal_test(
        eda,
        "Signal_Daily_Change",
        "TLT_Fwd_1D_Ret",
        "Daily yield change vs next 1D TLT return"
    )
)

results.append(
    signal_test(
        eda,
        "Signal_5D_Yield_Change",
        "TLT_Fwd_5D_Ret",
        "Past 5D yield change vs next 5D TLT return"
    )
)

results.append(
    signal_test(
        eda,
        "Signal_10D_Yield_Change",
        "TLT_Fwd_10D_Ret",
        "Past 10D yield change vs next 10D TLT return"
    )
)

results.append(
    signal_test(
        eda,
        "Signal_10D_MA_Change",
        "TLT_Fwd_5D_Ret",
        "10D MA yield change vs next 5D TLT return"
    )
)

signal_results = pd.DataFrame(results)

# Round for display
signal_results_rounded = signal_results.copy()
for col in ["Correlation", "Beta (pct return per bp)", "t-stat", "p-value", "R-squared", "Directional Hit Rate"]:
    signal_results_rounded[col] = signal_results_rounded[col].round(4)

print(signal_results_rounded)


# =========================
# 5. SAVE CLEAN LATEX TABLE
# =========================

signal_table = signal_results.copy()

# Shorter, cleaner signal names so the table fits better
signal_table["Signal"] = [
    "Daily yield change / next 1D TLT return",
    "Past 5D yield change / next 5D TLT return",
    "Past 10D yield change / next 10D TLT return",
    "10D MA yield change / next 5D TLT return"
]

# Rename columns professionally
signal_table = signal_table.rename(columns={
    "Signal": "Signal specification",
    "Observations": "Obs.",
    "Correlation": "Correlation",
    "Beta (pct return per bp)": r"Beta (\% return per bp)",
    "t-stat": r"$t$-statistic",
    "p-value": r"$p$-value",
    "R-squared": r"$R^2$",
    "Directional Hit Rate": "Directional hit rate"
})

# Round numerical columns
round_cols = [
    "Correlation",
    r"Beta (\% return per bp)",
    r"$t$-statistic",
    r"$p$-value",
    r"$R^2$",
    "Directional hit rate"
]

for col in round_cols:
    signal_table[col] = signal_table[col].round(4)

# Save CSV version
signal_table.to_csv(TABLE_PATH / "signal_predictive_tests.csv", index=False)

# Save raw LaTeX tabular only: no caption, no label
latex_table = signal_table.to_latex(
    index=False,
    escape=False,
    float_format="%.4f",
    column_format="lrrrrrrr"
)

with open(TABLE_PATH / "signal_predictive_tests.tex", "w") as f:
    f.write(latex_table)

print(signal_table)


# =========================
# 6. PLOT 1: 10Y YIELD WITH MOVING AVERAGES
# =========================

eda["US10Y_MA_20"] = eda["US10Y_Yield"].rolling(20).mean()
eda["US10Y_MA_60"] = eda["US10Y_Yield"].rolling(60).mean()

plt.figure(figsize=(10, 5))
plt.plot(eda.index, eda["US10Y_Yield"], label="US 10Y Yield", color=COLORS["US10Y"], linewidth=1.2)
plt.plot(eda.index, eda["US10Y_MA_20"], label="20D MA", color="#1f77b4", linewidth=1.2)
plt.plot(eda.index, eda["US10Y_MA_60"], label="60D MA", color="#2ca02c", linewidth=1.2)

plt.title("US 10Y Treasury Yield with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Yield (%)")
plt.legend()
plt.grid(alpha=0.3)

savefig("us10y_yield_moving_averages")


# =========================
# 7. PLOT 2: PAST 5D YIELD CHANGE VS NEXT 5D TLT RETURN
# =========================

plot_data = eda[["Signal_5D_Yield_Change", "TLT_Fwd_5D_Ret"]].dropna()

plt.figure(figsize=(7, 6))
plt.scatter(
    plot_data["Signal_5D_Yield_Change"],
    plot_data["TLT_Fwd_5D_Ret"] * 100,
    alpha=0.25,
    color=COLORS["TLT"],
    edgecolors="none"
)

# Regression line
x = plot_data["Signal_5D_Yield_Change"].values
y = plot_data["TLT_Fwd_5D_Ret"].values * 100
slope, intercept, *_ = stats.linregress(x, y)

x_line = np.linspace(x.min(), x.max(), 100)
y_line = intercept + slope * x_line
plt.plot(x_line, y_line, color="black", linewidth=1.2, label="Regression line")

plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
plt.axvline(0, color="black", linewidth=0.8, alpha=0.5)

plt.title("Past 5D Yield Change vs Next 5D TLT Return")
plt.xlabel("Past 5D Change in US 10Y Yield (bps)")
plt.ylabel("Next 5D TLT Return (%)")
plt.legend()
plt.grid(alpha=0.3)

savefig("past5d_yield_change_vs_next5d_tlt_return")


# =========================
# 8. PLOT 3: PAST 10D YIELD CHANGE VS NEXT 10D TLT RETURN
# =========================

plot_data = eda[["Signal_10D_Yield_Change", "TLT_Fwd_10D_Ret"]].dropna()

plt.figure(figsize=(7, 6))
plt.scatter(
    plot_data["Signal_10D_Yield_Change"],
    plot_data["TLT_Fwd_10D_Ret"] * 100,
    alpha=0.25,
    color=COLORS["TLT"],
    edgecolors="none"
)

x = plot_data["Signal_10D_Yield_Change"].values
y = plot_data["TLT_Fwd_10D_Ret"].values * 100
slope, intercept, *_ = stats.linregress(x, y)

x_line = np.linspace(x.min(), x.max(), 100)
y_line = intercept + slope * x_line
plt.plot(x_line, y_line, color="black", linewidth=1.2, label="Regression line")

plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
plt.axvline(0, color="black", linewidth=0.8, alpha=0.5)

plt.title("Past 10D Yield Change vs Next 10D TLT Return")
plt.xlabel("Past 10D Change in US 10Y Yield (bps)")
plt.ylabel("Next 10D TLT Return (%)")
plt.legend()
plt.grid(alpha=0.3)

savefig("past10d_yield_change_vs_next10d_tlt_return")


# =========================
# 9. PLOT 4: SIGNAL CANDIDATES OVER TIME
# =========================

plt.figure(figsize=(10, 5))
plt.plot(
    eda.index,
    eda["Signal_5D_Yield_Change"],
    label="5D Yield Change",
    color="#1f77b4",
    linewidth=1.0,
    alpha=0.8
)
plt.plot(
    eda.index,
    eda["Signal_10D_Yield_Change"],
    label="10D Yield Change",
    color="#ff7f0e",
    linewidth=1.0,
    alpha=0.8
)
plt.plot(
    eda.index,
    eda["Signal_10D_MA_Change"],
    label="10D MA Daily Yield Change",
    color="#2ca02c",
    linewidth=1.0,
    alpha=0.8
)

plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)

plt.title("Yield-Based Signal Candidates")
plt.xlabel("Date")
plt.ylabel("Yield Signal (bps)")
plt.legend()
plt.grid(alpha=0.3)

savefig("yield_signal_candidates")

# =========================
# 10. PLOT 5: AUTOCORRELATION OF DAILY YIELD CHANGES
# =========================

max_lag = 20
autocorr_values = [
    eda["US10Y_Change_bps"].autocorr(lag=lag)
    for lag in range(1, max_lag + 1)
]

plt.figure(figsize=(8, 5))
plt.bar(range(1, max_lag + 1), autocorr_values, color=COLORS["US10Y"], alpha=0.75)
plt.axhline(0, color="black", linewidth=0.8)

plt.title("Autocorrelation of Daily US 10Y Yield Changes")
plt.xlabel("Lag (days)")
plt.ylabel("Autocorrelation")
plt.grid(axis="y", alpha=0.3)

savefig("autocorrelation_daily_yield_changes")

# =====================================================
# CONDITIONAL RETURN AND THRESHOLD ANALYSIS
# =====================================================

plot_data = eda[["Signal_10D_Yield_Change", "TLT_Fwd_10D_Ret"]].dropna().copy()

signal_col = "Signal_10D_Yield_Change"
return_col = "TLT_Fwd_10D_Ret"

# Convert future return to %
plot_data["Fwd_Return_%"] = plot_data[return_col] * 100

# =========================
# 1. CONDITIONAL RETURN BY DECILE
# =========================

plot_data["Signal_Decile"] = pd.qcut(
    plot_data[signal_col],
    q=10,
    labels=[f"D{i}" for i in range(1, 11)]
)

decile_stats = (
    plot_data
    .groupby("Signal_Decile")
    .agg(
        mean_signal_bps=(signal_col, "mean"),
        min_signal_bps=(signal_col, "min"),
        max_signal_bps=(signal_col, "max"),
        mean_fwd_return_pct=("Fwd_Return_%", "mean"),
        std_fwd_return_pct=("Fwd_Return_%", "std"),
        count=("Fwd_Return_%", "count")
    )
)

# 95% confidence interval for mean return
decile_stats["se"] = decile_stats["std_fwd_return_pct"] / np.sqrt(decile_stats["count"])
decile_stats["ci95"] = 1.96 * decile_stats["se"]

# Cleaner x-axis labels
decile_stats["label"] = decile_stats.apply(
    lambda row: f"{row['mean_signal_bps']:.1f} bps\nn={int(row['count'])}",
    axis=1
)

plt.figure(figsize=(10, 5.5))
plt.bar(
    decile_stats["label"],
    decile_stats["mean_fwd_return_pct"],
    yerr=decile_stats["ci95"],
    capsize=4,
    alpha=0.85
)

plt.axhline(0, color="black", linewidth=1)
plt.title("Average Future TLT Return by Past 10D Yield-Change Decile")
plt.xlabel("Average past 10D change in US 10Y yield by decile")
plt.ylabel("Average next 10D TLT return (%)")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)

savefig("conditional_return_by_signal_decile_ci")


# =========================
# 2. EXTREME VS NON-EXTREME COMPARISON
# =========================

threshold = 25  # bps; change later after looking at threshold analysis

plot_data["Extreme_Group"] = np.select(
    [
        plot_data[signal_col] <= -threshold,
        plot_data[signal_col] >= threshold
    ],
    [
        f"Large yield fall\n≤ -{threshold} bps",
        f"Large yield rise\n≥ +{threshold} bps"
    ],
    default=f"Moderate move\nbetween ±{threshold} bps"
)

group_order = [
    f"Large yield fall\n≤ -{threshold} bps",
    f"Moderate move\nbetween ±{threshold} bps",
    f"Large yield rise\n≥ +{threshold} bps"
]

group_stats = (
    plot_data
    .groupby("Extreme_Group")
    .agg(
        mean_signal_bps=(signal_col, "mean"),
        mean_fwd_return_pct=("Fwd_Return_%", "mean"),
        std_fwd_return_pct=("Fwd_Return_%", "std"),
        count=("Fwd_Return_%", "count")
    )
    .reindex(group_order)
)

group_stats["se"] = group_stats["std_fwd_return_pct"] / np.sqrt(group_stats["count"])
group_stats["ci95"] = 1.96 * group_stats["se"]

plt.figure(figsize=(8, 5))
plt.bar(
    group_stats.index,
    group_stats["mean_fwd_return_pct"],
    yerr=group_stats["ci95"],
    capsize=5,
    alpha=0.85
)

plt.axhline(0, color="black", linewidth=1)
plt.title(f"Future TLT Returns After Extreme 10D Yield Moves\nThreshold = {threshold} bps")
plt.xlabel("Past 10D yield-change regime")
plt.ylabel("Average next 10D TLT return (%)")
plt.grid(axis="y", alpha=0.3)

savefig("extreme_vs_moderate_yield_moves")


# =========================
# 3. HIT RATE AND TRADE FREQUENCY BY THRESHOLD
# =========================

thresholds = np.arange(5, 55, 5)
threshold_results = []

for th in thresholds:
    signal = np.where(
        plot_data[signal_col] > th, -1,
        np.where(plot_data[signal_col] < -th, 1, 0)
    )

    actual = np.sign(plot_data["Fwd_Return_%"])
    valid = signal != 0

    n_trades = valid.sum()
    trade_freq = n_trades / len(plot_data)

    if n_trades > 0:
        hit_rate = np.mean(signal[valid] == actual[valid])
        avg_return_when_active = np.mean(signal[valid] * plot_data.loc[valid, "Fwd_Return_%"])
    else:
        hit_rate = np.nan
        avg_return_when_active = np.nan

    threshold_results.append({
        "Threshold_bps": th,
        "Hit_Rate": hit_rate,
        "Trade_Frequency": trade_freq,
        "Number_of_Trades": n_trades,
        "Avg_Strategy_Return_When_Active_%": avg_return_when_active
    })

threshold_stats = pd.DataFrame(threshold_results)

fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(
    threshold_stats["Threshold_bps"],
    threshold_stats["Hit_Rate"],
    marker="o",
    label="Hit rate"
)
ax1.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)
ax1.set_xlabel("Absolute 10D yield-change threshold (bps)")
ax1.set_ylabel("Directional hit rate")
ax1.set_ylim(0.45, 0.60)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(
    threshold_stats["Threshold_bps"],
    threshold_stats["Trade_Frequency"],
    marker="s",
    linestyle="--",
    label="Trade frequency"
)
ax2.set_ylabel("Fraction of observations traded")

plt.title("Signal Accuracy Improves as Yield-Move Threshold Increases")
fig.tight_layout()

savefig("hit_rate_and_trade_frequency_vs_threshold")


# =========================
# 4. AVERAGE STRATEGY RETURN BY THRESHOLD
# =========================

plt.figure(figsize=(9, 5))
plt.plot(
    threshold_stats["Threshold_bps"],
    threshold_stats["Avg_Strategy_Return_When_Active_%"],
    marker="o"
)

plt.axhline(0, color="black", linewidth=1)
plt.title("Average Active Strategy Return by Threshold")
plt.xlabel("Absolute 10D yield-change threshold (bps)")
plt.ylabel("Average signed next 10D return when active (%)")
plt.grid(alpha=0.3)

savefig("average_active_return_vs_threshold")

# =========================
# 5. SAVE LATEX-COMPATIBLE TABLES
# =========================

def save_latex_table(df, path, column_format=None):
    latex_table = df.to_latex(
        index=False,
        escape=False,
        float_format="%.4f",
        column_format=column_format
    )

    with open(path, "w") as f:
        f.write(latex_table)


# Decile conditional return table
decile_table = decile_stats.reset_index().copy()

decile_table = decile_table[
    [
        "Signal_Decile",
        "mean_signal_bps",
        "min_signal_bps",
        "max_signal_bps",
        "mean_fwd_return_pct",
        "std_fwd_return_pct",
        "count",
        "ci95"
    ]
]

decile_table = decile_table.rename(columns={
    "Signal_Decile": "Signal decile",
    "mean_signal_bps": "Mean signal (bps)",
    "min_signal_bps": "Minimum signal (bps)",
    "max_signal_bps": "Maximum signal (bps)",
    "mean_fwd_return_pct": "Mean forward return (\\%)",
    "std_fwd_return_pct": "Standard deviation (\\%)",
    "count": "Observations",
    "ci95": "95\\% confidence interval"
})

save_latex_table(
    decile_table,
    TABLE_PATH / "conditional_return_by_decile.tex",
    column_format="lrrrrrrr"
)


# Threshold signal analysis table
threshold_table = threshold_stats.copy()

threshold_table = threshold_table.rename(columns={
    "Threshold_bps": "Threshold (bps)",
    "Hit_Rate": "Hit rate",
    "Trade_Frequency": "Trade frequency",
    "Number_of_Trades": "Number of trades",
    "Avg_Strategy_Return_When_Active_%": "Average active return (\\%)"
})

save_latex_table(
    threshold_table,
    TABLE_PATH / "threshold_signal_analysis.tex",
    column_format="rrrrr"
)

print("\nDecile conditional return stats:")
print(decile_table.round(4))

print("\nThreshold signal analysis:")
print(threshold_table.round(4))


# =====================================================
# THRESHOLD OBJECTIVE: AVG RETURN × NUMBER OF TRADES
# =====================================================

thresholds = np.arange(5, 55, 5)
objective_results = []

for th in thresholds:
    signal = np.where(
        plot_data["Signal_10D_Yield_Change"] > th, -1,
        np.where(plot_data["Signal_10D_Yield_Change"] < -th, 1, 0)
    )

    valid = signal != 0
    n_trades = valid.sum()

    if n_trades > 0:
        signed_returns = signal[valid] * plot_data.loc[valid, "Fwd_Return_%"]

        avg_active_return = signed_returns.mean()
        total_return_contribution = signed_returns.sum()

        objective_results.append({
            "Threshold_bps": th,
            "Number_of_Trades": n_trades,
            "Avg_Active_Return_%": avg_active_return,
            "Objective_AvgReturn_x_Trades": avg_active_return * n_trades,
            "Total_Signed_Return_%": total_return_contribution
        })

objective_df = pd.DataFrame(objective_results)

best_row = objective_df.loc[
    objective_df["Objective_AvgReturn_x_Trades"].idxmax()
]

print(objective_df.round(4))
print("\nBest threshold based on avg return × number of trades:")
print(best_row.round(4))

plt.figure(figsize=(9, 5))
plt.plot(
    objective_df["Threshold_bps"],
    objective_df["Objective_AvgReturn_x_Trades"],
    marker="o"
)

best_threshold = best_row["Threshold_bps"]
best_objective = best_row["Objective_AvgReturn_x_Trades"]

plt.axvline(best_threshold, color="black", linestyle="--", linewidth=1)
plt.scatter(best_threshold, best_objective, s=80, zorder=5)

plt.title("Threshold Objective: Average Active Return × Number of Trades")
plt.xlabel("Absolute 10D yield-change threshold (bps)")
plt.ylabel("Objective value")
plt.grid(alpha=0.3)

savefig("threshold_objective_avg_return_times_trades")

objective_table = objective_df.copy()

objective_table = objective_table.rename(columns={
    "Threshold_bps": "Threshold (bps)",
    "Number_of_Trades": "Number of trades",
    "Avg_Active_Return_%": "Average active return (\\%)",
    "Objective_AvgReturn_x_Trades": "Objective value",
    "Total_Signed_Return_%": "Total signed return (\\%)"
})

save_latex_table(
    objective_table,
    TABLE_PATH / "threshold_objective_avg_return_times_trades.tex",
    column_format="rrrrr"
)

# =====================================================
# ROBUST THRESHOLD OBJECTIVE:
# ACTIVE SHARPE × SQRT(NUMBER OF TRADES)
# =====================================================

robust_results = []

for th in thresholds:
    signal = np.where(
        plot_data["Signal_10D_Yield_Change"] > th, -1,
        np.where(plot_data["Signal_10D_Yield_Change"] < -th, 1, 0)
    )

    valid = signal != 0
    n_trades = valid.sum()

    if n_trades > 1:
        signed_returns = signal[valid] * plot_data.loc[valid, "Fwd_Return_%"]

        mean_ret = signed_returns.mean()
        std_ret = signed_returns.std()

        active_sharpe = mean_ret / std_ret if std_ret != 0 else np.nan
        robust_objective = active_sharpe * np.sqrt(n_trades)

        robust_results.append({
            "Threshold_bps": th,
            "Number_of_Trades": n_trades,
            "Mean_Active_Return_%": mean_ret,
            "Std_Active_Return_%": std_ret,
            "Active_Sharpe": active_sharpe,
            "Robust_Objective": robust_objective
        })

robust_df = pd.DataFrame(robust_results)

best_robust_row = robust_df.loc[
    robust_df["Robust_Objective"].idxmax()
]

print(robust_df.round(4))
print("\nBest threshold based on active Sharpe × sqrt(number of trades):")
print(best_robust_row.round(4))


robust_table = robust_df.copy()

robust_table = robust_table.rename(columns={
    "Threshold_bps": "Threshold (bps)",
    "Number_of_Trades": "Number of trades",
    "Mean_Active_Return_%": "Mean active return (\\%)",
    "Std_Active_Return_%": "Standard deviation (\\%)",
    "Active_Sharpe": "Active Sharpe ratio",
    "Robust_Objective": "Robust objective"
})

save_latex_table(
    robust_table,
    TABLE_PATH / "robust_threshold_objective.tex",
    column_format="rrrrrr"
)

# =========================
# PLOT ROBUST OBJECTIVE
# =========================

plt.figure(figsize=(9, 5))

plt.plot(
    robust_df["Threshold_bps"],
    robust_df["Robust_Objective"],
    marker="o",
    linewidth=1.8
)

best_threshold = best_robust_row["Threshold_bps"]
best_objective = best_robust_row["Robust_Objective"]

plt.axvline(
    best_threshold,
    color="black",
    linestyle="--",
    linewidth=1,
    label=f"Best threshold = {best_threshold:.0f} bps"
)

plt.scatter(
    best_threshold,
    best_objective,
    s=80,
    zorder=5
)

plt.title("Robust Threshold Objective: Active Sharpe × √Number of Trades")
plt.xlabel("Absolute 10D yield-change threshold (bps)")
plt.ylabel("Robust objective value")
plt.legend()
plt.grid(alpha=0.3)

savefig("robust_threshold_objective_active_sharpe")


# =========================
# PLOT ACTIVE SHARPE ONLY
# =========================

plt.figure(figsize=(9, 5))

plt.plot(
    robust_df["Threshold_bps"],
    robust_df["Active_Sharpe"],
    marker="o",
    linewidth=1.8
)

plt.axhline(0, color="black", linewidth=1)

plt.axvline(
    best_threshold,
    color="black",
    linestyle="--",
    linewidth=1,
    label=f"Best robust threshold = {best_threshold:.0f} bps"
)

plt.title("Active Sharpe Ratio by Yield-Move Threshold")
plt.xlabel("Absolute 10D yield-change threshold (bps)")
plt.ylabel("Mean active return / standard deviation")
plt.legend()
plt.grid(alpha=0.3)

savefig("active_sharpe_by_threshold")

# =====================================================
# MOVING-AVERAGE YIELD TREND SIGNAL
# 20D MA - 60D MA AS SIMPLE ALTERNATIVE SIGNAL
# =====================================================

# =========================
# 1. CREATE MA TREND SIGNAL
# =========================

ma_data = eda.copy()

ma_data["US10Y_MA_20"] = ma_data["US10Y_Yield"].rolling(20).mean()
ma_data["US10Y_MA_60"] = ma_data["US10Y_Yield"].rolling(60).mean()

# Positive = yields trending upward
# Negative = yields trending downward
ma_data["Yield_MA_Trend"] = ma_data["US10Y_MA_20"] - ma_data["US10Y_MA_60"]

# Future 10D TLT return in %
ma_data["Fwd_Return_%"] = ma_data["TLT_Fwd_10D_Ret"] * 100

ma_test = ma_data[
    ["Yield_MA_Trend", "Fwd_Return_%"]
].dropna().copy()


# =========================
# 2. DEFINE TRADING SIGNAL
# =========================

# If yields are trending up, short TLT
# If yields are trending down, long TLT
ma_test["Position"] = np.where(
    ma_test["Yield_MA_Trend"] > 0, -1,
    np.where(ma_test["Yield_MA_Trend"] < 0, 1, 0)
)

ma_test["Signed_Return_%"] = ma_test["Position"] * ma_test["Fwd_Return_%"]

valid = ma_test["Position"] != 0

predicted_direction = ma_test.loc[valid, "Position"]
realised_direction = np.sign(ma_test.loc[valid, "Fwd_Return_%"])

non_zero_realised = realised_direction != 0

hit_rate = np.mean(
    predicted_direction[non_zero_realised] == realised_direction[non_zero_realised]
)

n_trades = valid.sum()
trade_frequency = n_trades / len(ma_test)

mean_active_return = ma_test.loc[valid, "Signed_Return_%"].mean()
std_active_return = ma_test.loc[valid, "Signed_Return_%"].std()
active_sharpe = mean_active_return / std_active_return

correlation = ma_test["Yield_MA_Trend"].corr(ma_test["Fwd_Return_%"])


# =========================
# 3. SAVE SUMMARY TABLE
# =========================

ma_summary = pd.DataFrame({
    "Signal": ["20D minus 60D yield moving average"],
    "Observations": [len(ma_test)],
    "Number of trades": [n_trades],
    "Trade frequency": [trade_frequency],
    "Correlation": [correlation],
    "Directional hit rate": [hit_rate],
    "Mean active return (\\%)": [mean_active_return],
    "Standard deviation (\\%)": [std_active_return],
    "Active Sharpe ratio": [active_sharpe]
})

round_cols = [
    "Trade frequency",
    "Correlation",
    "Directional hit rate",
    "Mean active return (\\%)",
    "Standard deviation (\\%)",
    "Active Sharpe ratio"
]

for col in round_cols:
    ma_summary[col] = ma_summary[col].round(4)

ma_summary.to_csv(TABLE_PATH / "ma_trend_signal_summary.csv", index=False)

latex_table = ma_summary.to_latex(
    index=False,
    escape=False,
    float_format="%.4f",
    column_format="lrrrrrrrr"
)

with open(TABLE_PATH / "ma_trend_signal_summary.tex", "w") as f:
    f.write(latex_table)

print("\nMoving-average trend signal summary:")
print(ma_summary)


# =========================
# 4. PLOT MA TREND SIGNAL THROUGH TIME
# =========================

plt.figure(figsize=(10, 5))

plt.plot(
    ma_data.index,
    ma_data["Yield_MA_Trend"],
    linewidth=1.2,
    label="20D MA - 60D MA"
)

plt.axhline(0, color="black", linewidth=1)

plt.title("Yield Moving-Average Trend Signal")
plt.xlabel("Date")
plt.ylabel("20D minus 60D US 10Y yield moving average")
plt.legend()
plt.grid(alpha=0.3)

savefig("ma_trend_signal")


# =========================
# 5. PLOT AVERAGE RETURN BY MA TREND REGIME
# =========================

ma_test["Trend_Regime"] = np.where(
    ma_test["Yield_MA_Trend"] > 0,
    "Upward yield trend\nShort TLT signal",
    "Downward yield trend\nLong TLT signal"
)

ma_regime_stats = (
    ma_test
    .groupby("Trend_Regime")
    .agg(
        mean_fwd_return_pct=("Fwd_Return_%", "mean"),
        std_fwd_return_pct=("Fwd_Return_%", "std"),
        count=("Fwd_Return_%", "count")
    )
)

ma_regime_stats["se"] = (
    ma_regime_stats["std_fwd_return_pct"] / np.sqrt(ma_regime_stats["count"])
)

ma_regime_stats["ci95"] = 1.96 * ma_regime_stats["se"]

plt.figure(figsize=(7, 5))

plt.bar(
    ma_regime_stats.index,
    ma_regime_stats["mean_fwd_return_pct"],
    yerr=ma_regime_stats["ci95"],
    capsize=5,
    alpha=0.85
)

plt.axhline(0, color="black", linewidth=1)

plt.title("Future TLT Returns by Yield Trend Regime")
plt.xlabel("Yield trend regime")
plt.ylabel("Average next 10D TLT return (%)")
plt.grid(axis="y", alpha=0.3)

savefig("future_returns_by_ma_trend_regime")

# =====================================================
# COMBINED SIGNAL ANALYSIS:
# 25BPS 10D YIELD-CHANGE THRESHOLD + MA TREND FILTER
# =====================================================

combined_data = eda.copy()

# Ensure required variables exist
combined_data["Signal_10D_Yield_Change"] = (
    combined_data["US10Y_Yield"] - combined_data["US10Y_Yield"].shift(10)
) * 100

combined_data["US10Y_MA_20"] = combined_data["US10Y_Yield"].rolling(20).mean()
combined_data["US10Y_MA_60"] = combined_data["US10Y_Yield"].rolling(60).mean()
combined_data["Yield_MA_Trend"] = combined_data["US10Y_MA_20"] - combined_data["US10Y_MA_60"]

combined_data["Fwd_Return_%"] = combined_data["TLT_Fwd_10D_Ret"] * 100

combined_test = combined_data[
    ["Signal_10D_Yield_Change", "Yield_MA_Trend", "Fwd_Return_%"]
].dropna().copy()

threshold = 25


# =========================
# 1. DEFINE STRATEGY SIGNALS
# =========================

# Baseline threshold signal
combined_test["Baseline_Position"] = np.where(
    combined_test["Signal_10D_Yield_Change"] <= -threshold, 1,
    np.where(combined_test["Signal_10D_Yield_Change"] >= threshold, -1, 0)
)

# MA trend signal alone
combined_test["MA_Position"] = np.where(
    combined_test["Yield_MA_Trend"] < 0, 1,
    np.where(combined_test["Yield_MA_Trend"] > 0, -1, 0)
)

# Combined signal:
# only trade when the threshold signal agrees with the MA trend regime
combined_test["Combined_Position"] = np.where(
    (combined_test["Signal_10D_Yield_Change"] <= -threshold)
    & (combined_test["Yield_MA_Trend"] < 0),
    1,
    np.where(
        (combined_test["Signal_10D_Yield_Change"] >= threshold)
        & (combined_test["Yield_MA_Trend"] > 0),
        -1,
        0
    )
)


# =========================
# 2. PERFORMANCE SUMMARY FUNCTION
# =========================

def signal_summary(data, position_col, signal_name):
    temp = data[[position_col, "Fwd_Return_%"]].dropna().copy()

    valid = temp[position_col] != 0
    n_trades = valid.sum()
    trade_frequency = n_trades / len(temp)

    if n_trades > 0:
        signed_returns = temp.loc[valid, position_col] * temp.loc[valid, "Fwd_Return_%"]

        realised_direction = np.sign(temp.loc[valid, "Fwd_Return_%"])
        predicted_direction = temp.loc[valid, position_col]

        non_zero_realised = realised_direction != 0
        hit_rate = np.mean(
            predicted_direction[non_zero_realised] == realised_direction[non_zero_realised]
        )

        mean_active_return = signed_returns.mean()
        std_active_return = signed_returns.std()
        active_sharpe = mean_active_return / std_active_return if std_active_return != 0 else np.nan
        total_signed_return = signed_returns.sum()

    else:
        hit_rate = np.nan
        mean_active_return = np.nan
        std_active_return = np.nan
        active_sharpe = np.nan
        total_signed_return = np.nan

    return {
        "Signal": signal_name,
        "Number of trades": n_trades,
        "Trade frequency": trade_frequency,
        "Directional hit rate": hit_rate,
        "Mean active return (\\%)": mean_active_return,
        "Standard deviation (\\%)": std_active_return,
        "Active Sharpe ratio": active_sharpe,
        "Total signed return (\\%)": total_signed_return
    }


# =========================
# 3. CREATE COMPARISON TABLE
# =========================

combined_summary = pd.DataFrame([
    signal_summary(
        combined_test,
        "Baseline_Position",
        "25bps threshold signal"
    ),
    signal_summary(
        combined_test,
        "MA_Position",
        "20D--60D MA trend signal"
    ),
    signal_summary(
        combined_test,
        "Combined_Position",
        "25bps threshold + MA confirmation"
    )
])

round_cols = [
    "Trade frequency",
    "Directional hit rate",
    "Mean active return (\\%)",
    "Standard deviation (\\%)",
    "Active Sharpe ratio",
    "Total signed return (\\%)"
]

for col in round_cols:
    combined_summary[col] = combined_summary[col].round(4)

combined_summary.to_csv(TABLE_PATH / "combined_signal_comparison.csv", index=False)

latex_table = combined_summary.to_latex(
    index=False,
    escape=False,
    float_format="%.4f",
    column_format="lrrrrrrr"
)

with open(TABLE_PATH / "combined_signal_comparison.tex", "w") as f:
    f.write(latex_table)

print("\nCombined signal comparison:")
print(combined_summary)


# =========================
# 4. ADD SIGNED RETURNS TO DATAFRAME
# =========================

combined_test["Baseline_Signed_Return_%"] = (
    combined_test["Baseline_Position"] * combined_test["Fwd_Return_%"]
)

combined_test["MA_Signed_Return_%"] = (
    combined_test["MA_Position"] * combined_test["Fwd_Return_%"]
)

combined_test["Combined_Signed_Return_%"] = (
    combined_test["Combined_Position"] * combined_test["Fwd_Return_%"]
)


# =========================
# 5. PLOT: HIT RATE AND ACTIVE SHARPE COMPARISON
# =========================

plot_summary = combined_summary.set_index("Signal")

x = np.arange(len(plot_summary))
width = 0.35

fig, ax1 = plt.subplots(figsize=(9, 5))

bars1 = ax1.bar(
    x - width / 2,
    plot_summary["Directional hit rate"],
    width,
    label="Directional hit rate"
)

ax1.axhline(0.5, color="black", linestyle="--", linewidth=1)
ax1.set_ylabel("Directional hit rate")
ax1.set_ylim(0.45, max(0.65, plot_summary["Directional hit rate"].max() + 0.05))
ax1.set_xticks(x)
ax1.set_xticklabels(plot_summary.index, rotation=20, ha="right")
ax1.grid(axis="y", alpha=0.3)

ax2 = ax1.twinx()

bars2 = ax2.bar(
    x + width / 2,
    plot_summary["Active Sharpe ratio"],
    width,
    label="Active Sharpe ratio",
    alpha=0.65
)

ax2.set_ylabel("Active Sharpe ratio")

plt.title("Signal Comparison: Hit Rate and Active Sharpe")

fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2
)

fig.tight_layout()

savefig("combined_signal_hit_rate_sharpe_comparison")


# =========================
# 6. PLOT: TRADE FREQUENCY AND MEAN ACTIVE RETURN
# =========================

fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.bar(
    x - width / 2,
    plot_summary["Trade frequency"],
    width,
    label="Trade frequency"
)

ax1.set_ylabel("Trade frequency")
ax1.set_ylim(0, max(1.05, plot_summary["Trade frequency"].max() + 0.05))
ax1.set_xticks(x)
ax1.set_xticklabels(plot_summary.index, rotation=20, ha="right")
ax1.grid(axis="y", alpha=0.3)

ax2 = ax1.twinx()

ax2.bar(
    x + width / 2,
    plot_summary["Mean active return (\\%)"],
    width,
    label="Mean active return (%)",
    alpha=0.65
)

ax2.set_ylabel("Mean active return (%)")

plt.title("Signal Comparison: Trade Frequency and Active Return")

fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2
)

fig.tight_layout()

savefig("combined_signal_trade_frequency_return_comparison")

# =====================================================
# CURVE SIGNAL – PREDICTIVE ANALYSIS
# 10Y-3M AND 10Y-2Y SPREAD COMPARISON
# =====================================================

curve_eda = eda.copy()

# =========================
# TARGET: FORWARD RELATIVE RETURNS
# =========================

curve_eda["TLT_SHY_Fwd_10D_Ret"] = (
    curve_eda["TLT_Fwd_10D_Ret"] - curve_eda["SHY_Fwd_10D_Ret"]
)

curve_eda["IEF_Fwd_10D_Ret"] = (
    curve_eda["IEF_AdjClose"].shift(-10) / curve_eda["IEF_AdjClose"] - 1
)

curve_eda["TLT_IEF_Fwd_10D_Ret"] = (
    curve_eda["TLT_Fwd_10D_Ret"] - curve_eda["IEF_Fwd_10D_Ret"]
)

# =========================
# SIGNAL CONSTRUCTION
# =========================

def add_curve_signals(data, spread_col, spread_label):
    data = data.copy()

    data[f"Signal_{spread_label}_10D_Change"] = (
        data[spread_col] - data[spread_col].shift(10)
    ) * 100

    data[f"Signal_{spread_label}_20D_Change"] = (
        data[spread_col] - data[spread_col].shift(20)
    ) * 100

    data[f"Signal_{spread_label}_Level"] = data[spread_col]

    rolling_mean = data[spread_col].rolling(252).mean()
    rolling_std = data[spread_col].rolling(252).std()

    data[f"Signal_{spread_label}_ZScore"] = (
        data[spread_col] - rolling_mean
    ) / rolling_std

    return data


curve_eda["US10Y_3M_Spread"] = curve_eda["US10Y_Yield"] - curve_eda["US3M_Yield"]
curve_eda["US10Y_2Y_Spread"] = curve_eda["US10Y_Yield"] - curve_eda["US2Y_Yield"]

curve_eda = add_curve_signals(
    curve_eda,
    spread_col="US10Y_3M_Spread",
    spread_label="10Y_3M"
)

curve_eda = add_curve_signals(
    curve_eda,
    spread_col="US10Y_2Y_Spread",
    spread_label="10Y_2Y"
)

# =========================
# TEST FUNCTION
# =========================

def curve_signal_test(data, signal_col, return_col, signal_name):

    temp = data[[signal_col, return_col]].dropna().copy()

    x = temp[signal_col].values
    y = temp[return_col].values * 100

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    corr = np.corrcoef(x, y)[0, 1]
    r_squared = r_value ** 2
    t_stat = slope / std_err if std_err != 0 else np.nan

    predicted_direction = -np.sign(x)
    realised_direction = np.sign(y)

    valid = (predicted_direction != 0) & (realised_direction != 0)

    hit_rate = (
        np.mean(predicted_direction[valid] == realised_direction[valid])
        if valid.sum() > 0
        else np.nan
    )

    return {
        "Signal": signal_name,
        "Observations": len(temp),
        "Correlation": corr,
        "Beta (pct return per unit)": slope,
        "t-stat": t_stat,
        "p-value": p_value,
        "R-squared": r_squared,
        "Directional Hit Rate": hit_rate
    }


def run_curve_signal_tests(data, spread_label, return_col, return_label):
    display_label = spread_label.replace("_", "--")

    tests = [
        (
            f"Signal_{spread_label}_10D_Change",
            f"{display_label} past 10D curve change / next 10D {return_label} return"
        ),
        (
            f"Signal_{spread_label}_20D_Change",
            f"{display_label} past 20D curve change / next 10D {return_label} return"
        ),
        (
            f"Signal_{spread_label}_Level",
            f"{display_label} curve level / next 10D {return_label} return"
        ),
        (
            f"Signal_{spread_label}_ZScore",
            f"{display_label} curve z-score / next 10D {return_label} return"
        )
    ]

    results = []

    for signal_col, signal_name in tests:
        results.append(
            curve_signal_test(
                data=data,
                signal_col=signal_col,
                return_col=return_col,
                signal_name=signal_name
            )
        )

    results_df = pd.DataFrame(results)

    for col in [
        "Correlation",
        "Beta (pct return per unit)",
        "t-stat",
        "p-value",
        "R-squared",
        "Directional Hit Rate"
    ]:
        results_df[col] = results_df[col].round(4)

    print(f"\nCurve signal predictive tests: {spread_label}")
    print(results_df)

    results_df.to_csv(
        TABLE_PATH / f"curve_signal_predictive_tests_{spread_label.lower()}.csv",
        index=False
    )

    results_df.to_latex(
        TABLE_PATH / f"curve_signal_predictive_tests_{spread_label.lower()}.tex",
        index=False,
        escape=True,
        float_format="%.4f"
    )

    return results_df


curve_10y3m_results = run_curve_signal_tests(
    data=curve_eda,
    spread_label="10Y_3M",
    return_col="TLT_SHY_Fwd_10D_Ret",
    return_label="TLT–SHY"
)

curve_10y2y_results = run_curve_signal_tests(
    data=curve_eda,
    spread_label="10Y_2Y",
    return_col="TLT_IEF_Fwd_10D_Ret",
    return_label="TLT–IEF"
)

# =========================
# SCATTER PLOT FUNCTIONS
# =========================

def plot_curve_signal(data, signal_col, return_col, title, xlabel, ylabel, filename):

    plot_data = data[[signal_col, return_col]].dropna().copy()

    plt.figure(figsize=(7, 6))
    plt.scatter(
        plot_data[signal_col],
        plot_data[return_col] * 100,
        alpha=0.25,
        edgecolors="none"
    )

    x = plot_data[signal_col].values
    y = plot_data[return_col].values * 100

    slope, intercept, *_ = stats.linregress(x, y)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = intercept + slope * x_line

    plt.plot(
        x_line,
        y_line,
        color="black",
        linewidth=1.2,
        label="Regression line"
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)

    savefig(filename)


def make_curve_scatter_plots(data, spread_label, return_col, return_label):

    plot_specs = [
        (
            f"Signal_{spread_label}_10D_Change",
            "Past 10D curve change",
            f"Past 10D change in {spread_label} spread (bps)",
            f"curve_{spread_label.lower()}_10d_change_vs_forward_return"
        ),
        (
            f"Signal_{spread_label}_20D_Change",
            "Past 20D curve change",
            f"Past 20D change in {spread_label} spread (bps)",
            f"curve_{spread_label.lower()}_20d_change_vs_forward_return"
        ),
        (
            f"Signal_{spread_label}_Level",
            "Curve spread level",
            f"{spread_label} spread level (%)",
            f"curve_{spread_label.lower()}_level_vs_forward_return"
        ),
        (
            f"Signal_{spread_label}_ZScore",
            "Curve spread z-score",
            f"{spread_label} spread z-score",
            f"curve_{spread_label.lower()}_zscore_vs_forward_return"
        )
    ]

    for signal_col, signal_desc, xlabel, filename in plot_specs:
        plot_curve_signal(
            data=data,
            signal_col=signal_col,
            return_col=return_col,
            title=f"{spread_label}: {signal_desc} vs future {return_label} return",
            xlabel=xlabel,
            ylabel=f"Next 10D {return_label} return (%)",
            filename=filename
        )


make_curve_scatter_plots(
    data=curve_eda,
    spread_label="10Y_3M",
    return_col="TLT_SHY_Fwd_10D_Ret",
    return_label="TLT–SHY"
)

make_curve_scatter_plots(
    data=curve_eda,
    spread_label="10Y_2Y",
    return_col="TLT_IEF_Fwd_10D_Ret",
    return_label="TLT–IEF"
)

# =====================================================
# DEEPER THRESHOLD ANALYSIS:
# 10Y-2Y CURVE LEVEL SIGNAL + TLT-IEF SPREAD RETURN
# =====================================================

curve_level_data = curve_eda[
    ["Signal_10Y_2Y_Level", "TLT_IEF_Fwd_10D_Ret"]
].dropna().copy()

signal_col = "Signal_10Y_2Y_Level"
return_col = "TLT_IEF_Fwd_10D_Ret"

curve_level_data["Fwd_Return_%"] = curve_level_data[return_col] * 100

# Interpretation:
# High positive 10Y-2Y spread = steep curve
# Signal rule used here:
#   spread level high  -> long TLT-IEF
#   spread level low/inverted -> short TLT-IEF


# =====================================================
# 1. CONDITIONAL RETURN BY DECILE
# =====================================================

curve_level_data["Signal_Decile"] = pd.qcut(
    curve_level_data[signal_col],
    q=10,
    labels=[f"D{i}" for i in range(1, 11)]
)

curve_decile_stats = (
    curve_level_data
    .groupby("Signal_Decile")
    .agg(
        mean_signal=(signal_col, "mean"),
        min_signal=(signal_col, "min"),
        max_signal=(signal_col, "max"),
        mean_fwd_return_pct=("Fwd_Return_%", "mean"),
        std_fwd_return_pct=("Fwd_Return_%", "std"),
        count=("Fwd_Return_%", "count")
    )
)

curve_decile_stats["se"] = (
    curve_decile_stats["std_fwd_return_pct"] / np.sqrt(curve_decile_stats["count"])
)
curve_decile_stats["ci95"] = 1.96 * curve_decile_stats["se"]

curve_decile_stats["label"] = curve_decile_stats.apply(
    lambda row: f"{row['mean_signal']:.2f}%\nn={int(row['count'])}",
    axis=1
)

plt.figure(figsize=(10, 5.5))
plt.bar(
    curve_decile_stats["label"],
    curve_decile_stats["mean_fwd_return_pct"],
    yerr=curve_decile_stats["ci95"],
    capsize=4,
    alpha=0.85
)

plt.axhline(0, color="black", linewidth=1)
plt.title("Average Future TLT-IEF Return by 10Y-2Y Curve-Level Decile")
plt.xlabel("Average 10Y-2Y spread level by decile")
plt.ylabel("Average next 10D TLT-IEF return (%)")
plt.grid(axis="y", alpha=0.3)

savefig("curve_10y2y_level_conditional_return_by_decile_ci")


# =====================================================
# 2. SAVE DECILE TABLE
# =====================================================

curve_decile_table = curve_decile_stats.reset_index().copy()

curve_decile_table = curve_decile_table[
    [
        "Signal_Decile",
        "mean_signal",
        "min_signal",
        "max_signal",
        "mean_fwd_return_pct",
        "std_fwd_return_pct",
        "count",
        "ci95"
    ]
]

curve_decile_table = curve_decile_table.rename(columns={
    "Signal_Decile": "Signal decile",
    "mean_signal": "Mean 10Y-2Y spread (\\%)",
    "min_signal": "Minimum 10Y-2Y spread (\\%)",
    "max_signal": "Maximum 10Y-2Y spread (\\%)",
    "mean_fwd_return_pct": "Mean forward TLT-IEF return (\\%)",
    "std_fwd_return_pct": "Standard deviation (\\%)",
    "count": "Observations",
    "ci95": "95\\% confidence interval"
})

save_latex_table(
    curve_decile_table,
    TABLE_PATH / "curve_10y2y_level_conditional_return_by_decile.tex",
    column_format="lrrrrrrr"
)

curve_decile_table.to_csv(
    TABLE_PATH / "curve_10y2y_level_conditional_return_by_decile.csv",
    index=False
)

print("\n10Y-2Y curve-level decile conditional return stats:")
print(curve_decile_table.round(4))


# =====================================================
# 3. THRESHOLD ANALYSIS AROUND CURVE LEVEL
# =====================================================

# Because this is a level signal, thresholds are in percentage points, not bps.
# Example: threshold = 0.50 means 10Y-2Y spread >= +0.50% or <= -0.50%.

thresholds = np.arange(0.00, 2.25, 0.25)

threshold_results = []

for th in thresholds:

    position = np.where(
        curve_level_data[signal_col] >= th, 1,     # LONG when spread is high
        np.where(curve_level_data[signal_col] <= -th, -1, 0)  # SHORT when very low
    )

    actual = np.sign(curve_level_data["Fwd_Return_%"])
    valid = position != 0

    n_trades = valid.sum()
    trade_freq = n_trades / len(curve_level_data)

    if n_trades > 0:
        signed_returns = position[valid] * curve_level_data.loc[valid, "Fwd_Return_%"]

        hit_rate = np.mean(position[valid] == actual[valid])
        avg_active_return = signed_returns.mean()
        std_active_return = signed_returns.std()
        active_sharpe = (
            avg_active_return / std_active_return
            if std_active_return != 0
            else np.nan
        )
        total_signed_return = signed_returns.sum()

    else:
        hit_rate = np.nan
        avg_active_return = np.nan
        std_active_return = np.nan
        active_sharpe = np.nan
        total_signed_return = np.nan

    threshold_results.append({
        "Threshold_%": th,
        "Number_of_Trades": n_trades,
        "Trade_Frequency": trade_freq,
        "Hit_Rate": hit_rate,
        "Avg_Active_Return_%": avg_active_return,
        "Std_Active_Return_%": std_active_return,
        "Active_Sharpe": active_sharpe,
        "Total_Signed_Return_%": total_signed_return
    })

curve_threshold_stats = pd.DataFrame(threshold_results)

print("\n10Y-2Y curve-level threshold analysis:")
print(curve_threshold_stats.round(4))


# =====================================================
# 4. SAVE THRESHOLD TABLE
# =====================================================

curve_threshold_table = curve_threshold_stats.copy()

curve_threshold_table = curve_threshold_table.rename(columns={
    "Threshold_%": "Threshold (\\%)",
    "Number_of_Trades": "Number of trades",
    "Trade_Frequency": "Trade frequency",
    "Hit_Rate": "Hit rate",
    "Avg_Active_Return_%": "Average active return (\\%)",
    "Std_Active_Return_%": "Standard deviation (\\%)",
    "Active_Sharpe": "Active Sharpe ratio",
    "Total_Signed_Return_%": "Total signed return (\\%)"
})

save_latex_table(
    curve_threshold_table,
    TABLE_PATH / "curve_10y2y_level_threshold_analysis.tex",
    column_format="rrrrrrrr"
)

curve_threshold_table.to_csv(
    TABLE_PATH / "curve_10y2y_level_threshold_analysis.csv",
    index=False
)


# =====================================================
# 5. HIT RATE AND TRADE FREQUENCY BY THRESHOLD
# =====================================================

fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(
    curve_threshold_stats["Threshold_%"],
    curve_threshold_stats["Hit_Rate"],
    marker="o",
    label="Hit rate"
)

ax1.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)
ax1.set_xlabel("Absolute 10Y-2Y curve-level threshold (%)")
ax1.set_ylabel("Directional hit rate")
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()

ax2.plot(
    curve_threshold_stats["Threshold_%"],
    curve_threshold_stats["Trade_Frequency"],
    marker="s",
    linestyle="--",
    label="Trade frequency"
)

ax2.set_ylabel("Fraction of observations traded")

plt.title("10Y-2Y Curve Level: Hit Rate and Trade Frequency by Threshold")
fig.tight_layout()

savefig("curve_10y2y_level_hit_rate_trade_frequency_by_threshold")


# =====================================================
# 6. AVERAGE ACTIVE RETURN BY THRESHOLD
# =====================================================

plt.figure(figsize=(9, 5))

plt.plot(
    curve_threshold_stats["Threshold_%"],
    curve_threshold_stats["Avg_Active_Return_%"],
    marker="o"
)

plt.axhline(0, color="black", linewidth=1)

plt.title("10Y-2Y Curve Level: Average Active Return by Threshold")
plt.xlabel("Absolute 10Y-2Y curve-level threshold (%)")
plt.ylabel("Average signed next 10D TLT-IEF return when active (%)")
plt.grid(alpha=0.3)

savefig("curve_10y2y_level_average_active_return_by_threshold")


# =====================================================
# 7. ROBUST THRESHOLD OBJECTIVE:
# ACTIVE SHARPE × SQRT(NUMBER OF TRADES)
# =====================================================

robust_curve_results = []

for th in thresholds:

    position = np.where(
        curve_level_data[signal_col] >= th, 1,
        np.where(curve_level_data[signal_col] <= -th, -1, 0)
    )

    valid = position != 0
    n_trades = valid.sum()

    if n_trades > 1:

        signed_returns = position[valid] * curve_level_data.loc[valid, "Fwd_Return_%"]

        mean_ret = signed_returns.mean()
        std_ret = signed_returns.std()

        active_sharpe = mean_ret / std_ret if std_ret != 0 else np.nan
        robust_objective = active_sharpe * np.sqrt(n_trades)

        robust_curve_results.append({
            "Threshold_%": th,
            "Number_of_Trades": n_trades,
            "Mean_Active_Return_%": mean_ret,
            "Std_Active_Return_%": std_ret,
            "Active_Sharpe": active_sharpe,
            "Robust_Objective": robust_objective
        })

curve_robust_df = pd.DataFrame(robust_curve_results)

best_curve_robust_row = curve_robust_df.loc[
    curve_robust_df["Robust_Objective"].idxmax()
]

print("\n10Y-2Y curve-level robust threshold objective:")
print(curve_robust_df.round(4))

print("\nBest 10Y-2Y threshold based on active Sharpe × sqrt(number of trades):")
print(best_curve_robust_row.round(4))


# =====================================================
# 8. SAVE ROBUST OBJECTIVE TABLE
# =====================================================

curve_robust_table = curve_robust_df.copy()

curve_robust_table = curve_robust_table.rename(columns={
    "Threshold_%": "Threshold (\\%)",
    "Number_of_Trades": "Number of trades",
    "Mean_Active_Return_%": "Mean active return (\\%)",
    "Std_Active_Return_%": "Standard deviation (\\%)",
    "Active_Sharpe": "Active Sharpe ratio",
    "Robust_Objective": "Robust objective"
})

save_latex_table(
    curve_robust_table,
    TABLE_PATH / "curve_10y2y_level_robust_threshold_objective.tex",
    column_format="rrrrrr"
)

curve_robust_table.to_csv(
    TABLE_PATH / "curve_10y2y_level_robust_threshold_objective.csv",
    index=False
)


# =====================================================
# 9. PLOT ROBUST OBJECTIVE
# =====================================================

plt.figure(figsize=(9, 5))

plt.plot(
    curve_robust_df["Threshold_%"],
    curve_robust_df["Robust_Objective"],
    marker="o",
    linewidth=1.8
)

best_threshold = best_curve_robust_row["Threshold_%"]
best_objective = best_curve_robust_row["Robust_Objective"]

plt.axvline(
    best_threshold,
    color="black",
    linestyle="--",
    linewidth=1,
    label=f"Best threshold = {best_threshold:.2f}%"
)

plt.scatter(
    best_threshold,
    best_objective,
    s=80,
    zorder=5
)

plt.title("10Y-2Y Curve Level: Robust Threshold Objective")
plt.xlabel("Absolute 10Y-2Y curve-level threshold (%)")
plt.ylabel("Active Sharpe × √Number of Trades")
plt.legend()
plt.grid(alpha=0.3)

savefig("curve_10y2y_level_robust_threshold_objective")


# =====================================================
# 10. PLOT ACTIVE SHARPE ONLY
# =====================================================

plt.figure(figsize=(9, 5))

plt.plot(
    curve_robust_df["Threshold_%"],
    curve_robust_df["Active_Sharpe"],
    marker="o",
    linewidth=1.8
)

plt.axhline(0, color="black", linewidth=1)

plt.axvline(
    best_threshold,
    color="black",
    linestyle="--",
    linewidth=1,
    label=f"Best robust threshold = {best_threshold:.2f}%"
)

plt.title("10Y-2Y Curve Level: Active Sharpe by Threshold")
plt.xlabel("Absolute 10Y-2Y curve-level threshold (%)")
plt.ylabel("Mean active return / standard deviation")
plt.legend()
plt.grid(alpha=0.3)

savefig("curve_10y2y_level_active_sharpe_by_threshold")


# =====================================================
# MODULAR BACKTESTING FRAMEWORK
# STRATEGY-SPECIFIC WALK-FORWARD THRESHOLD SELECTION
# =====================================================


# =====================================================
# 1. GENERAL SETTINGS
# =====================================================

initial_wealth = 1.0
periods_per_year = 252

holding_period = 10
threshold_grid = np.arange(5, 60, 5)
spread_upper_threshold_grid = np.arange(1.0, 2.6, 0.1)
spread_lower_threshold_grid = np.arange(-0.30, -0.01, 0.01)

lookback_window = 1260
rebalance_frequency = 252

default_transaction_cost_bps = 1.0

# =====================================================
# 2. NAMING HELPERS
# =====================================================

def make_strategy_name(
    strategy_type="directional",
    sizing="binary",
    spread_short_asset="SHY",
    use_dv01_neutral=False,
    use_vol_target=False,
    transaction_cost_bps=0.0
):
    parts = []

    if strategy_type == "directional":
        parts.append("Directional_TLT")
    elif strategy_type == "spread":
        parts.append(f"Spread_TLT_{spread_short_asset}")
    else:
        raise ValueError("strategy_type must be 'directional' or 'spread'.")

    parts.append(sizing.capitalize())

    if use_dv01_neutral:
        parts.append("DV01Neutral")

    if use_vol_target:
        parts.append("VolTarget")

    if transaction_cost_bps > 0:
        parts.append(f"TCost_{transaction_cost_bps:g}bps")
    else:
        parts.append("NoTCost")

    return "_".join(parts)

def make_strategy_display_name(
    strategy_type="directional",
    sizing="binary",
    spread_short_asset="SHY",
    use_dv01_neutral=False,
    use_vol_target=False
):
    if strategy_type == "directional":
        name = "Directional TLT"
    elif strategy_type == "spread":
        name = f"TLT--{spread_short_asset} Spread"
    else:
        raise ValueError("strategy_type must be 'directional' or 'spread'.")

    if sizing == "binary":
        name += " | Threshold Positioning"
    elif sizing == "scaled":
        name += " | Continuous Position Sizing"

    if use_dv01_neutral:
        name += " | DV01-Neutral"

    if use_vol_target:
        name += " | Volatility Targeted"

    return name

def clean_filename(name):
    return (
        name.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "pct")
    )

def latex_escape_text(s):
    """
    Escapes LaTeX-sensitive characters in text columns.
    Needed because strategy names contain underscores.
    """
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


# =====================================================
# 3. DATA PREPARATION
# =====================================================

def prepare_backtest_data(data, forward_horizon=10):
    """
    Creates returns, directional signal, 10Y-2Y curve-level signal,
    and forward ETF returns.

    Directional signal:
        10D change in US 10Y yield, in bps

    Spread signal:
        10Y-2Y curve level, in percentage points
    """

    bt = data.copy()

    # Directional 10Y yield-change signal, in bps
    bt["Signal_10D_Yield_Change"] = (
        bt["US10Y_Yield"] - bt["US10Y_Yield"].shift(10)
    ) * 100

    # 10Y-2Y curve spread level, in percentage points
    bt["US10Y_2Y_Spread"] = (
        bt["US10Y_Yield"] - bt["US2Y_Yield"]
    )

    # This is the spread strategy signal from the EDA
    bt["Signal_10Y_2Y_Curve_Level"] = bt["US10Y_2Y_Spread"]

    for ticker in ["TLT", "IEF", "SHY"]:

        price_col = f"{ticker}_AdjClose"
        ret_col = f"{ticker}_Return"
        fwd_col = f"{ticker}_Fwd_{forward_horizon}D_Ret"

        if price_col in bt.columns:
            bt[ret_col] = bt[price_col].pct_change()
            bt[fwd_col] = (
                bt[price_col].shift(-forward_horizon) / bt[price_col] - 1
            )

    bt["TLT_Fwd_10D_Ret"] = bt[f"TLT_Fwd_{forward_horizon}D_Ret"]

    return bt

def add_empirical_durations(bt, duration_window=252):
    bt = bt.copy()

    yield_map = {
        "TLT": "US10Y_Change_pctpts",
        "IEF": "US10Y_Change_pctpts",   # better: add DGS7 from FRED later
        "SHY": "US2Y_Change_pctpts"
    }

    for ticker, y_col in yield_map.items():
        r = bt[f"{ticker}_Return"]
        dy = bt[y_col]

        beta = (
            r.rolling(duration_window).cov(dy)
            / dy.rolling(duration_window).var()
        )

        # r ≈ -D * Δy_decimal, and Δy_pctpts = 100 * Δy_decimal
        bt[f"{ticker}_Duration"] = (-100 * beta).shift(1)

        bt[f"{ticker}_Duration"] = (
            bt[f"{ticker}_Duration"]
            .clip(lower=0.1, upper=40)
            .ffill()
        )

    return bt

def add_spread_weights(bt, spread_short_asset="IEF", use_dv01_neutral=False):
    bt = bt.copy()

    if use_dv01_neutral:
        d_long = bt["TLT_Duration"]
        d_short = bt[f"{spread_short_asset}_Duration"]

        bt["Spread_Long_Weight"] = d_short / (d_long + d_short)
        bt["Spread_Short_Weight"] = -d_long / (d_long + d_short)
    else:
        bt["Spread_Long_Weight"] = 0.5
        bt["Spread_Short_Weight"] = -0.5

    return bt


def add_forward_objective_return(
    bt,
    strategy_type="directional",
    spread_short_asset="SHY",
    use_dv01_neutral=False,
    forward_horizon=10
):
    bt = bt.copy()

    if strategy_type == "directional":
        bt["Forward_Objective_Return"] = bt[f"TLT_Fwd_{forward_horizon}D_Ret"]

    elif strategy_type == "spread":

        bt = add_spread_weights(
            bt,
            spread_short_asset=spread_short_asset,
            use_dv01_neutral=use_dv01_neutral
        )

        bt["Forward_Objective_Return"] = (
            bt["Spread_Long_Weight"] * bt[f"TLT_Fwd_{forward_horizon}D_Ret"]
            + bt["Spread_Short_Weight"] * bt[f"{spread_short_asset}_Fwd_{forward_horizon}D_Ret"]
        )

        bt["Objective_Long_Weight"] = bt["Spread_Long_Weight"]
        bt["Objective_Short_Weight"] = bt["Spread_Short_Weight"]

    else:
        raise ValueError("strategy_type must be 'directional' or 'spread'.")

    return bt


# =====================================================
# 4. WALK-FORWARD THRESHOLD SELECTION
# =====================================================

def select_best_threshold(
    train_data,
    thresholds,
    strategy_type="directional",
    signal_col="Signal_Input",
    forward_return_col="Forward_Objective_Return",
    min_trades=20
):
    temp = train_data[[signal_col, forward_return_col]].dropna().copy()

    if strategy_type == "directional":

        best_threshold = np.nan
        best_objective = -np.inf

        for th in thresholds:

            signal = np.where(
                temp[signal_col] <= -th, 1,
                np.where(temp[signal_col] >= th, -1, 0)
            )

            valid = signal != 0
            n_trades = valid.sum()

            if n_trades < min_trades:
                continue

            signed_returns = signal[valid] * temp.loc[valid, forward_return_col].values

            mean_ret = signed_returns.mean()
            std_ret = signed_returns.std()

            if std_ret == 0 or np.isnan(std_ret):
                continue

            active_sharpe = mean_ret / std_ret
            objective = active_sharpe * np.sqrt(n_trades)

            if objective > best_objective:
                best_objective = objective
                best_threshold = th

        return best_threshold

    elif strategy_type == "spread":

        upper_grid = thresholds["upper"]
        lower_grid = thresholds["lower"]

        best_upper = np.nan
        best_lower = np.nan
        best_objective = -np.inf

        for upper_th in upper_grid:
            for lower_th in lower_grid:

                if lower_th >= upper_th:
                    continue

                signal = np.where(
                    temp[signal_col] >= upper_th, 1,
                    np.where(temp[signal_col] <= lower_th, -1, 0)
                )

                valid = signal != 0
                n_trades = valid.sum()

                if n_trades < min_trades:
                    continue

                signed_returns = signal[valid] * temp.loc[valid, forward_return_col].values

                mean_ret = signed_returns.mean()
                std_ret = signed_returns.std()

                if std_ret == 0 or np.isnan(std_ret):
                    continue

                active_sharpe = mean_ret / std_ret
                objective = active_sharpe * np.sqrt(n_trades)

                if objective > best_objective:
                    best_objective = objective
                    best_upper = upper_th
                    best_lower = lower_th

        return best_upper, best_lower

    else:
        raise ValueError("strategy_type must be 'directional' or 'spread'.")

def add_walk_forward_thresholds(
    bt,
    threshold_grid,
    lookback_window=1260,
    rebalance_frequency=252,
    strategy_type="directional",
    signal_col="Signal_Input",
    forward_return_col="Forward_Objective_Return"
):
    bt = bt.copy()

    if strategy_type == "directional":
        bt["Selected_Threshold"] = np.nan

    elif strategy_type == "spread":
        bt["Selected_Upper_Threshold"] = np.nan
        bt["Selected_Lower_Threshold"] = np.nan

    else:
        raise ValueError("strategy_type must be 'directional' or 'spread'.")

    for i in range(lookback_window, len(bt), rebalance_frequency):

        train_start = i - lookback_window
        train_end = i
        train_data = bt.iloc[train_start:train_end].copy()

        best_threshold = select_best_threshold(
            train_data=train_data,
            thresholds=threshold_grid,
            strategy_type=strategy_type,
            signal_col=signal_col,
            forward_return_col=forward_return_col
        )

        apply_start = i
        apply_end = min(i + rebalance_frequency, len(bt))

        if strategy_type == "directional":

            bt.iloc[
                apply_start:apply_end,
                bt.columns.get_loc("Selected_Threshold")
            ] = best_threshold

        elif strategy_type == "spread":

            best_upper, best_lower = best_threshold

            bt.iloc[
                apply_start:apply_end,
                bt.columns.get_loc("Selected_Upper_Threshold")
            ] = best_upper

            bt.iloc[
                apply_start:apply_end,
                bt.columns.get_loc("Selected_Lower_Threshold")
            ] = best_lower

    if strategy_type == "directional":
        bt["Selected_Threshold"] = bt["Selected_Threshold"].ffill()

    elif strategy_type == "spread":
        bt["Selected_Upper_Threshold"] = bt["Selected_Upper_Threshold"].ffill()
        bt["Selected_Lower_Threshold"] = bt["Selected_Lower_Threshold"].ffill()

    return bt

# =====================================================
# 5. SIGNAL GENERATION
# =====================================================

def generate_raw_signal(
    bt,
    sizing="binary",
    strategy_type="directional",
    signal_col="Signal_Input",
    scaling_multiple=2.0
):
    bt = bt.copy()

    signal_value = bt[signal_col]

    if sizing == "binary":

        if strategy_type == "directional":

            threshold = bt["Selected_Threshold"]

            bt["Signal_Position"] = np.where(
                signal_value <= -threshold, 1,
                np.where(signal_value >= threshold, -1, 0)
            )

        elif strategy_type == "spread":

            upper_threshold = bt["Selected_Upper_Threshold"]
            lower_threshold = bt["Selected_Lower_Threshold"]

            bt["Signal_Position"] = np.where(
                signal_value >= upper_threshold, 1,
                np.where(signal_value <= lower_threshold, -1, 0)
            )

        else:
            raise ValueError("strategy_type must be 'directional' or 'spread'.")

    elif sizing == "scaled":

        if strategy_type == "directional":

            threshold = bt["Selected_Threshold"]

            signal_strength = (
                (signal_value.abs() - threshold)
                / ((scaling_multiple - 1) * threshold)
            ).clip(0, 1)

            bt["Signal_Position"] = -np.sign(signal_value) * signal_strength

        elif strategy_type == "spread":

            upper_threshold = bt["Selected_Upper_Threshold"]
            lower_threshold = bt["Selected_Lower_Threshold"]

            long_strength = (
                (signal_value - upper_threshold)
                / ((scaling_multiple - 1) * upper_threshold)
            ).clip(0, 1)

            short_strength = (
                (lower_threshold - signal_value)
                / ((scaling_multiple - 1) * lower_threshold.abs())
            ).clip(0, 1)

            bt["Signal_Position"] = np.where(
                signal_value >= upper_threshold,
                long_strength,
                np.where(signal_value <= lower_threshold, -short_strength, 0)
            )

        else:
            raise ValueError("strategy_type must be 'directional' or 'spread'.")

    else:
        raise ValueError("sizing must be 'binary' or 'scaled'.")

    bt["Signal_Position"] = bt["Signal_Position"].clip(-1, 1)

    return bt


# =====================================================
# 6. SIGNAL PERSISTENCE
# =====================================================

def apply_signal_persistence(bt, holding_period=10):
    """
    Applies 10-day signal persistence.
    """

    bt = bt.copy()

    positions = []
    current_position = 0
    days_remaining = 0

    signals = bt["Signal_Position"].values

    for signal in signals:

        if signal != 0:
            current_position = signal
            days_remaining = holding_period
            positions.append(current_position)

        elif days_remaining > 0:
            positions.append(current_position)
            days_remaining -= 1

        else:
            current_position = 0
            positions.append(0)

    bt["Unlagged_Strategy_Position"] = positions

    # Avoid look-ahead bias
    bt["Strategy_Position"] = bt["Unlagged_Strategy_Position"].shift(1)

    return bt


# =====================================================
# 7. RETURN CONSTRUCTION
# =====================================================

def construct_directional_returns(bt):
    """
    Directional TLT strategy.
    """

    bt = bt.copy()

    bt["Gross_Strategy_Return"] = (
        bt["Strategy_Position"] * bt["TLT_Return"]
    )

    # Benchmark for directional strategy
    bt["Benchmark_Return"] = bt["TLT_Return"]
    bt["Benchmark_Name"] = "Buy-and-hold TLT"

    return bt 


def construct_spread_returns(bt, spread_short_asset="IEF", use_dv01_neutral=False):
    bt = bt.copy()

    bt = add_spread_weights(
        bt,
        spread_short_asset=spread_short_asset,
        use_dv01_neutral=use_dv01_neutral
    )

    # Return actually traded by the strategy
    bt["Spread_Return"] = (
        bt["Spread_Long_Weight"] * bt["TLT_Return"]
        + bt["Spread_Short_Weight"] * bt[f"{spread_short_asset}_Return"]
    )

    bt["Gross_Strategy_Return"] = bt["Strategy_Position"] * bt["Spread_Return"]

    # Final leg weights after signal sizing / vol targeting
    bt["TLT_Final_Weight"] = bt["Strategy_Position"] * bt["Spread_Long_Weight"]
    bt[f"{spread_short_asset}_Final_Weight"] = (
        bt["Strategy_Position"] * bt["Spread_Short_Weight"]
    )

    # Keep benchmark as original simple 50/50 spread
    bt["Benchmark_Return"] = (
        0.5 * bt["TLT_Return"]
        - 0.5 * bt[f"{spread_short_asset}_Return"]
    )

    bt["Benchmark_Name"] = f"Buy-and-hold 50/50 TLT-{spread_short_asset} spread"

    return bt


# =====================================================
# 8. TRANSACTION COSTS
# =====================================================

def apply_transaction_costs(bt, transaction_cost_bps=0.0, spread_short_asset=None):
    bt = bt.copy()
    cost_rate = transaction_cost_bps / 10000

    if spread_short_asset is not None and "TLT_Final_Weight" in bt.columns:
        turnover = (
            bt["TLT_Final_Weight"].diff().abs().fillna(0)
            + bt[f"{spread_short_asset}_Final_Weight"].diff().abs().fillna(0)
        )
    else:
        turnover = bt["Strategy_Position"].diff().abs().fillna(0)

    bt["Turnover"] = turnover
    bt["Transaction_Cost"] = cost_rate * turnover
    bt["Strategy_Return"] = bt["Gross_Strategy_Return"] - bt["Transaction_Cost"]

    return bt


# =====================================================
# 9. VOLATILITY TARGETING
# =====================================================

def apply_volatility_targeting(
    bt,
    target_vol=0.15,
    vol_window=126,
    max_leverage=2.0
):
    bt = bt.copy()

    if "Spread_Return" in bt.columns:
        vol_base_col = "Spread_Return"
    else:
        vol_base_col = "TLT_Return"

    rolling_vol = bt[vol_base_col].rolling(vol_window).std().shift(1) * np.sqrt(252)

    bt["Vol_Estimate"] = rolling_vol
    bt["Vol_Target_Multiplier"] = target_vol / bt["Vol_Estimate"]

    bt["Vol_Target_Multiplier"] = (
        bt["Vol_Target_Multiplier"]
        .replace([np.inf, -np.inf], np.nan)
        .clip(upper=max_leverage)
        .fillna(1.0)
    )

    bt["Strategy_Position"] = bt["Strategy_Position"] * bt["Vol_Target_Multiplier"]

    return bt

# =====================================================
# 10. WEALTH AND PERFORMANCE METRICS
# =====================================================

def max_drawdown(wealth_series):
    running_max = wealth_series.cummax()
    drawdown = wealth_series / running_max - 1
    return drawdown.min(), drawdown


def compute_wealth(bt, initial_wealth=1.0):
    bt = bt.copy()

    bt["Strategy_Wealth"] = initial_wealth * (
        1 + bt["Strategy_Return"]
    ).cumprod()

    bt["Benchmark_Wealth"] = initial_wealth * (
        1 + bt["Benchmark_Return"]
    ).cumprod()

    _, strategy_drawdown = max_drawdown(bt["Strategy_Wealth"])
    _, benchmark_drawdown = max_drawdown(bt["Benchmark_Wealth"])

    bt["Strategy_Drawdown"] = strategy_drawdown
    bt["Benchmark_Drawdown"] = benchmark_drawdown

    bt["Excess_Return"] = bt["Strategy_Return"] - bt["Benchmark_Return"]

    bt["Cumulative_Excess_Return"] = (
        bt["Strategy_Wealth"] / bt["Benchmark_Wealth"] - 1
    )

    return bt


def performance_metrics(return_series, wealth_series, name, periods_per_year=252):
    return_series = return_series.dropna()
    wealth_series = wealth_series.loc[return_series.index]

    total_return = wealth_series.iloc[-1] / wealth_series.iloc[0] - 1

    n_years = len(return_series) / periods_per_year
    annual_return = (1 + total_return) ** (1 / n_years) - 1

    annual_volatility = return_series.std() * np.sqrt(periods_per_year)

    sharpe_ratio = (
        annual_return / annual_volatility
        if annual_volatility != 0
        else np.nan
    )

    mdd, _ = max_drawdown(wealth_series)

    calmar_ratio = (
        annual_return / abs(mdd)
        if mdd != 0
        else np.nan
    )

    positive_days = (return_series > 0).mean()

    var_95 = return_series.quantile(0.05)
    es_95 = return_series[return_series <= var_95].mean()

    return {
        "Strategy": name,
        "Total return (\\%)": total_return * 100,
        "Annualised return (\\%)": annual_return * 100,
        "Annualised volatility (\\%)": annual_volatility * 100,
        "Sharpe ratio": sharpe_ratio,
        "Maximum drawdown (\\%)": mdd * 100,
        "Calmar ratio": calmar_ratio,
        "Positive daily return rate": positive_days,
        "Daily VaR 95\\% (\\%)": var_95 * 100,
        "Daily ES 95\\% (\\%)": es_95 * 100
    }


def create_performance_summary(bt, strategy_name):
    strategy_metrics = performance_metrics(
        bt["Strategy_Return"],
        bt["Strategy_Wealth"],
        strategy_name
    )

    benchmark_name = (
        bt["Benchmark_Name"].dropna().iloc[0]
        if "Benchmark_Name" in bt.columns and bt["Benchmark_Name"].dropna().shape[0] > 0
        else "Benchmark"
    )

    benchmark_metrics = performance_metrics(
        bt["Benchmark_Return"],
        bt["Benchmark_Wealth"],
        benchmark_name
    )

    summary = pd.DataFrame({
        "Metric": list(strategy_metrics.keys())[1:],
        strategy_name: list(strategy_metrics.values())[1:],
        benchmark_name: list(benchmark_metrics.values())[1:]
    })

    for col in summary.columns.drop("Metric"):
        summary[col] = summary[col].round(4)

    return summary


def create_exposure_summary(bt, holding_period=10):
    active = bt["Strategy_Position"] != 0

    if active.sum() > 0:
        strategy_hit_rate = np.mean(
            np.sign(bt.loc[active, "Strategy_Position"])
            == np.sign(bt.loc[active, "TLT_Return"])
        )
    else:
        strategy_hit_rate = np.nan

    turnover = bt["Turnover"] if "Turnover" in bt.columns else bt["Strategy_Position"].diff().abs().fillna(0)

    metrics = [
        "Number of observations",
        "Active trading days",
        "Trade frequency",
        "Long exposure frequency",
        "Short exposure frequency",
        "Cash frequency",
        "Directional hit rate when active",
        "Number of position changes",
        "Average absolute daily turnover",
        "Average transaction cost",
        "Total transaction cost",
        "Maximum signal persistence"
    ]

    values = [
        len(bt),
        active.sum(),
        active.mean(),
        (bt["Strategy_Position"] > 0).mean(),
        (bt["Strategy_Position"] < 0).mean(),
        (bt["Strategy_Position"] == 0).mean(),
        strategy_hit_rate,
        (turnover > 0).sum(),
        turnover.mean(),
        bt["Transaction_Cost"].mean() if "Transaction_Cost" in bt.columns else 0,
        bt["Transaction_Cost"].sum() if "Transaction_Cost" in bt.columns else 0,
        holding_period
    ]

    if "Selected_Threshold" in bt.columns:
        metrics += [
            "Average selected threshold",
            "Minimum selected threshold",
            "Maximum selected threshold"
        ]

        values += [
            bt["Selected_Threshold"].mean(),
            bt["Selected_Threshold"].min(),
            bt["Selected_Threshold"].max()
        ]

    if "Selected_Upper_Threshold" in bt.columns:
        metrics += [
            "Average selected upper threshold",
            "Minimum selected upper threshold",
            "Maximum selected upper threshold"
        ]

        values += [
            bt["Selected_Upper_Threshold"].mean(),
            bt["Selected_Upper_Threshold"].min(),
            bt["Selected_Upper_Threshold"].max()
        ]

    if "Selected_Lower_Threshold" in bt.columns:
        metrics += [
            "Average selected lower threshold",
            "Minimum selected lower threshold",
            "Maximum selected lower threshold"
        ]

        values += [
            bt["Selected_Lower_Threshold"].mean(),
            bt["Selected_Lower_Threshold"].min(),
            bt["Selected_Lower_Threshold"].max()
        ]

    exposure_summary = pd.DataFrame({
        "Metric": metrics,
        "Value": values
    })

    exposure_summary["Value"] = exposure_summary["Value"].round(6)

    return exposure_summary

def create_trade_summary(bt):
    temp = bt.copy()

    position_change = temp["Strategy_Position"].diff().abs().fillna(0)
    trade_starts = position_change > 0

    trade_ids = trade_starts.cumsum()
    active = temp["Strategy_Position"] != 0

    trades = temp.loc[active].copy()
    trades["Trade_ID"] = trade_ids.loc[active]

    trade_returns = (
        trades
        .groupby("Trade_ID")["Strategy_Return"]
        .apply(lambda x: (1 + x).prod() - 1)
    )

    winning_trades = trade_returns[trade_returns > 0]
    losing_trades = trade_returns[trade_returns < 0]

    trade_summary = pd.DataFrame({
        "Metric": [
            "Number of trades",
            "Win rate",
            "Average trade return (\\%)",
            "Median trade return (\\%)",
            "Average winning trade (\\%)",
            "Average losing trade (\\%)",
            "Profit-loss ratio",
            "Best trade (\\%)",
            "Worst trade (\\%)"
        ],
        "Value": [
            len(trade_returns),
            (trade_returns > 0).mean(),
            trade_returns.mean() * 100,
            trade_returns.median() * 100,
            winning_trades.mean() * 100 if len(winning_trades) > 0 else np.nan,
            losing_trades.mean() * 100 if len(losing_trades) > 0 else np.nan,
            abs(winning_trades.mean() / losing_trades.mean()) if len(winning_trades) > 0 and len(losing_trades) > 0 else np.nan,
            trade_returns.max() * 100,
            trade_returns.min() * 100
        ]
    })

    trade_summary["Value"] = trade_summary["Value"].round(4)

    return trade_summary

# =====================================================
# 11. PLOTTING FUNCTIONS
# =====================================================

def plot_backtest_results(bt, strategy_name, fig_path, filename_name=None):
    fname = clean_filename(filename_name if filename_name is not None else strategy_name)

    benchmark_name = (
        bt["Benchmark_Name"].dropna().iloc[0]
        if "Benchmark_Name" in bt.columns and bt["Benchmark_Name"].dropna().shape[0] > 0
        else "Benchmark"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(bt.index, bt["Strategy_Wealth"], label=strategy_name, linewidth=1.6)
    plt.plot(bt.index, bt["Benchmark_Wealth"], label=benchmark_name, linewidth=1.6)
    plt.title(f"Cumulative Wealth: {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Wealth index")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / f"{fname}_equity_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(bt.index, bt["Strategy_Drawdown"] * 100, label=strategy_name, linewidth=1.4)
    plt.plot(bt.index, bt["Benchmark_Drawdown"] * 100, label=benchmark_name, linewidth=1.4)
    plt.title(f"Drawdown Comparison: {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / f"{fname}_drawdown.png", dpi=300, bbox_inches="tight")
    plt.show()

    rolling_window = 252

    bt["Strategy_Rolling_Sharpe"] = (
        bt["Strategy_Return"].rolling(rolling_window).mean()
        / bt["Strategy_Return"].rolling(rolling_window).std()
    ) * np.sqrt(252)

    bt["Benchmark_Rolling_Sharpe"] = (
        bt["Benchmark_Return"].rolling(rolling_window).mean()
        / bt["Benchmark_Return"].rolling(rolling_window).std()
    ) * np.sqrt(252)

    plt.figure(figsize=(10, 5))
    plt.plot(bt.index, bt["Strategy_Rolling_Sharpe"], label=strategy_name, linewidth=1.4)
    plt.plot(bt.index, bt["Benchmark_Rolling_Sharpe"], label=benchmark_name, linewidth=1.4)
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Rolling 252-Day Sharpe Ratio: {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe ratio")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / f"{fname}_rolling_sharpe.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.step(bt.index, bt["Strategy_Position"], where="post", linewidth=1.1)
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Strategy Position Through Time: {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Position weight")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / f"{fname}_position.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 4))

    if "Selected_Threshold" in bt.columns:
        plt.step(bt.index, bt["Selected_Threshold"], where="post", linewidth=1.2, label="Selected threshold")

    if "Selected_Upper_Threshold" in bt.columns:
        plt.step(bt.index, bt["Selected_Upper_Threshold"], where="post", linewidth=1.2, label="Upper threshold")

    if "Selected_Lower_Threshold" in bt.columns:
        plt.step(bt.index, bt["Selected_Lower_Threshold"], where="post", linewidth=1.2, label="Lower threshold")

    plt.title(f"Walk-Forward Selected Thresholds: {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Selected threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / f"{fname}_selected_threshold.png", dpi=300, bbox_inches="tight")
    plt.show()

    bt["Strategy_Rolling_Vol"] = (
        bt["Strategy_Return"].rolling(rolling_window).std() * np.sqrt(252)
    )

    bt["Benchmark_Rolling_Vol"] = (
        bt["Benchmark_Return"].rolling(rolling_window).std() * np.sqrt(252)
    )

    plt.figure(figsize=(10, 5))
    plt.plot(bt.index, bt["Strategy_Rolling_Vol"] * 100, label=strategy_name, linewidth=1.4)
    plt.plot(bt.index, bt["Benchmark_Rolling_Vol"] * 100, label=benchmark_name, linewidth=1.4)
    plt.title(f"Rolling 252-Day Volatility: {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Annualised volatility (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / f"{fname}_rolling_volatility.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(bt.index, bt["Cumulative_Excess_Return"] * 100, label="Cumulative excess return", linewidth=1.5)
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Cumulative Excess Return vs Benchmark: {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative excess return (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / f"{fname}_cumulative_excess_return.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(bt["Strategy_Return"].dropna() * 100, bins=60, alpha=0.75)
    plt.axvline(0, color="black", linewidth=1)
    plt.title(f"Daily Return Distribution: {strategy_name}")
    plt.xlabel("Daily return (%)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / f"{fname}_return_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

# =====================================================
# 12. MAIN BACKTEST FUNCTION
# =====================================================
def add_strategy_signal_input(bt, strategy_type="directional"):
    """
    Chooses the correct signal for the strategy.

    Directional:
        10D change in US 10Y yield, in bps

    Spread:
        10Y-2Y curve level, in percentage points
    """

    bt = bt.copy()

    if strategy_type == "directional":
        bt["Signal_Input"] = bt["Signal_10D_Yield_Change"]

    elif strategy_type == "spread":
        bt["Signal_Input"] = bt["Signal_10Y_2Y_Curve_Level"]

    else:
        raise ValueError("strategy_type must be 'directional' or 'spread'.")

    return bt

def run_backtest(
    data,
    strategy_type="directional",
    sizing="binary",
    spread_short_asset="SHY",
    use_dv01_neutral=False,
    use_vol_target=False,
    target_vol=0.15,
    vol_window=63,
    max_leverage=2.0,
    duration_window=252,
    transaction_cost_bps=0.0,
    holding_period=10,
    threshold_grid=np.arange(5, 60, 5),
    lookback_window=1260,
    rebalance_frequency=252,
    forward_horizon=10,
    save_outputs=True,
    make_plots=True,
    table_path=TABLE_PATH,
    fig_path=FIG_PATH
):
    """
    Fully modular strategy backtest with strategy-specific threshold selection.
    """

    strategy_name = make_strategy_name(
        strategy_type=strategy_type,
        sizing=sizing,
        spread_short_asset=spread_short_asset,
        use_dv01_neutral=use_dv01_neutral,
        use_vol_target=use_vol_target,
        transaction_cost_bps=transaction_cost_bps
    )

    display_name = make_strategy_display_name(
        strategy_type=strategy_type,
        sizing=sizing,
        spread_short_asset=spread_short_asset,
        use_dv01_neutral=use_dv01_neutral,
        use_vol_target=use_vol_target
    )

    print(f"\nRunning backtest: {strategy_name}")

    bt = prepare_backtest_data(
        data,
        forward_horizon=forward_horizon
    )

    if strategy_type == "spread" and use_dv01_neutral:
        bt = add_empirical_durations(bt, duration_window=duration_window)

    bt = add_strategy_signal_input(
        bt,
        strategy_type=strategy_type
    )

    bt = add_forward_objective_return(
        bt,
        strategy_type=strategy_type,
        spread_short_asset=spread_short_asset,
        use_dv01_neutral=use_dv01_neutral,
        forward_horizon=forward_horizon
    )

    bt = add_walk_forward_thresholds(
        bt,
        threshold_grid=threshold_grid,
        lookback_window=lookback_window,
        rebalance_frequency=rebalance_frequency,
        strategy_type=strategy_type,
        signal_col="Signal_Input",
        forward_return_col="Forward_Objective_Return"
    )

    bt = generate_raw_signal(
        bt,
        sizing=sizing,
        strategy_type=strategy_type,
        signal_col="Signal_Input"
    )

    bt = apply_signal_persistence(
        bt,
        holding_period=holding_period
    )

    if strategy_type == "directional":

        if use_dv01_neutral:
            raise ValueError("use_dv01_neutral=True only applies to spread strategies.")

        bt = construct_directional_returns(bt)

    elif strategy_type == "spread":

        bt = construct_spread_returns(
            bt,
            spread_short_asset=spread_short_asset,
            use_dv01_neutral=use_dv01_neutral
        )

    else:
        raise ValueError("strategy_type must be 'directional' or 'spread'.")

    if use_vol_target:

        bt = apply_volatility_targeting(
            bt,
            target_vol=target_vol,
            vol_window=vol_window,
            max_leverage=max_leverage
        )

        if strategy_type == "directional":
            bt = construct_directional_returns(bt)

        elif strategy_type == "spread":
            bt = construct_spread_returns(
                bt,
                spread_short_asset=spread_short_asset,
                use_dv01_neutral=use_dv01_neutral
            )

    bt = apply_transaction_costs(
        bt,
        transaction_cost_bps=transaction_cost_bps,
        spread_short_asset=spread_short_asset if strategy_type == "spread" else None
    )

    keep_cols = [
        "TLT_Return",
        "Benchmark_Return",
        "Benchmark_Name",
        "Forward_Objective_Return",
        "Gross_Strategy_Return",
        "Strategy_Return",
        "Strategy_Position",
        "Unlagged_Strategy_Position",
        "Signal_Position",
        "Signal_10D_Yield_Change",
        "Signal_10Y_2Y_Curve_Level",
        "Signal_Input",
        "US10Y_2Y_Spread",
        "Selected_Threshold",
        "Selected_Upper_Threshold",
        "Selected_Lower_Threshold",
        "Turnover",
        "Transaction_Cost"
    ]

    optional_cols = [
        "Spread_Return",
        "Spread_Long_Weight",
        "Spread_Short_Weight",
        "TLT_Final_Weight",
        f"{spread_short_asset}_Final_Weight",
        "Objective_Long_Weight",
        "Objective_Short_Weight",
        "TLT_Duration",
        f"{spread_short_asset}_Duration",
        "Vol_Estimate",
        "Vol_Target_Multiplier"
    ]

    keep_cols += [col for col in optional_cols if col in bt.columns]

    keep_cols = [col for col in keep_cols if col in bt.columns]
    bt = bt[keep_cols].dropna().copy()

    bt = compute_wealth(
        bt,
        initial_wealth=initial_wealth
    )

    performance_summary = create_performance_summary(
        bt,
        strategy_name=display_name
    )

    exposure_summary = create_exposure_summary(
        bt,
        holding_period=holding_period
    )

    trade_summary = create_trade_summary(bt)

    if save_outputs:

        fname = clean_filename(strategy_name)

        bt.to_csv(table_path / f"{fname}_timeseries.csv", index=True)

        performance_summary.to_csv(
            table_path / f"{fname}_performance_summary.csv",
            index=False
        )

        exposure_summary.to_csv(
            table_path / f"{fname}_exposure_summary.csv",
            index=False
        )

        performance_summary.to_latex(
            table_path / f"{fname}_performance_summary.tex",
            index=False,
            escape=False,
            float_format="%.4f"
        )

        exposure_summary.to_latex(
            table_path / f"{fname}_exposure_summary.tex",
            index=False,
            escape=True,
            float_format="%.4f"
        )

        trade_summary.to_csv(
            table_path / f"{fname}_trade_summary.csv",
            index=False
            )

        trade_summary.to_latex(
            table_path / f"{fname}_trade_summary.tex",
            index=False,
            escape=True,
            float_format="%.4f"
        )

    if make_plots:
        plot_backtest_results(
            bt,
            strategy_name=display_name,
            fig_path=fig_path,
            filename_name=strategy_name
        )

    print("\nPerformance summary:")
    print(performance_summary)

    print("\nExposure summary:")
    print(exposure_summary)
    
    print("\nTrade summary:")
    print(trade_summary)

    return {
        "name": strategy_name,
        "display_name": display_name,
        "timeseries": bt,
        "performance": performance_summary,
        "exposure": exposure_summary,
        "trade": trade_summary
    }

# =====================================================
# 13. RUN ONE MAIN STRATEGY
# =====================================================

#spread_base_result = run_backtest(
    data=eda,
    strategy_type="spread",
    sizing="scaled",
    spread_short_asset="IEF",
    use_dv01_neutral=True,
    duration_window=252,
    use_vol_target=True,
    target_vol=0.04,
    vol_window=63,
    max_leverage=3.0,
    transaction_cost_bps=1.0,
    holding_period=holding_period,
    threshold_grid={
        "upper": spread_upper_threshold_grid,
        "lower": spread_lower_threshold_grid
    },
    lookback_window=63,
    rebalance_frequency=3,
    save_outputs=True,
    make_plots=True
#)

# =====================================================
# COMBINED STRATEGY COMPARISON TABLE
# RAW LATEX TABULAR ONLY
# =====================================================

strategy_files = {
    "Buy-and-hold TLT": None,
    "Directional Binary": "Directional_TLT_Binary_TCost_1bps",
    "Directional Scaled": "Directional_TLT_Scaled_TCost_1bps",
    "Directional Scaled + Vol Target": "Directional_TLT_Scaled_VolTarget_TCost_1bps",

    # placed before all active spread variants
    "Buy-and-hold TLT--IEF": "Spread_TLT_IEF_Binary_TCost_1bps",

    "Spread Binary": "Spread_TLT_IEF_Binary_TCost_1bps",
    "Spread Scaled": "Spread_TLT_IEF_Scaled_TCost_1bps",
    "Spread Scaled + Vol Target": "Spread_TLT_IEF_Scaled_VolTarget_TCost_1bps",
    "Spread DV01-Neutral + Vol Target": "Spread_TLT_IEF_Scaled_DV01Neutral_VolTarget_TCost_1bps",
}

def load_strategy_performance(strategy_label, file_prefix):
    perf_path = TABLE_PATH / f"{file_prefix}_performance_summary.csv"
    perf = pd.read_csv(perf_path)

    # For active strategy rows, use first strategy column after Metric
    strategy_col = perf.columns[1]
    row = perf.set_index("Metric")[strategy_col].copy()
    row.name = strategy_label

    return row

def load_benchmark_performance(strategy_label, file_prefix):
    perf_path = TABLE_PATH / f"{file_prefix}_performance_summary.csv"
    perf = pd.read_csv(perf_path)

    # For benchmark rows, use second strategy column after Metric
    benchmark_col = perf.columns[2]
    row = perf.set_index("Metric")[benchmark_col].copy()
    row.name = strategy_label

    return row

combined_rows = []

# Buy-and-hold TLT benchmark
combined_rows.append(
    load_benchmark_performance(
        "Buy-and-hold TLT",
        "Directional_TLT_Binary_TCost_1bps"
    )
)

# Directional active strategies
for label, prefix in strategy_files.items():
    if label == "Buy-and-hold TLT":
        continue
    if label == "Buy-and-hold TLT--IEF":
        combined_rows.append(
            load_benchmark_performance(label, prefix)
        )
    else:
        combined_rows.append(
            load_strategy_performance(label, prefix)
        )

combined_table = pd.DataFrame(combined_rows)

metric_order = [
    "Total return (\\%)",
    "Annualised return (\\%)",
    "Annualised volatility (\\%)",
    "Sharpe ratio",
    "Maximum drawdown (\\%)",
    "Calmar ratio",
    "Daily VaR 95\\% (\\%)",
    "Daily ES 95\\% (\\%)"
]

combined_table = combined_table[metric_order]

combined_table = combined_table.astype(float).round(4)

combined_table = (
    combined_table
    .reset_index()
    .rename(columns={"index": "Strategy"})
)

combined_table.to_csv(
    TABLE_PATH / "combined_strategy_comparison.csv",
    index=False
)

latex_table = combined_table.to_latex(
    index=False,
    escape=False,
    float_format="%.4f",
    column_format="lrrrrrrrr"
)

with open(TABLE_PATH / "combined_strategy_comparison.tex", "w") as f:
    f.write(latex_table)

print(combined_table)

# =====================================================
# COMBINED EQUITY CURVE AND DRAWDOWN PLOTS
# =====================================================

plot_strategies = {
    "Buy-and-hold TLT": {
        "file": "Directional_TLT_Binary_TCost_1bps_timeseries.csv",
        "wealth_col": "Benchmark_Wealth",
        "drawdown_col": "Benchmark_Drawdown"
    },

    "Directional | Binary": {
        "file": "Directional_TLT_Binary_TCost_1bps_timeseries.csv",
        "wealth_col": "Strategy_Wealth",
        "drawdown_col": "Strategy_Drawdown"
    },

    "Directional | Scaled + Vol Target": {
        "file": "Directional_TLT_Scaled_VolTarget_TCost_1bps_timeseries.csv",
        "wealth_col": "Strategy_Wealth",
        "drawdown_col": "Strategy_Drawdown"
    },

    "Spread | Binary": {
        "file": "Spread_TLT_IEF_Binary_TCost_1bps_timeseries.csv",
        "wealth_col": "Strategy_Wealth",
        "drawdown_col": "Strategy_Drawdown"
    },

    "Spread | Scaled + Vol Target": {
        "file": "Spread_TLT_IEF_Scaled_VolTarget_TCost_1bps_timeseries.csv",
        "wealth_col": "Strategy_Wealth",
        "drawdown_col": "Strategy_Drawdown"
    },

    "Spread | DV01-Neutral + Vol Target": {
        "file": "Spread_TLT_IEF_Scaled_DV01Neutral_VolTarget_TCost_1bps_timeseries.csv",
        "wealth_col": "Strategy_Wealth",
        "drawdown_col": "Strategy_Drawdown"
    },
}


def load_timeseries(file_name):
    path = TABLE_PATH / file_name
    temp = pd.read_csv(path, index_col=0, parse_dates=True)
    return temp


# =====================================================
# CLEAN COMBINED EQUITY CURVE AND DRAWDOWN PLOTS
# Best directional + best spread + benchmarks
# =====================================================

plot_strategies_clean = {
    "Buy-and-hold TLT": {
        "file": "Directional_TLT_Binary_TCost_1bps_timeseries.csv",
        "wealth_col": "Benchmark_Wealth",
        "drawdown_col": "Benchmark_Drawdown",
        "linewidth": 1.4
    },

    "Buy-and-hold TLT--IEF spread": {
        "file": "Spread_TLT_IEF_Binary_TCost_1bps_timeseries.csv",
        "wealth_col": "Benchmark_Wealth",
        "drawdown_col": "Benchmark_Drawdown",
        "linewidth": 1.4
    },

    "Best directional: Binary": {
        "file": "Directional_TLT_Binary_TCost_1bps_timeseries.csv",
        "wealth_col": "Strategy_Wealth",
        "drawdown_col": "Strategy_Drawdown",
        "linewidth": 2.0
    },

    "Best spread: Scaled + Vol Target": {
        "file": "Spread_TLT_IEF_Scaled_VolTarget_TCost_1bps_timeseries.csv",
        "wealth_col": "Strategy_Wealth",
        "drawdown_col": "Strategy_Drawdown",
        "linewidth": 2.0
    },
}


def load_timeseries(file_name):
    path = TABLE_PATH / file_name
    return pd.read_csv(path, index_col=0, parse_dates=True)


# =====================================================
# 1. CLEAN COMBINED EQUITY CURVE
# =====================================================

plt.figure(figsize=(11, 6))

for label, spec in plot_strategies_clean.items():
    temp = load_timeseries(spec["file"])

    wealth = temp[spec["wealth_col"]].dropna()
    wealth = wealth / wealth.iloc[0]

    plt.plot(
        wealth.index,
        wealth,
        linewidth=spec["linewidth"],
        label=label
    )

plt.title("Strategy Wealth Comparison: Best Directional and Spread Specifications")
plt.xlabel("Date")
plt.ylabel("Wealth index")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_PATH / "clean_strategy_equity_curve_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# 2. CLEAN COMBINED DRAWDOWN PLOT
# =====================================================

plt.figure(figsize=(11, 6))

for label, spec in plot_strategies_clean.items():
    temp = load_timeseries(spec["file"])

    drawdown = temp[spec["drawdown_col"]].dropna() * 100

    plt.plot(
        drawdown.index,
        drawdown,
        linewidth=spec["linewidth"],
        label=label
    )

plt.title("Strategy Drawdown Comparison: Best Directional and Spread Specifications")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_PATH / "clean_strategy_drawdown_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# RISK ANALYSIS, STRESS TESTING AND BOOTSTRAP ROBUSTNESS
# Best strategy:
# Spread_TLT_IEF_Scaled_VolTarget_TCost_1bps
# NON-DV01 version
# =====================================================

# =====================================================
# 0. LOAD / SET BEST STRATEGY
# =====================================================

best_file_name = "Spread_TLT_IEF_Scaled_VolTarget_TCost_1bps"
best_name = "TLT--IEF Spread | Continuous Position Sizing | Volatility Targeted"

best_bt = pd.read_csv(
    TABLE_PATH / f"{best_file_name}_timeseries.csv",
    index_col=0,
    parse_dates=True
)

# =====================================================
# 0.1 LATEX TABLE HELPER
# =====================================================

def save_clean_latex_table(df, path, column_format=None):
    latex_table = df.to_latex(
        index=False,
        escape=True,
        float_format="%.4f",
        column_format=column_format
    )

    with open(path, "w") as f:
        f.write(latex_table)


# =====================================================
# 0.2 ADD DURATIONS FOR STRESS TESTING
# =====================================================

duration_data = add_empirical_durations(
    prepare_backtest_data(eda),
    duration_window=252
)

best_bt = best_bt.join(
    duration_data[["TLT_Duration", "IEF_Duration"]],
    how="left"
)

best_bt[["TLT_Duration", "IEF_Duration"]] = (
    best_bt[["TLT_Duration", "IEF_Duration"]]
    .ffill()
    .bfill()
)

# =====================================================
# 1. AVERAGE-ACTIVE YIELD / CURVE SHOCK STRESS TESTING
# =====================================================

def average_active_exposures(bt):
    temp = bt.copy()
    active = temp["Strategy_Position"].abs() > 0

    if active.sum() == 0:
        active = temp.index == temp.index[-1]

    return {
        "TLT weight": temp.loc[active, "TLT_Final_Weight"].mean(),
        "IEF weight": temp.loc[active, "IEF_Final_Weight"].mean(),
        "TLT duration": temp.loc[active, "TLT_Duration"].mean(),
        "IEF duration": temp.loc[active, "IEF_Duration"].mean(),
        "Vol target multiplier": (
            temp.loc[active, "Vol_Target_Multiplier"].mean()
            if "Vol_Target_Multiplier" in temp.columns
            else np.nan
        )
    }


def run_yield_curve_stress_tests(
    bt,
    shock_scenarios=None,
    initial_wealth=1.0
):
    exp = average_active_exposures(bt)

    if shock_scenarios is None:
        shock_scenarios = {
            "Parallel +50 bps": {"TLT_yield_shock_bps": 50, "IEF_yield_shock_bps": 50},
            "Parallel +100 bps": {"TLT_yield_shock_bps": 100, "IEF_yield_shock_bps": 100},
            "Parallel +150 bps": {"TLT_yield_shock_bps": 150, "IEF_yield_shock_bps": 150},
            "Parallel -50 bps": {"TLT_yield_shock_bps": -50, "IEF_yield_shock_bps": -50},
            "Parallel -100 bps": {"TLT_yield_shock_bps": -100, "IEF_yield_shock_bps": -100},
            "Parallel -150 bps": {"TLT_yield_shock_bps": -150, "IEF_yield_shock_bps": -150},
            "Bear steepening": {"TLT_yield_shock_bps": 125, "IEF_yield_shock_bps": 50},
            "Bull steepening": {"TLT_yield_shock_bps": -125, "IEF_yield_shock_bps": -50},
            "Bear flattening": {"TLT_yield_shock_bps": 50, "IEF_yield_shock_bps": 125},
            "Bull flattening": {"TLT_yield_shock_bps": -50, "IEF_yield_shock_bps": -125},
            "Long-end selloff": {"TLT_yield_shock_bps": 150, "IEF_yield_shock_bps": 25},
            "Long-end rally": {"TLT_yield_shock_bps": -150, "IEF_yield_shock_bps": -25},
        }

    rows = []

    for scenario, shocks in shock_scenarios.items():
        dy_tlt = shocks["TLT_yield_shock_bps"] / 10000
        dy_ief = shocks["IEF_yield_shock_bps"] / 10000

        tlt_return = -exp["TLT duration"] * dy_tlt
        ief_return = -exp["IEF duration"] * dy_ief

        strategy_return = (
            exp["TLT weight"] * tlt_return
            + exp["IEF weight"] * ief_return
        )

        rows.append({
            "Scenario": scenario,
            "TLT shock (bps)": shocks["TLT_yield_shock_bps"],
            "IEF shock (bps)": shocks["IEF_yield_shock_bps"],
            "TLT duration": exp["TLT duration"],
            "IEF duration": exp["IEF duration"],
            "TLT weight": exp["TLT weight"],
            "IEF weight": exp["IEF weight"],
            "Estimated TLT return (%)": tlt_return * 100,
            "Estimated IEF return (%)": ief_return * 100,
            "Estimated strategy return (%)": strategy_return * 100,
            "Estimated wealth after shock": initial_wealth * (1 + strategy_return)
        })

    stress_table = pd.DataFrame(rows)
    numeric_cols = stress_table.columns.drop("Scenario")
    stress_table[numeric_cols] = stress_table[numeric_cols].astype(float).round(4)

    return stress_table


stress_table = run_yield_curve_stress_tests(
    best_bt,
    initial_wealth=1.0
)

stress_table.to_csv(
    TABLE_PATH / f"{best_file_name}_yield_curve_stress_tests.csv",
    index=False
)

save_clean_latex_table(
    stress_table,
    TABLE_PATH / f"{best_file_name}_yield_curve_stress_tests.tex",
    column_format="lrrrrrrrrrr"
)

print("\nYield curve stress-test table:")
print(stress_table)


plt.figure(figsize=(10, 5))
plt.bar(
    stress_table["Scenario"],
    stress_table["Estimated strategy return (%)"],
    alpha=0.85
)
plt.axhline(0, color="black", linewidth=1)
plt.title("Stress Test: Estimated Strategy Return Under Yield Curve Shocks")
plt.xlabel("Scenario")
plt.ylabel("Estimated one-off strategy return (%)")
plt.xticks(rotation=35, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIG_PATH / f"{best_file_name}_yield_curve_stress_test_barplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# =====================================================
# 2. HISTORICAL STRESS EVENT / REGIME SUBPERIOD TABLE
# TRANSPOSED FORMAT: METRICS AS ROWS, PERIODS AS COLUMNS
# =====================================================

def subperiod_performance_metrics(bt, start, end, label):
    temp = bt.loc[start:end].copy()

    if len(temp) < 30:
        return None

    strategy_wealth = (1 + temp["Strategy_Return"]).cumprod()
    benchmark_wealth = (1 + temp["Benchmark_Return"]).cumprod()

    strategy_metrics = performance_metrics(
        temp["Strategy_Return"],
        strategy_wealth,
        label
    )

    benchmark_metrics = performance_metrics(
        temp["Benchmark_Return"],
        benchmark_wealth,
        "Benchmark"
    )

    active = temp["Strategy_Position"].abs() > 0

    return {
        "Period": label,
        "Observations": len(temp),
        "Strategy total return (%)": strategy_metrics["Total return (\\%)"],
        "Strategy annualised return (%)": strategy_metrics["Annualised return (\\%)"],
        "Strategy annualised volatility (%)": strategy_metrics["Annualised volatility (\\%)"],
        "Strategy Sharpe ratio": strategy_metrics["Sharpe ratio"],
        "Strategy maximum drawdown (%)": strategy_metrics["Maximum drawdown (\\%)"],
        "Strategy daily VaR 95% (%)": strategy_metrics["Daily VaR 95\\% (\\%)"],
        "Strategy daily ES 95% (%)": strategy_metrics["Daily ES 95\\% (\\%)"],
        "Benchmark total return (%)": benchmark_metrics["Total return (\\%)"],
        "Benchmark Sharpe ratio": benchmark_metrics["Sharpe ratio"],
        "Benchmark maximum drawdown (%)": benchmark_metrics["Maximum drawdown (\\%)"],
        "Active trading frequency": active.mean(),
        "Average absolute position": temp["Strategy_Position"].abs().mean(),
        "Average daily turnover": temp["Turnover"].mean(),
        "Average transaction cost (%)": temp["Transaction_Cost"].mean() * 100,
        "Total transaction cost (%)": temp["Transaction_Cost"].sum() * 100
    }


stress_periods = [
    ("Global Financial Crisis", "2008-01-01", "2009-12-31"),
    ("Eurozone / low-rate regime", "2010-01-01", "2013-12-31"),
    ("Pre-COVID late cycle", "2014-01-01", "2019-12-31"),
    ("COVID shock", "2020-01-01", "2020-12-31"),
    ("Inflation and Fed tightening", "2021-01-01", "2022-12-31"),
    ("Higher-for-longer regime", "2023-01-01", "2026-12-31"),
]

subperiod_rows = []

for label, start, end in stress_periods:
    row = subperiod_performance_metrics(
        best_bt,
        start=start,
        end=end,
        label=label
    )

    if row is not None:
        subperiod_rows.append(row)

subperiod_table_raw = pd.DataFrame(subperiod_rows)

numeric_cols = subperiod_table_raw.columns.drop("Period")
subperiod_table_raw[numeric_cols] = (
    subperiod_table_raw[numeric_cols]
    .astype(float)
    .round(4)
)

subperiod_table = (
    subperiod_table_raw
    .set_index("Period")
    .T
    .reset_index()
    .rename(columns={"index": "Metric"})
)

subperiod_table.to_csv(
    TABLE_PATH / f"{best_file_name}_historical_stress_periods_transposed.csv",
    index=False
)

save_clean_latex_table(
    subperiod_table,
    TABLE_PATH / f"{best_file_name}_historical_stress_periods.tex",
    column_format="lrrrrrr"
)

print("\nHistorical stress-period table, transposed:")
print(subperiod_table)


# =====================================================
# 3. PARAMETRIC VAR BOOTSTRAP BACKTEST
# =====================================================

def estimate_var1(x):
    x = np.asarray(x)
    y = x[1:]
    x_lag = x[:-1]

    x_design = np.column_stack([np.ones(len(x_lag)), x_lag])
    beta = np.linalg.lstsq(x_design, y, rcond=None)[0]

    c = beta[0]
    A = beta[1:].T

    resid = y - x_design @ beta
    sigma = np.cov(resid.T)

    return c, A, sigma, resid


def simulate_var1_paths(
    initial_state,
    c,
    A,
    sigma,
    n_steps,
    n_paths=50,
    random_seed=42,
    use_student_t=True,
    df_t=5
):
    rng = np.random.default_rng(random_seed)

    k = len(initial_state)
    paths = np.zeros((n_paths, n_steps, k))
    paths[:, 0, :] = initial_state

    for p in range(n_paths):
        for t in range(1, n_steps):

            if use_student_t:
                z = rng.multivariate_normal(np.zeros(k), sigma)
                scale = np.sqrt(df_t / rng.chisquare(df_t))
                eps = z * scale
            else:
                eps = rng.multivariate_normal(np.zeros(k), sigma)

            paths[p, t, :] = c + A @ paths[p, t - 1, :] + eps
            paths[p, t, :] = np.clip(paths[p, t, :], 0.0, 10.0)

    return paths


def estimate_duration_level(original_df, ticker, y_col):
    temp = original_df.copy()
    ret_col = f"{ticker}_Ret"

    beta = (
        temp[ret_col].rolling(252).cov(temp[y_col])
        / temp[y_col].rolling(252).var()
    )

    duration = (-100 * beta).replace([np.inf, -np.inf], np.nan).dropna()
    duration = duration[(duration > 0.1) & (duration < 40)]

    if len(duration) > 20:
        return float(duration.median())

    fallback = {"TLT": 17.0, "IEF": 7.0, "SHY": 2.0}
    return fallback[ticker]


def create_synthetic_dataset_from_yields(
    original_df,
    simulated_yields,
    tlt_duration,
    ief_duration,
    start_index=None,
    random_seed=None
):
    n_steps = simulated_yields.shape[0]

    if start_index is None:
        index = original_df.index[-n_steps:]
    else:
        index = start_index[:n_steps]

    sim = pd.DataFrame(index=index)

    sim["US10Y_Yield"] = simulated_yields[:, 0]
    sim["US2Y_Yield"] = simulated_yields[:, 1]

    sim["US10Y_Change_pctpts"] = sim["US10Y_Yield"].diff()
    sim["US2Y_Change_pctpts"] = sim["US2Y_Yield"].diff()

    sim["US10Y_Change_bps"] = sim["US10Y_Change_pctpts"] * 100
    sim["US2Y_Change_bps"] = sim["US2Y_Change_pctpts"] * 100

    sim["US3M_Yield"] = original_df["US3M_Yield"].median()
    sim["US3M_Change_pctpts"] = 0.0
    sim["US3M_Change_bps"] = 0.0
    sim["CPI"] = original_df["CPI"].median()
    sim["CPI_YoY"] = original_df["CPI_YoY"].median()

    dy10_decimal = sim["US10Y_Change_pctpts"] / 100
    dy2_decimal = sim["US2Y_Change_pctpts"] / 100

    sim["TLT_Ret"] = -tlt_duration * dy10_decimal
    sim["IEF_Ret"] = -ief_duration * dy2_decimal
    sim["SHY_Ret"] = -2.0 * dy2_decimal

    hist = original_df.copy()

    hist_dy10 = hist["US10Y_Change_pctpts"] / 100
    hist_dy2 = hist["US2Y_Change_pctpts"] / 100

    tlt_resid = hist["TLT_Ret"] - (-tlt_duration * hist_dy10)
    ief_resid = hist["IEF_Ret"] - (-ief_duration * hist_dy2)

    resid_data = pd.concat([tlt_resid, ief_resid], axis=1).dropna()
    resid_cov = np.cov(resid_data.T)

    rng = np.random.default_rng(random_seed)

    noise = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=resid_cov,
        size=n_steps
    )

    sim["TLT_Ret"] = sim["TLT_Ret"].fillna(0.0) + noise[:, 0]
    sim["IEF_Ret"] = sim["IEF_Ret"].fillna(0.0) + noise[:, 1]
    sim["SHY_Ret"] = sim["SHY_Ret"].fillna(0.0)

    sim["TLT_AdjClose"] = 100 * (1 + sim["TLT_Ret"]).cumprod()
    sim["IEF_AdjClose"] = 100 * (1 + sim["IEF_Ret"]).cumprod()
    sim["SHY_AdjClose"] = 100 * (1 + sim["SHY_Ret"]).cumprod()

    sim["US10Y_3M_Spread"] = sim["US10Y_Yield"] - sim["US3M_Yield"]
    sim["US10Y_2Y_Spread"] = sim["US10Y_Yield"] - sim["US2Y_Yield"]

    sim["US10Y_3M_Spread_Change_bps"] = sim["US10Y_3M_Spread"].diff() * 100
    sim["US10Y_2Y_Spread_Change_bps"] = sim["US10Y_2Y_Spread"].diff() * 100

    sim["TLT_minus_IEF_Ret"] = sim["TLT_Ret"] - sim["IEF_Ret"]
    sim["TLT_minus_SHY_Ret"] = sim["TLT_Ret"] - sim["SHY_Ret"]
    sim["IEF_minus_SHY_Ret"] = sim["IEF_Ret"] - sim["SHY_Ret"]

    sim = sim.replace([np.inf, -np.inf], np.nan).dropna().copy()

    return sim


def run_parametric_var_bootstrap(
    original_df,
    n_paths=50,
    n_steps=None,
    random_seed=42,
    use_student_t=True,
    make_individual_plots=False
):
    if n_steps is None:
        n_steps = len(original_df)

    var_data = original_df[["US10Y_Yield", "US2Y_Yield"]].dropna().copy()

    c, A, sigma, resid = estimate_var1(var_data.values)
    initial_state = var_data.iloc[-1].values

    simulated_paths = simulate_var1_paths(
        initial_state=initial_state,
        c=c,
        A=A,
        sigma=sigma,
        n_steps=n_steps,
        n_paths=n_paths,
        random_seed=random_seed,
        use_student_t=use_student_t
    )

    tlt_duration = estimate_duration_level(
        original_df,
        ticker="TLT",
        y_col="US10Y_Change_pctpts"
    )

    ief_duration = estimate_duration_level(
        original_df,
        ticker="IEF",
        y_col="US2Y_Change_pctpts"
    )

    bootstrap_rows = []
    wealth_paths = []

    for i in range(n_paths):
        print(f"Running bootstrap path {i + 1}/{n_paths}")

        sim_df = create_synthetic_dataset_from_yields(
            original_df=original_df,
            simulated_yields=simulated_paths[i],
            tlt_duration=tlt_duration,
            ief_duration=ief_duration,
            start_index=original_df.index[-n_steps:],
            random_seed=random_seed + i
        )

        try:
            result = run_backtest(
                data=sim_df,
                strategy_type="spread",
                sizing="scaled",
                spread_short_asset="IEF",
                use_dv01_neutral=False,
                use_vol_target=True,
                target_vol=0.04,
                vol_window=63,
                max_leverage=3.0,
                transaction_cost_bps=1.0,
                holding_period=holding_period,
                threshold_grid={
                    "upper": spread_upper_threshold_grid,
                    "lower": spread_lower_threshold_grid
                },
                lookback_window=63,
                rebalance_frequency=3,
                save_outputs=False,
                make_plots=make_individual_plots
            )

            bt_i = result["timeseries"].copy()

            metrics_i = performance_metrics(
                bt_i["Strategy_Return"],
                bt_i["Strategy_Wealth"],
                f"Bootstrap path {i + 1}"
            )

            row = {
                "Path": i + 1,
                "Terminal wealth": bt_i["Strategy_Wealth"].iloc[-1],
                "Total return (%)": metrics_i["Total return (\\%)"],
                "Annualised return (%)": metrics_i["Annualised return (\\%)"],
                "Annualised volatility (%)": metrics_i["Annualised volatility (\\%)"],
                "Sharpe ratio": metrics_i["Sharpe ratio"],
                "Maximum drawdown (%)": metrics_i["Maximum drawdown (\\%)"],
                "Calmar ratio": metrics_i["Calmar ratio"],
                "Daily VaR 95% (%)": metrics_i["Daily VaR 95\\% (\\%)"],
                "Daily ES 95% (%)": metrics_i["Daily ES 95\\% (\\%)"],
                "Average turnover": bt_i["Turnover"].mean(),
                "Total transaction cost (%)": bt_i["Transaction_Cost"].sum() * 100
            }

            bootstrap_rows.append(row)
            wealth_paths.append(bt_i["Strategy_Wealth"].reset_index(drop=True))

        except Exception as e:
            print(f"Bootstrap path {i + 1} failed: {e}")

    bootstrap_results = pd.DataFrame(bootstrap_rows)

    return bootstrap_results, wealth_paths, {
        "VAR constant": c,
        "VAR A matrix": A,
        "VAR covariance": sigma,
        "TLT duration used": tlt_duration,
        "IEF duration used": ief_duration
    }


bootstrap_results, bootstrap_wealth_paths, bootstrap_model_info = run_parametric_var_bootstrap(
    original_df=eda,
    n_paths=50,
    n_steps=len(eda),
    random_seed=42,
    use_student_t=True,
    make_individual_plots=False
)

bootstrap_results_rounded = bootstrap_results.copy()

for col in bootstrap_results_rounded.columns.drop("Path"):
    bootstrap_results_rounded[col] = (
        bootstrap_results_rounded[col]
        .astype(float)
        .round(4)
    )

bootstrap_results_rounded.to_csv(
    TABLE_PATH / f"{best_file_name}_parametric_var_bootstrap_paths.csv",
    index=False
)

save_clean_latex_table(
    bootstrap_results_rounded,
    TABLE_PATH / f"{best_file_name}_parametric_var_bootstrap_paths.tex",
    column_format="rrrrrrrrrrrr"
)

print("\nParametric VAR bootstrap path-level results:")
print(bootstrap_results_rounded)


# =====================================================
# 4. BOOTSTRAP SUMMARY TABLE
# =====================================================

def summarise_bootstrap_results(bootstrap_results):
    metric_cols = bootstrap_results.columns.drop("Path")
    rows = []

    for col in metric_cols:
        x = bootstrap_results[col].dropna()

        rows.append({
            "Metric": col,
            "Mean": x.mean(),
            "Median": x.median(),
            "Std. dev.": x.std(),
            "5th percentile": x.quantile(0.05),
            "95th percentile": x.quantile(0.95),
            "Probability > 0": (
                (x > 0).mean()
                if "return" in col.lower() or "sharpe" in col.lower()
                else np.nan
            )
        })

    summary = pd.DataFrame(rows)

    numeric_cols = summary.columns.drop("Metric")
    summary[numeric_cols] = summary[numeric_cols].astype(float).round(4)

    return summary


bootstrap_summary = summarise_bootstrap_results(bootstrap_results)

bootstrap_summary.to_csv(
    TABLE_PATH / f"{best_file_name}_parametric_var_bootstrap_summary.csv",
    index=False
)

save_clean_latex_table(
    bootstrap_summary,
    TABLE_PATH / f"{best_file_name}_parametric_var_bootstrap_summary.tex",
    column_format="lrrrrrr"
)

print("\nParametric VAR bootstrap summary:")
print(bootstrap_summary)


# =====================================================
# 5. BOOTSTRAP PLOTS
# =====================================================

if len(bootstrap_wealth_paths) > 0:

    wealth_matrix = pd.concat(bootstrap_wealth_paths, axis=1)
    wealth_matrix.columns = [
        f"Path_{i+1}" for i in range(wealth_matrix.shape[1])
    ]

    wealth_mean = wealth_matrix.mean(axis=1)
    wealth_p05 = wealth_matrix.quantile(0.05, axis=1)
    wealth_p95 = wealth_matrix.quantile(0.95, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(
        wealth_mean.index,
        wealth_mean,
        label="Mean bootstrap wealth",
        linewidth=1.8
    )
    plt.fill_between(
        wealth_mean.index,
        wealth_p05,
        wealth_p95,
        alpha=0.25,
        label="5%--95% interval"
    )
    plt.axhline(1.0, color="black", linewidth=1)
    plt.title("Parametric VAR Bootstrap: Wealth Paths")
    plt.xlabel("Time step")
    plt.ylabel("Wealth index")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIG_PATH / f"{best_file_name}_parametric_var_bootstrap_wealth_paths.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(
        bootstrap_results["Terminal wealth"].dropna(),
        bins=15,
        alpha=0.75
    )
    plt.axvline(1.0, color="black", linewidth=1)
    plt.title("Parametric VAR Bootstrap: Terminal Wealth Distribution")
    plt.xlabel("Terminal wealth")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIG_PATH / f"{best_file_name}_parametric_var_bootstrap_terminal_wealth.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(
        bootstrap_results["Sharpe ratio"].dropna(),
        bins=15,
        alpha=0.75
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Parametric VAR Bootstrap: Sharpe Ratio Distribution")
    plt.xlabel("Sharpe ratio")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIG_PATH / f"{best_file_name}_parametric_var_bootstrap_sharpe_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()