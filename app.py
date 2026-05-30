# ============================================================
# Financial Intelligence Platform — Flask Version
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import time, io, json
from flask import Flask, render_template, request, jsonify, send_file
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_OK = True
except ImportError:
    ARIMA_OK = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(float(obj)) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj.date()) if hasattr(obj, 'date') else str(obj)
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = NpEncoder

# ============================================================
# SIMPLE TTL CACHE (replaces @st.cache_data)
# ============================================================

_CACHE: dict = {}

def _cached(key: str, fn, ttl: int = 300):
    now = time.time()
    if key in _CACHE and now - _CACHE[key][1] < ttl:
        return _CACHE[key][0]
    result = fn()
    _CACHE[key] = (result, now)
    return result

# ============================================================
# CONSTANTS
# ============================================================

TRADING_DAYS = 252

BENCHMARK_INDICES = {
    "S&P 500 (^GSPC)":          "^GSPC",
    "Dow Jones (^DJI)":         "^DJI",
    "NASDAQ Composite (^IXIC)": "^IXIC",
    "Russell 2000 (^RUT)":      "^RUT",
    "FTSE 100 (^FTSE)":         "^FTSE",
    "DAX (^GDAXI)":             "^GDAXI",
    "Nikkei 225 (^N225)":       "^N225",
    "Hang Seng (^HSI)":         "^HSI",
    "MSCI World ETF (URTH)":    "URTH",
    "MSCI Emerging (EEM)":      "EEM",
    "Total Market (VTI)":       "VTI",
}

POPULAR = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B",
    "JPM","V","JNJ","UNH","PG","HD","MA","BAC","XOM","CVX",
    "PFE","KO","PEP","WMT","MCD","DIS","NFLX","INTC","AMD","PYPL",
    "SPY","QQQ","IWM","GLD","SLV","BTC-USD","ETH-USD",
    "^MXX","^BVSP","^MERV","EWW","EWZ",
]

PLOTLY_COLORS = [
    "#00e87a","#ffd700","#4da6ff","#ff6b35","#da70d6",
    "#ff4545","#40e0d0","#ff69b4","#c0c020","#20c0c0",
]

_ASSET_DEFAULTS = ["AAPL", "MSFT", "NVDA"]

# ============================================================
# UTILITY
# ============================================================

def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _date_str(d) -> str:
    if hasattr(d, "date"):
        return str(d.date())
    return str(d)

def _safe(v):
    """Return None for NaN/Inf, else the value."""
    try:
        if v is None: return None
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return None

# ============================================================
# FINANCIAL MATH  (identical to Streamlit version)
# ============================================================

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha["HA_Close"].iloc[i-1]) / 2)
    ha["HA_Open"]  = ha_open
    ha["HA_High"]  = pd.concat([df["High"], ha["HA_Open"], ha["HA_Close"]], axis=1).max(axis=1)
    ha["HA_Low"]   = pd.concat([df["Low"],  ha["HA_Open"], ha["HA_Close"]], axis=1).min(axis=1)
    ha["Volume"]   = df["Volume"]
    return ha

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    sma = series.rolling(period).mean()
    sd  = series.rolling(period).std()
    return sma + std_mult * sd, sma, sma - std_mult * sd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    m = ema(series, fast) - ema(series, slow)
    s = ema(m, signal)
    return m, s, m - s

def ann_return(ret: pd.Series) -> float:
    if len(ret) < 2: return np.nan
    return float((1 + ret.mean()) ** TRADING_DAYS - 1)

def ann_vol(ret: pd.Series) -> float:
    if len(ret) < 2: return np.nan
    return float(ret.std() * np.sqrt(TRADING_DAYS))

def sharpe(ret: pd.Series, rfr: float = 0.05) -> float:
    excess = ret - rfr / TRADING_DAYS
    sd = excess.std()
    if sd == 0 or np.isnan(sd): return np.nan
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))

def sortino(ret: pd.Series, rfr: float = 0.05) -> float:
    down = ret[ret < 0]
    if len(down) < 2: return np.nan
    dsd = down.std() * np.sqrt(TRADING_DAYS)
    return float((ann_return(ret) - rfr) / dsd) if dsd != 0 else np.nan

def max_dd(prices: pd.Series) -> float:
    roll_max = prices.cummax()
    return float(((prices - roll_max) / roll_max).min())

def beta_calc(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    aligned = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 10: return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else np.nan

def alpha_calc(asset_ret: pd.Series, bench_ret: pd.Series, rfr: float = 0.05) -> float:
    b = beta_calc(asset_ret, bench_ret)
    if np.isnan(b): return np.nan
    return float(ann_return(asset_ret) - (rfr + b * (ann_return(bench_ret) - rfr)))

def var_hist(ret: pd.Series, conf: float = 0.95) -> float:
    return float(np.percentile(ret.dropna(), (1 - conf) * 100))

def cvar_hist(ret: pd.Series, conf: float = 0.95) -> float:
    v = var_hist(ret, conf)
    tail = ret[ret <= v]
    return float(tail.mean()) if len(tail) > 0 else v

def treynor(asset_ret: pd.Series, bench_ret: pd.Series, rfr: float = 0.05) -> float:
    b = beta_calc(asset_ret, bench_ret)
    if np.isnan(b) or b == 0: return np.nan
    return float((ann_return(asset_ret) - rfr) / b)

def calmar(ret: pd.Series, prices: pd.Series) -> float:
    mdd = abs(max_dd(prices))
    return float(ann_return(ret) / mdd) if mdd != 0 else np.nan

def info_ratio(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    active = (asset_ret - bench_ret).dropna()
    if active.std() == 0 or len(active) < 2: return np.nan
    return float(active.mean() / active.std() * np.sqrt(TRADING_DAYS))

# ============================================================
# ML SIGNALS
# ============================================================

def ml_signals(df: pd.DataFrame) -> pd.Series:
    if not SKLEARN_OK or len(df) < 100:
        return pd.Series(0, index=df.index)
    try:
        p = df["Close"].copy()
        feat = pd.DataFrame(index=df.index)
        feat["rsi14"]   = rsi(p, 14);  feat["rsi7"]    = rsi(p, 7)
        feat["ema7r"]   = ema(p, 7)  / p - 1; feat["ema30r"]  = ema(p, 30) / p - 1
        feat["ema50r"]  = ema(p, 50) / p - 1; feat["ema200r"] = ema(p, 200)/ p - 1
        feat["ret1"]    = p.pct_change(1); feat["ret5"]  = p.pct_change(5)
        feat["ret20"]   = p.pct_change(20)
        feat["vol20"]   = feat["ret1"].rolling(20).std()
        bb_u, _, bb_l   = bollinger(p)
        feat["bb_pct"]  = (p - bb_l) / (bb_u - bb_l + 1e-9)
        ml_line, ms, _  = macd(p); feat["macd_h"] = ml_line - ms
        future_ret      = p.pct_change(10).shift(-10)
        labels = pd.cut(future_ret, bins=[-np.inf, -0.02, 0.02, np.inf], labels=[-1, 0, 1])
        data   = feat.join(labels.rename("y")).dropna()
        if len(data) < 60: return pd.Series(0, index=df.index)
        X = data.drop("y", axis=1).values; y = data["y"].astype(int).values
        scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
        split  = int(len(X_sc) * 0.8)
        clf    = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)
        clf.fit(X_sc[:split], y[:split])
        return pd.Series(clf.predict(X_sc), index=data.index).reindex(df.index).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

# ============================================================
# TREND FOLLOW SIGNALS
# ============================================================

def trend_follow_signals(df: pd.DataFrame) -> dict:
    p = df["Close"].dropna()
    e5   = ema(p, 5); e18 = ema(p, 18); e20 = ema(p, 20); e50 = ema(p, 50)
    e100 = ema(p, 100)
    e200 = ema(p, 200) if len(p) >= 50  else pd.Series(np.nan, index=p.index)
    e65  = ema(p, 65)
    e130 = ema(p, 130) if len(p) >= 100 else pd.Series(np.nan, index=p.index)
    rsi14 = rsi(p, 14)
    def sg(a, b): return (a > b).fillna(False)
    weekly_up  = sg(e65, e130); daily_up = sg(e50, e200)
    swing_up   = sg(e20, e50);  short_up = sg(e5,  e20)
    oversold   = (rsi14 < 30).fillna(False); overbought = (rsi14 > 70).fillna(False)
    candle_color = pd.Series("default", index=p.index)
    candle_color[weekly_up & daily_up & short_up]  = "green"
    candle_color[weekly_up & daily_up & ~short_up] = "purple"
    near_e18  = ((p >= e18 * 0.98) & (p <= e18 * 1.02))
    buy_green = (near_e18 & weekly_up & daily_up).fillna(False)
    buy_blue  = (oversold  & weekly_up & daily_up).fillna(False)
    sell_red    = (near_e18 & ~daily_up).fillna(False)
    sell_fucsia = (overbought & ~weekly_up).fillna(False)
    bounce_b18  = ((p.shift(1) < e18.shift(1)) & (p >= e18) & weekly_up).fillna(False)
    bounce_b50  = ((p.shift(1) < e50.shift(1)) & (p >= e50) & weekly_up).fillna(False)
    e200_ok     = e200.notna()
    bounce_b200 = ((p.shift(1) < e200.shift(1)) & (p >= e200) & e200_ok).fillna(False)
    sell_b18    = ((p.shift(1) > e18.shift(1)) & (p <= e18) & ~weekly_up).fillna(False)
    ema18_cross = ((e18.shift(1) <= e50.shift(1)) & (e18 > e50)).fillna(False)
    trend_score = (weekly_up.astype(int) + daily_up.astype(int) +
                   swing_up.astype(int)  + short_up.astype(int))
    return dict(
        candle_color=candle_color, buy_green=buy_green, buy_blue=buy_blue,
        sell_red=sell_red, sell_fucsia=sell_fucsia, bounce_b18=bounce_b18,
        bounce_b50=bounce_b50, bounce_b200=bounce_b200, sell_b18=sell_b18,
        ema18_cross=ema18_cross, trend_score=trend_score,
        weekly_up=weekly_up, daily_up=daily_up, swing_up=swing_up, short_up=short_up,
        oversold=oversold, overbought=overbought, rsi14=rsi14,
        e5=e5, e18=e18, e20=e20, e50=e50, e100=e100, e200=e200, e65=e65, e130=e130,
    )

def ml_combo_score(df: pd.DataFrame, tfk: dict) -> pd.Series:
    rule = pd.Series(0.0, index=df.index)
    rule += tfk["buy_green"].astype(float)   * 0.8
    rule += tfk["buy_blue"].astype(float)    * 1.5
    rule -= tfk["sell_red"].astype(float)    * 0.8
    rule -= tfk["sell_fucsia"].astype(float) * 1.5
    rule += tfk["bounce_b18"].astype(float)  * 0.6
    rule += tfk["bounce_b50"].astype(float)  * 0.4
    rule -= tfk["sell_b18"].astype(float)    * 0.6
    rule += (tfk["trend_score"] - 2) * 0.12

    def to_sig(s):
        sig = pd.Series(0, index=df.index)
        sig[s >  1.0] = 2;  sig[(s > 0.35)  & (s <= 1.0)]  = 1
        sig[s < -1.0] = -2; sig[(s < -0.35) & (s >= -1.0)] = -1
        return sig

    if not SKLEARN_OK or len(df) < 150:
        return to_sig(rule)
    try:
        p = df["Close"].copy()
        feat = pd.DataFrame(index=df.index)
        feat["trend_sc"] = tfk["trend_score"]; feat["rsi"] = tfk["rsi14"]
        feat["p_e18"]    = (p / tfk["e18"] - 1).clip(-0.20, 0.20)
        feat["p_e50"]    = (p / tfk["e50"] - 1).clip(-0.30, 0.30)
        e200_ok = tfk["e200"].notna().any()
        feat["p_e200"]   = (p / tfk["e200"] - 1).clip(-0.50, 0.50) if e200_ok else pd.Series(0.0, index=df.index)
        feat["e18_e50"]  = (tfk["e18"] / tfk["e50"] - 1).clip(-0.15, 0.15)
        feat["e50_e200"] = (tfk["e50"] / tfk["e200"] - 1).clip(-0.20, 0.20) if e200_ok else pd.Series(0.0, index=df.index)
        feat["ret1"] = p.pct_change(1); feat["ret5"] = p.pct_change(5); feat["ret20"] = p.pct_change(20)
        feat["vol20"] = feat["ret1"].rolling(20).std()
        bb_u, _, bb_l = bollinger(p)
        feat["bb_pos"]  = ((p - bb_l) / (bb_u - bb_l + 1e-9)).clip(0, 1)
        ml_l, ms_l, _  = macd(p); feat["macd_h"] = (ml_l - ms_l).clip(-5, 5)
        feat["rule_sc"] = rule
        fwd  = p.pct_change(5).shift(-5)
        lbl  = pd.cut(fwd, bins=[-np.inf, -0.02, 0.02, np.inf], labels=[-1, 0, 1])
        data = feat.join(lbl.rename("y")).dropna()
        if len(data) < 100: raise ValueError("insufficient")
        X = data.drop("y", axis=1).values; y = data["y"].astype(int).values
        sc = StandardScaler(); X_s = sc.fit_transform(X); sp = int(len(X_s) * 0.75)
        clf = RandomForestClassifier(n_estimators=120, max_depth=7, min_samples_leaf=5,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_s[:sp], y[:sp])
        probs = clf.predict_proba(X_s); cls_list = list(clf.classes_)
        pb = probs[:, cls_list.index(1)]  if  1 in cls_list else np.zeros(len(probs))
        ps = probs[:, cls_list.index(-1)] if -1 in cls_list else np.zeros(len(probs))
        ml_sc    = pd.Series(pb - ps, index=data.index).reindex(df.index).fillna(0)
        combined = ml_sc * 1.4 + rule.reindex(df.index).fillna(0) * 0.6
        sig = pd.Series(0, index=df.index)
        sig[combined >  1.2] = 2;  sig[(combined >  0.4) & (combined <= 1.2)]  = 1
        sig[combined < -1.2] = -2; sig[(combined < -0.4) & (combined >= -1.2)] = -1
        sig[(sig > 0) & (tfk["trend_score"].reindex(df.index).fillna(0) < 2)] = 0
        sig[(sig < 0) & (tfk["trend_score"].reindex(df.index).fillna(0) > 2)] = 0
        return sig
    except Exception:
        return to_sig(rule)

def options_strategies(tfk: dict, info: dict, av: float) -> list:
    def last(s):
        v = s.dropna(); return bool(v.iloc[-1]) if len(v) else False
    wu = last(tfk["weekly_up"]); du = last(tfk["daily_up"])
    su = last(tfk["swing_up"]);  stu = last(tfk["short_up"])
    ov = last(tfk["oversold"]);  ob  = last(tfk["overbought"])
    ts = int(wu) + int(du) + int(su) + int(stu)
    rn   = float(tfk["rsi14"].dropna().iloc[-1]) if len(tfk["rsi14"].dropna()) else 50
    e18n = float(tfk["e18"].dropna().iloc[-1])   if len(tfk["e18"].dropna())   else 0
    e50n = float(tfk["e50"].dropna().iloc[-1])   if len(tfk["e50"].dropna())   else 0
    ivl  = "Low" if av < 0.25 else "Medium" if av < 0.45 else "High"
    out  = []
    if wu and du and ts >= 3:
        out.append(dict(type="bullish", conf="★★★", name="Long Call / LEAPS",
            why="Full trend alignment (weekly+daily+short). Green-candle regime.",
            entry=f"Buy ATM-5%OTM call | 3-6m expiry | Add at EMA18 (${e18n:.2f}) dips"))
        out.append(dict(type="bullish", conf="★★★", name="Poor Man's Covered Call (PMCC)",
            why="Strong sustained uptrend. Long deep-ITM LEAPS + sell short-dated calls.",
            entry="Long 80-delta LEAPS 6-12m + short 30-delta call 30-45 DTE"))
    if wu and du and ov:
        out.append(dict(type="bullish", conf="★★★", name="Buy / DCA (Blue Triangle)",
            why=f"RSI={rn:.0f} oversold inside confirmed uptrend. Highest-probability entry.",
            entry=f"Scale in at current level | Hard stop below EMA50 (${e50n:.2f})"))
    if wu and du and ts >= 2:
        out.append(dict(type="income", conf="★★☆", name="Cash-Secured Put (CSP)",
            why=f"Weekly+daily uptrend. Sell put near EMA50 support (${e50n:.2f}).",
            entry=f"Sell put ~${e50n:.2f} | 30-45 DTE | {ivl} IV ({av:.0%} HV)"))
        out.append(dict(type="income", conf="★★☆", name="Covered Call",
            why=f"Uptrend in place. EMA18 (${e18n:.2f}) near-term resistance.",
            entry=f"Sell call 5-8% OTM | 30 DTE | {ivl} IV"))
    if wu and du and av < 0.40:
        out.append(dict(type="income", conf="★★☆", name="Jade Lizard",
            why=f"Positive trend + {ivl} IV ({av:.0%}). No upside risk if structured correctly.",
            entry="Sell OTM put + sell OTM call spread | Net credit eliminates upside loss"))
    if not wu and not du and ts <= 1:
        out.append(dict(type="bearish", conf="★★☆", name="Bear Put Spread / Long Put",
            why="Weekly and daily trend bearish. Price below key EMAs.",
            entry="Buy ATM put + sell 3-5% lower put | 30-60 DTE"))
    if not out:
        out.append(dict(type="neutral", conf="★☆☆", name="No Setup - Wait",
            why=f"Mixed signals (trend score {ts}/4). No clean setup.",
            entry="Monitor: EMA50>EMA200 + price>EMA18 = readiness for CSP or long call"))
    return out

# ============================================================
# FORECASTING
# ============================================================

def forecast_3m(prices: pd.Series):
    N = 63
    last_date    = prices.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=N)
    result       = {}
    last_price   = float(prices.iloc[-1])
    if ARIMA_OK and len(prices) >= 80:
        try:
            lp  = np.log(prices.dropna())
            mdl = ARIMA(lp, order=(5, 1, 0)).fit()
            fc  = mdl.get_forecast(steps=N)
            fv  = np.exp(fc.predicted_mean.values)
            ci  = fc.conf_int()
            result["arima"] = dict(dates=future_dates, values=fv,
                lower=np.exp(ci.iloc[:, 0].values), upper=np.exp(ci.iloc[:, 1].values),
                last_price=last_price)
        except Exception:
            pass
    if SKLEARN_OK and len(prices) >= 30:
        try:
            y_arr = prices.dropna().values
            X_arr = np.arange(len(y_arr)).reshape(-1, 1)
            lr    = LinearRegression().fit(X_arr, y_arr)
            Xf    = np.arange(len(y_arr), len(y_arr) + N).reshape(-1, 1)
            result["linear"] = dict(dates=future_dates, values=lr.predict(Xf),
                                    last_price=last_price)
        except Exception:
            pass
    return result

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommend(ticker_df: pd.DataFrame, info: dict, bench_ret: pd.Series, rfr: float):
    prices = ticker_df["Close"].dropna()
    ret    = prices.pct_change().dropna()
    score  = 0; pros = []; cons = []; risks = []
    r14     = rsi(prices, 14)
    cur_rsi = float(r14.iloc[-1]) if not r14.empty else 50
    cp      = float(prices.iloc[-1])
    e7_v    = float(ema(prices, 7).iloc[-1]); e30_v  = float(ema(prices, 30).iloc[-1])
    e50_v   = float(ema(prices, 50).iloc[-1])
    e200_v  = float(ema(prices, 200).iloc[-1]) if len(prices) >= 200 else np.nan
    if cur_rsi < 30:
        score += 2; pros.append(f"RSI {cur_rsi:.1f} - oversold, potential reversal")
    elif cur_rsi > 70:
        score -= 2; cons.append(f"RSI {cur_rsi:.1f} - overbought, correction risk")
    if not np.isnan(e200_v):
        if cp > e200_v:
            score += 1; pros.append("Price above 200-day EMA - long-term uptrend intact")
        else:
            score -= 1; cons.append("Price below 200-day EMA - long-term trend broken")
            risks.append("Structural downtrend (price < EMA 200)")
    if e7_v > e30_v > e50_v:
        score += 2; pros.append("Bullish EMA alignment (7 > 30 > 50) - positive momentum")
    elif e7_v < e30_v < e50_v:
        score -= 2; cons.append("Bearish EMA alignment (7 < 30 < 50) - negative momentum")
    av = ann_vol(ret)
    if av > 0.45: risks.append(f"High annualized volatility: {av:.1%}")
    mdd_v = max_dd(prices)
    if mdd_v < -0.30: risks.append(f"Significant historical drawdown: {mdd_v:.1%}")
    sh = sharpe(ret, rfr)
    if not np.isnan(sh):
        if sh > 1.0:   score += 1; pros.append(f"Strong risk-adjusted return - Sharpe {sh:.2f}")
        elif sh < 0:   score -= 1; cons.append(f"Negative Sharpe ratio ({sh:.2f})")
    if len(prices) >= 126:
        mom = prices.iloc[-1] / prices.iloc[-126] - 1
        if mom > 0.10:    score += 1; pros.append(f"6-month momentum: +{mom:.1%}")
        elif mom < -0.15: score -= 1; cons.append(f"Weak 6-month momentum: {mom:.1%}")
    valuation = "fairly valued"
    pe = info.get("trailingPE"); d2e = info.get("debtToEquity"); roe_v = info.get("returnOnEquity")
    if pe:
        if pe < 15:   score += 1; pros.append(f"P/E {pe:.1f} - potentially undervalued"); valuation = "undervalued"
        elif pe > 40: score -= 1; cons.append(f"P/E {pe:.1f} - premium valuation"); valuation = "overvalued"
    if d2e and d2e > 200: risks.append(f"High leverage: Debt/Equity {d2e:.0f}%")
    if roe_v and roe_v > 0.20: score += 1; pros.append(f"Strong ROE {roe_v:.1%}")
    def to_sig(s):
        if s >= 3:  return "BUY",  "buy"
        if s <= -3: return "SELL", "sell"
        return "HOLD", "hold"
    r1m, c1m = to_sig(score); r3m, c3m = to_sig(int(score * 1.05)); r6m, c6m = to_sig(int(score * 0.9))
    return dict(score=score, rec_1m=r1m, cls_1m=c1m, rec_3m=r3m, cls_3m=c3m,
                rec_6m=r6m, cls_6m=c6m, pros=pros, cons=cons, risks=risks, valuation=valuation)

# ============================================================
# DCF SIMULATOR
# ============================================================

def simple_dcf(info: dict, g: float, wacc: float, tg: float, years: int):
    fcf    = info.get("freeCashflow"); shares = info.get("sharesOutstanding")
    price  = info.get("currentPrice") or info.get("regularMarketPrice")
    if not fcf or not shares or fcf <= 0: return None
    pv_flows = sum(fcf * (1 + g) ** y / (1 + wacc) ** y for y in range(1, years + 1))
    terminal = fcf * (1 + g) ** years * (1 + tg) / (wacc - tg) / (1 + wacc) ** years
    total    = pv_flows + terminal; iv = total / shares
    return dict(iv=iv, price=price, upside=(iv / price - 1) if price else None,
                pv_flows=pv_flows, terminal=terminal, total=total)

# ============================================================
# EXCEL EXPORT
# ============================================================

def build_excel(hist_data: dict, risk_df: pd.DataFrame, returns_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for t, df in hist_data.items():
            if df.empty: continue
            d = df.copy()
            if hasattr(d.index, "tz") and d.index.tz:
                d.index = d.index.tz_localize(None)
            d.to_excel(w, sheet_name=t[:28])
        if not returns_df.empty:
            rd = returns_df.copy()
            if hasattr(rd.index, "tz") and rd.index.tz:
                rd.index = rd.index.tz_localize(None)
            rd.to_excel(w, sheet_name="Returns")
        if not risk_df.empty:
            risk_df.to_excel(w, sheet_name="Risk_Metrics")
        if not returns_df.empty:
            returns_df.corr().to_excel(w, sheet_name="Correlation")
            (returns_df.cov() * TRADING_DAYS).to_excel(w, sheet_name="Covariance_Ann")
    buf.seek(0); return buf.read()

# ============================================================
# DATA FETCHING  (TTL cache replaces @st.cache_data)
# ============================================================

def fetch_hist(tickers: tuple, start: str, end: str, interval: str) -> dict:
    key = f"hist|{'|'.join(tickers)}|{start}|{end}|{interval}"
    def _fn():
        out = {}
        for t in tickers:
            try:
                df = yf.Ticker(t).history(start=start, end=end, interval=interval, auto_adjust=True)
                if not df.empty:
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    out[t] = df
            except Exception:
                pass
        return out
    return _cached(key, _fn, ttl=300)

def fetch_info(ticker: str) -> dict:
    key = f"info|{ticker}"
    def _fn():
        try: return yf.Ticker(ticker).info or {}
        except Exception: return {}
    return _cached(key, _fn, ttl=3600)

def fetch_news(ticker: str) -> list:
    key = f"news|{ticker}"
    def _fn():
        try:
            n = yf.Ticker(ticker).news
            return n[:6] if n else []
        except Exception: return []
    return _cached(key, _fn, ttl=900)

def fetch_vix() -> pd.DataFrame:
    def _fn():
        try:
            df = yf.Ticker("^VIX").history(period="1y", auto_adjust=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception: return pd.DataFrame()
    return _cached("vix_1y", _fn, ttl=300)

# ============================================================
# ECHARTS HELPERS  (identical to Streamlit version - return dicts)
# ============================================================

def echarts_heatmap(matrix: pd.DataFrame, title: str,
                    color_min: str = "#ff4545", color_max: str = "#00e87a") -> dict:
    cols = matrix.columns.tolist()
    data = [[j, i, round(float(matrix.iloc[i, j]), 4)]
            for i in range(len(cols)) for j in range(len(cols))]
    return {
        "backgroundColor": "#0f1923",
        "title": {"text": title, "textStyle": {"color": "#e0e0e0", "fontSize": 13}},
        "tooltip": {"position": "top"},
        "grid": {"height": "72%", "top": "12%", "left": "12%", "right": "4%"},
        "xAxis": {"type": "category", "data": cols,
                  "axisLabel": {"color": "#94a3b8", "fontSize": 11},
                  "splitArea": {"show": True, "areaStyle": {"color": ["#162233", "#1a2840"]}}},
        "yAxis": {"type": "category", "data": cols,
                  "axisLabel": {"color": "#94a3b8", "fontSize": 11},
                  "splitArea": {"show": True, "areaStyle": {"color": ["#162233", "#1a2840"]}}},
        "visualMap": {
            "min": -1, "max": 1, "calculable": True, "orient": "horizontal",
            "left": "center", "bottom": "3%",
            "inRange": {"color": [color_min, "#1a2840", color_max]},
            "textStyle": {"color": "#94a3b8"},
        },
        "series": [{"name": title, "type": "heatmap", "data": data,
                    "label": {"show": True, "color": "#e0e0e0", "fontSize": 10},
                    "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,232,122,.4)"}}}],
    }

def echarts_area_animated(dates_list: list, series_dict: dict) -> dict:
    colors = PLOTLY_COLORS; series = []
    for i, (name, vals) in enumerate(series_dict.items()):
        series.append({
            "name": name, "type": "line", "data": [round(v, 2) for v in vals],
            "smooth": True, "showSymbol": False, "areaStyle": {"opacity": 0.08},
            "lineStyle": {"width": 2, "color": colors[i % len(colors)]},
            "itemStyle": {"color": colors[i % len(colors)]},
            "animationDuration": 1800, "animationEasing": "cubicOut",
        })
    return {
        "backgroundColor": "#0f1923", "animation": True,
        "tooltip": {"trigger": "axis", "backgroundColor": "rgba(15,25,35,.97)",
                    "borderColor": "#1e3550", "textStyle": {"color": "#dce8f5", "fontSize": 11}},
        "legend": {"type": "scroll", "top": "1%", "left": "center", "width": "90%",
                   "textStyle": {"color": "#94a3b8", "fontSize": 11},
                   "icon": "roundRect", "itemHeight": 8, "itemGap": 18},
        "grid": {"left": "6%", "right": "3%", "top": "10%", "bottom": "12%", "containLabel": True},
        "xAxis": {"type": "category", "data": dates_list,
                  "axisLabel": {"color": "#7a9ab8", "fontSize": 10},
                  "axisLine": {"lineStyle": {"color": "#1e3550"}}, "boundaryGap": False},
        "yAxis": {"type": "value", "splitLine": {"lineStyle": {"color": "#1e3550"}},
                  "axisLabel": {"color": "#7a9ab8", "fontSize": 10}},
        "dataZoom": [{"type": "inside", "start": 60, "end": 100},
                     {"type": "slider", "start": 60, "end": 100, "borderColor": "#1e3550",
                      "fillerColor": "rgba(0,232,122,.10)", "textStyle": {"color": "#7a9ab8"}, "bottom": "1%"}],
        "series": series,
    }

def echarts_candle(df: pd.DataFrame, ticker: str, ema_cfg: dict,
                   custom_emas: list, show_bb: bool, show_sigs: bool,
                   sigs: pd.Series = None, fc: dict = None, tfk_sigs: dict = None) -> dict:
    ha     = heikin_ashi(df)
    dates  = [_date_str(d) for d in ha.index]
    vol    = [int(v) if not np.isnan(v) else 0 for v in df["Volume"].fillna(0)]
    vol_colors = ["#00e87a" if ha.HA_Close.iloc[i] >= ha.HA_Open.iloc[i] else "#ff4545"
                  for i in range(len(ha))]
    candle_data = []
    cc = tfk_sigs["candle_color"] if tfk_sigs is not None else None
    for i, r in enumerate(ha.itertuples()):
        val = [round(r.HA_Open,2), round(r.HA_Close,2), round(r.HA_Low,2), round(r.HA_High,2)]
        trend_col = str(cc.iloc[i]) if (cc is not None and i < len(cc)) else "default"
        is_up = r.HA_Close >= r.HA_Open
        if trend_col == "green":
            sty = {"color":"#00e87a","color0":"#1a7a40","borderColor":"#00e87a","borderColor0":"#00e87a"}
        elif trend_col == "purple":
            sty = {"color":"#9b59b6","color0":"#6c3483","borderColor":"#9b59b6","borderColor0":"#9b59b6"}
        else:
            sty = {"color":"#00e87a","borderColor":"#00e87a"} if is_up else {
                "color0":"#ff4545","borderColor0":"#ff4545","color":"#00e87a","borderColor":"#00e87a"}
        candle_data.append({"value": val, "itemStyle": sty})

    series = [{"name": ticker, "type": "candlestick", "data": candle_data,
               "gridIndex": 0, "xAxisIndex": 0, "yAxisIndex": 0,
               "itemStyle": {"color":"#00e87a","color0":"#ff4545","borderColor":"#00e87a","borderColor0":"#ff4545"}}]
    ema_palette = {"7":"#ff6b35","30":"#ffd700","50":"#4da6ff","200":"#da70d6"}
    legend_items = [ticker]

    for period, show in ema_cfg.items():
        if show:
            vals  = ema(df["Close"], int(period)).round(2).tolist()
            color = ema_palette.get(period, "#aaa")
            series.append({"name": f"EMA {period}", "type": "line", "data": vals,
                           "smooth": True, "lineStyle": {"width": 1.5, "color": color},
                           "itemStyle": {"color": color}, "showSymbol": False,
                           "gridIndex": 0, "xAxisIndex": 0, "yAxisIndex": 0})
            legend_items.append(f"EMA {period}")

    if tfk_sigs is not None:
        e18v = tfk_sigs["e18"].round(2).reindex(df.index).tolist()
        series.append({"name": "EMA 18", "type": "line", "data": e18v, "smooth": True,
                       "lineStyle": {"width": 1.2, "color": "#20c0c0", "type": "dashed"},
                       "showSymbol": False, "gridIndex": 0, "xAxisIndex": 0, "yAxisIndex": 0})
        legend_items.append("EMA 18")

    extra_colors = ["#40e0d0","#ff69b4","#c0c020","#a0a060"]
    for idx_c, p_val in enumerate(custom_emas):
        vals  = ema(df["Close"], p_val).round(2).tolist()
        color = extra_colors[idx_c % len(extra_colors)]
        series.append({"name": f"EMA {p_val}", "type": "line", "data": vals, "smooth": True,
                       "lineStyle": {"width": 1.5, "color": color, "type": "dashed"},
                       "itemStyle": {"color": color}, "showSymbol": False,
                       "gridIndex": 0, "xAxisIndex": 0, "yAxisIndex": 0})
        legend_items.append(f"EMA {p_val}")

    if show_bb:
        bb_u, bb_m, bb_l = bollinger(df["Close"])
        for bname, bvals, bdash in [("BB Up",bb_u,"dashed"),("BB Mid",bb_m,"solid"),("BB Low",bb_l,"dashed")]:
            series.append({"name": bname, "type": "line",
                           "data": [round(v,2) if not np.isnan(v) else None for v in bvals],
                           "smooth": True, "lineStyle": {"width":1,"color":"rgba(90,90,200,.6)","type":bdash},
                           "showSymbol": False, "gridIndex":0,"xAxisIndex":0,"yAxisIndex":0})
        legend_items += ["BB Up","BB Mid","BB Low"]

    if show_sigs:
        def mk_scatter(name, idx_list, price_fn, symbol, size, color, rotate=0):
            data_pts = [[dates[i], round(price_fn(i), 2)] for i in idx_list]
            if not data_pts: return None
            s = {"name": name, "type": "scatter", "data": data_pts, "symbol": symbol,
                 "symbolSize": size, "itemStyle": {"color": color},
                 "gridIndex":0,"xAxisIndex":0,"yAxisIndex":0}
            if rotate: s["symbolRotate"] = rotate
            return s
        def valid_idx(mask):
            if mask is None: return []
            return [i for i, idx in enumerate(df.index) if idx in mask.index and bool(mask.loc[idx])]
        if tfk_sigs is not None:
            bg_idx = valid_idx(tfk_sigs["buy_green"])
            s = mk_scatter("Buy Zone", bg_idx, lambda i: float(df["Low"].iloc[i])*0.970, "triangle", 10, "#00e87a")
            if s: series.append(s); legend_items.append("Buy Zone")
            bb_idx = valid_idx(tfk_sigs["buy_blue"])
            s = mk_scatter("Strong Buy", bb_idx, lambda i: float(df["Low"].iloc[i])*0.955, "triangle", 14, "#4da6ff")
            if s: series.append(s); legend_items.append("Strong Buy")
            sr_idx = valid_idx(tfk_sigs["sell_red"])
            s = mk_scatter("Sell Zone", sr_idx, lambda i: float(df["High"].iloc[i])*1.030, "triangle", 10, "#ff4545", rotate=180)
            if s: series.append(s); legend_items.append("Sell Zone")
            sf_idx = valid_idx(tfk_sigs["sell_fucsia"])
            s = mk_scatter("Strong Sell", sf_idx, lambda i: float(df["High"].iloc[i])*1.045, "triangle", 14, "#ff69b4", rotate=180)
            if s: series.append(s); legend_items.append("Strong Sell")
            b18_idx = valid_idx(tfk_sigs["bounce_b18"])
            if b18_idx:
                series.append({"name":"B18","type":"scatter",
                    "data":[{"value":[dates[i],round(float(df["Low"].iloc[i])*0.965,2)],
                             "label":{"show":True,"formatter":"B18","position":"bottom","color":"#00e87a","fontSize":9,"fontWeight":"bold"}}
                            for i in b18_idx],
                    "symbol":"circle","symbolSize":6,"itemStyle":{"color":"#00e87a"},
                    "gridIndex":0,"xAxisIndex":0,"yAxisIndex":0})
                legend_items.append("B18")
            b50_idx = valid_idx(tfk_sigs["bounce_b50"])
            if b50_idx:
                series.append({"name":"B50","type":"scatter",
                    "data":[{"value":[dates[i],round(float(df["Low"].iloc[i])*0.960,2)],
                             "label":{"show":True,"formatter":"B50","position":"bottom","color":"#ffd700","fontSize":9,"fontWeight":"bold"}}
                            for i in b50_idx],
                    "symbol":"circle","symbolSize":6,"itemStyle":{"color":"#ffd700"},
                    "gridIndex":0,"xAxisIndex":0,"yAxisIndex":0})
                legend_items.append("B50")
            c_idx = valid_idx(tfk_sigs["ema18_cross"])
            if c_idx:
                series.append({"name":"Cont(C)","type":"scatter",
                    "data":[{"value":[dates[i],round(float(df["Low"].iloc[i])*0.975,2)],
                             "label":{"show":True,"formatter":"C","position":"bottom","color":"#c0c020",
                                      "fontSize":9,"fontWeight":"bold","borderColor":"#c0c020","borderWidth":1,"padding":[1,3]}}
                            for i in c_idx],
                    "symbol":"rect","symbolSize":[16,14],
                    "itemStyle":{"color":"rgba(192,192,32,.25)","borderColor":"#c0c020","borderWidth":1},
                    "gridIndex":0,"xAxisIndex":0,"yAxisIndex":0})
                legend_items.append("Cont(C)")
        elif sigs is not None:
            buy_idx_ml  = [i for i,idx in enumerate(df.index) if idx in sigs.index and sigs.loc[idx] in (1,2)]
            sell_idx_ml = [i for i,idx in enumerate(df.index) if idx in sigs.index and sigs.loc[idx] in (-1,-2)]
            s = mk_scatter("BUY",  buy_idx_ml,  lambda i: float(df["Low"].iloc[i])*0.98,  "triangle", 10, "#00e87a")
            if s: series.append(s); legend_items.append("BUY")
            s = mk_scatter("SELL", sell_idx_ml, lambda i: float(df["High"].iloc[i])*1.02, "triangle", 10, "#ff4545", rotate=180)
            if s: series.append(s); legend_items.append("SELL")

    all_dates = dates
    if fc and "arima" in fc:
        fd  = [_date_str(d) for d in fc["arima"]["dates"]]
        fv  = [round(float(v),2) for v in fc["arima"]["values"]]
        last_price = round(fc["arima"]["last_price"],2)
        fc_vals_conn  = [last_price] + fv
        fc_upper_conn = [last_price] + [round(float(v),2) for v in fc["arima"]["upper"]]
        fc_lower_conn = [last_price] + [round(float(v),2) for v in fc["arima"]["lower"]]
        all_dates = dates + fd; pad = len(dates) - 1
        for name, conn_vals, style in [
            ("FC Upper", fc_upper_conn, {"width":1,"color":"rgba(255,215,0,.25)","type":"dashed"}),
            ("FC Lower", fc_lower_conn, {"width":1,"color":"rgba(255,215,0,.25)","type":"dashed"}),
        ]:
            series.append({"name":name,"type":"line","data":[None]*pad+conn_vals,
                           "lineStyle":style,"showSymbol":False,"gridIndex":0,"xAxisIndex":0,"yAxisIndex":0})
        series.append({"name":"3M Forecast","type":"line","data":[None]*pad+fc_vals_conn,
                       "lineStyle":{"width":2,"color":"#ffd700","type":"dotted"},
                       "itemStyle":{"color":"#ffd700"},"showSymbol":False,
                       "gridIndex":0,"xAxisIndex":0,"yAxisIndex":0})
        legend_items.append("3M Forecast")

    series.append({"name":"Volume","type":"bar",
        "data":[{"value":vol[i],"itemStyle":{"color":vol_colors[i]}} for i in range(len(vol))],
        "gridIndex":1,"xAxisIndex":1,"yAxisIndex":1,"barMaxWidth":6})
    rsi_vals = [round(v,2) if not np.isnan(v) else None for v in rsi(df["Close"],14)]
    series.append({"name":"RSI 14","type":"line","data":rsi_vals,
                   "lineStyle":{"width":1.5,"color":"#9b59b6"},"showSymbol":False,
                   "gridIndex":2,"xAxisIndex":2,"yAxisIndex":2})

    return {
        "backgroundColor": "#0f1923", "animation": True, "animationDuration": 800,
        "tooltip": {"trigger":"axis","axisPointer":{"type":"cross"},
                    "backgroundColor":"rgba(15,25,35,.97)","borderColor":"#1e3550",
                    "textStyle":{"color":"#dce8f5","fontSize":11}},
        "legend": {"type":"scroll","data":legend_items,"top":"1%","left":"center","width":"98%",
                   "textStyle":{"color":"#c8daf0","fontSize":10},"icon":"roundRect","itemHeight":8,"itemGap":16},
        "axisPointer": {"link":[{"xAxisIndex":"all"}]},
        "grid": [{"left":"7%","right":"3%","top":"8%","height":"52%"},
                 {"left":"7%","right":"3%","top":"64%","height":"12%"},
                 {"left":"7%","right":"3%","top":"79%","height":"12%"}],
        "xAxis": [
            {"type":"category","data":all_dates,"gridIndex":0,"axisLabel":{"show":False},"axisLine":{"lineStyle":{"color":"#1e3550"}},"splitLine":{"show":False}},
            {"type":"category","data":dates,"gridIndex":1,"axisLabel":{"show":False},"axisLine":{"lineStyle":{"color":"#1e3550"}}},
            {"type":"category","data":dates,"gridIndex":2,"axisLabel":{"color":"#6a8aa8","fontSize":9},"axisLine":{"lineStyle":{"color":"#1e3550"}}},
        ],
        "yAxis": [
            {"scale":True,"gridIndex":0,"splitLine":{"lineStyle":{"color":"#1e3550"}},"axisLabel":{"color":"#7a9ab8","fontSize":10}},
            {"scale":True,"gridIndex":1,"splitNumber":2,"axisLabel":{"color":"#6a8aa8","fontSize":9},"splitLine":{"lineStyle":{"color":"#1e3550"}}},
            {"scale":True,"gridIndex":2,"min":0,"max":100,"splitNumber":2,"axisLabel":{"color":"#6a8aa8","fontSize":9},"splitLine":{"lineStyle":{"color":"#1e3550"}}},
        ],
        "dataZoom": [
            {"type":"inside","xAxisIndex":[0,1,2],"start":70,"end":100},
            {"type":"slider","xAxisIndex":[0,1,2],"start":70,"end":100,"bottom":"1%","height":18,
             "borderColor":"#1e3550","fillerColor":"rgba(0,232,122,.10)","textStyle":{"color":"#7a9ab8"}},
        ],
        "series": series,
    }

# ============================================================
# PLOTLY HELPERS
# ============================================================

def _dark_layout(**kw):
    base = dict(template="plotly_dark", paper_bgcolor="#0f1923", plot_bgcolor="#162233",
                font=dict(color="#94a3b8", size=11), margin=dict(l=50,r=20,t=45,b=40),
                hovermode="x unified", xaxis_gridcolor="#1e3550", xaxis_zerolinecolor="#1e3550",
                yaxis_gridcolor="#1e3550", yaxis_zerolinecolor="#1e3550")
    base.update(kw); return base

def _fig_json(fig) -> str:
    return pio.to_json(fig) if (PLOTLY_OK and fig is not None) else None

def plotly_perf(prices_df, bench, bench_name):
    fig = go.Figure()
    for i, col in enumerate(prices_df.columns):
        norm = prices_df[col] / prices_df[col].dropna().iloc[0] * 100
        fig.add_trace(go.Scatter(x=norm.index, y=norm, name=col,
                                 line=dict(color=PLOTLY_COLORS[i%len(PLOTLY_COLORS)], width=2)))
    if bench is not None and not bench.empty:
        nb = bench / bench.dropna().iloc[0] * 100
        fig.add_trace(go.Scatter(x=nb.index, y=nb, name=bench_name,
                                 line=dict(color="#4a6a8a", width=2, dash="dash")))
    fig.update_layout(title="Normalized Performance (Base 100)", yaxis_title="Indexed Price",
                      **_dark_layout(height=380)); return fig

def plotly_rolling_beta(returns_df, bench_ret):
    fig = go.Figure()
    for i, col in enumerate(returns_df.columns):
        rb = returns_df[col].rolling(60).cov(bench_ret) / bench_ret.rolling(60).var()
        fig.add_trace(go.Scatter(x=rb.index, y=rb, name=col,
                                 line=dict(color=PLOTLY_COLORS[i%len(PLOTLY_COLORS)], width=1.5)))
    fig.add_hline(y=1, line_dash="dash", line_color="#333",
                  annotation_text="beta=1", annotation_font_color="#7a9ab8")
    fig.update_layout(title="Rolling 60-Day Beta", yaxis_title="Beta", **_dark_layout(height=320)); return fig

def plotly_drawdown(prices_df):
    fig = go.Figure()
    for i, col in enumerate(prices_df.columns):
        p   = prices_df[col].dropna(); dd = (p - p.cummax()) / p.cummax() * 100
        col_hex = PLOTLY_COLORS[i%len(PLOTLY_COLORS)]; col_fill = hex_to_rgba(col_hex, 0.07)
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=col,
                                 line=dict(color=col_hex, width=1.5), fill="tozeroy", fillcolor=col_fill))
    fig.update_layout(title="Drawdown from Peak (%)", yaxis_title="Drawdown (%)",
                      **_dark_layout(height=320)); return fig

def plotly_rr_scatter(risk_df, rfr, bench_ret):
    fig = go.Figure()
    for i, ticker in enumerate(risk_df.index):
        row = risk_df.loc[ticker]
        rv  = row.get("Ann. Volatility"); rr = row.get("Ann. Return")
        sh  = row.get("Sharpe Ratio", 0) or 0
        if rv is None or rr is None: continue
        c = PLOTLY_COLORS[i%len(PLOTLY_COLORS)]
        fig.add_trace(go.Scatter(x=[rv*100], y=[rr*100], mode="markers+text",
            text=[ticker], textposition="top center",
            marker=dict(size=max(8,min(28,abs(sh)*8)), color=c, opacity=.85,
                        line=dict(width=1, color="white")), name=ticker))
    if bench_ret is not None and not bench_ret.empty:
        bv = ann_vol(bench_ret)*100; br = ann_return(bench_ret)*100
        slope = (br - rfr*100) / (bv + 1e-9); xs = np.linspace(0, max(35, bv*1.6), 50)
        fig.add_trace(go.Scatter(x=xs, y=rfr*100+slope*xs, mode="lines",
                                 name="CML", line=dict(color="#333", dash="dash", width=1)))
    fig.update_layout(title="Risk-Return Map  (bubble size proportional to |Sharpe|)",
                      xaxis_title="Annualized Volatility (%)", yaxis_title="Annualized Return (%)",
                      **_dark_layout(height=440)); return fig

def plotly_vol_bar(returns_df):
    vols   = returns_df.std() * np.sqrt(TRADING_DAYS) * 100
    colors = ["#00e87a" if v < 25 else "#ffd700" if v < 40 else "#ff4545" for v in vols]
    fig    = go.Figure(go.Bar(x=vols.index.tolist(), y=vols.values, marker_color=colors,
                               text=[f"{v:.1f}%" for v in vols], textposition="outside"))
    fig.add_hline(y=20, line_dash="dash", line_color="#333", annotation_text="20% ref.")
    fig.update_layout(title="Annualized Volatility by Asset (%)", yaxis_title="Volatility (%)",
                      **_dark_layout(height=320)); return fig

def plotly_vix(vix_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vix_df.index, y=vix_df["Close"], name="VIX",
                             line=dict(color="#ff6b35", width=2),
                             fill="tozeroy", fillcolor="rgba(255,107,53,.07)"))
    for y0,y1,fc in [(0,15,"rgba(0,232,122,.04)"),(15,25,"rgba(255,215,0,.03)"),
                     (25,40,"rgba(255,107,53,.03)"),(40,100,"rgba(255,69,69,.04)")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=fc, line_width=0)
    for y, lbl, col in [(15,"15-Normal","rgba(0,232,122,.5)"),(25,"25-Elevated","rgba(255,215,0,.5)"),
                        (40,"40-Extreme","rgba(255,69,69,.5)")]:
        fig.add_hline(y=y, line_dash="dash", line_color=col,
                      annotation_text=lbl, annotation_font_color=col)
    fig.update_layout(title="CBOE Volatility Index - VIX (1Y)", yaxis_title="VIX",
                      **_dark_layout(height=280)); return fig

def plotly_pair_scatter(returns_df, t1, t2):
    if t1 not in returns_df.columns or t2 not in returns_df.columns: return None
    aligned  = pd.concat([returns_df[t1], returns_df[t2]], axis=1).dropna()
    x_vals   = aligned.iloc[:, 0].values * 100; y_vals = aligned.iloc[:, 1].values * 100
    corr_val = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="markers",
        marker=dict(size=4, color="#4da6ff", opacity=0.45), name=f"{t1} vs {t2}"))
    if SKLEARN_OK and len(x_vals) > 5:
        lr = LinearRegression().fit(x_vals.reshape(-1,1), y_vals)
        xs = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig.add_trace(go.Scatter(x=xs, y=lr.predict(xs.reshape(-1,1)), mode="lines",
                                 line=dict(color="#4da6ff", width=2), name="OLS Trend"))
    fig.update_layout(title=f"Pair Returns: {t1} vs {t2}  (rho={corr_val:.3f})",
                      xaxis_title=f"{t1} Daily Return (%)", yaxis_title=f"{t2} Daily Return (%)",
                      **_dark_layout(height=420)); return fig

def plotly_scatter_matrix(returns_df):
    fig = px.scatter_matrix(returns_df, dimensions=returns_df.columns.tolist(),
                            color_discrete_sequence=PLOTLY_COLORS)
    fig.update_traces(marker=dict(size=2, opacity=.4))
    fig.update_layout(title="Returns Scatter Matrix", **_dark_layout(height=600)); return fig

def monte_carlo_frontier(returns_df, n_portfolios=3000, rfr=0.05):
    n = len(returns_df.columns)
    if n < 2 or len(returns_df) < 30: return None
    tickers  = returns_df.columns.tolist()
    mean_ret = returns_df.mean() * TRADING_DAYS; cov_mat = returns_df.cov() * TRADING_DAYS
    np.random.seed(42)
    vols, rets, sharpes, weights_list = [], [], [], []
    for _ in range(n_portfolios):
        w  = np.random.dirichlet(np.ones(n)); pr = float(np.dot(w, mean_ret))
        pv = float(np.sqrt(w @ cov_mat.values @ w)); ps = (pr - rfr) / pv if pv > 0 else 0
        vols.append(pv*100); rets.append(pr*100); sharpes.append(ps); weights_list.append(w)
    msi = int(np.argmax(sharpes)); mvi = int(np.argmin(vols))
    return dict(vols=vols, rets=rets, sharpes=sharpes, weights=weights_list,
                tickers=tickers, max_sh_idx=msi, min_v_idx=mvi)

def plotly_mc_frontier(mc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc["vols"], y=mc["rets"], mode="markers",
        marker=dict(color=mc["sharpes"],
                    colorscale=[[0,"#ff4545"],[0.5,"#ffd700"],[1,"#00e87a"]],
                    size=3, opacity=0.55,
                    colorbar=dict(title=dict(text="Sharpe",font=dict(color="#94a3b8")),
                                  tickfont=dict(color="#94a3b8"),bgcolor="rgba(0,0,0,0)",outlinewidth=0)),
        text=[f"Sharpe: {s:.2f}" for s in mc["sharpes"]],
        hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<br>%{text}<extra></extra>", name="Portfolios"))
    msi = mc["max_sh_idx"]; mvi = mc["min_v_idx"]
    fig.add_trace(go.Scatter(x=[mc["vols"][msi]], y=[mc["rets"][msi]], mode="markers+text",
        marker=dict(size=16,color="#00e87a",symbol="star",line=dict(width=1,color="white")),
        text=["Max Sharpe"], textposition="top right", textfont=dict(color="#00e87a"), name="Max Sharpe"))
    fig.add_trace(go.Scatter(x=[mc["vols"][mvi]], y=[mc["rets"][mvi]], mode="markers+text",
        marker=dict(size=16,color="#4da6ff",symbol="diamond",line=dict(width=1,color="white")),
        text=["Min Risk"], textposition="top right", textfont=dict(color="#4da6ff"), name="Min Volatility"))
    fig.update_layout(title=f"Monte Carlo Efficient Frontier  ({len(mc['vols']):,} portfolios)",
                      xaxis_title="Annualized Volatility (%)", yaxis_title="Annualized Return (%)",
                      **_dark_layout(height=520)); return fig

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html',
        popular=POPULAR,
        benchmarks=BENCHMARK_INDICES,
        defaults=_ASSET_DEFAULTS)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        p = request.get_json(force=True)

        tickers    = [t.strip().upper() for t in p.get('tickers', []) if t.strip()]
        start_str  = p.get('start', (datetime.today()-timedelta(1095)).strftime('%Y-%m-%d'))
        end_str    = p.get('end',   datetime.today().strftime('%Y-%m-%d'))
        freq       = p.get('freq', '1d')
        bench_tick = p.get('bench_ticker', '^GSPC')
        bench_name = p.get('bench_name',  'S&P 500 (^GSPC)')
        rfr        = float(p.get('rfr', 0.0457))
        conf       = float(p.get('conf', 0.95))
        capital    = float(p.get('capital', 10000))
        ema_cfg    = p.get('ema_cfg', {"7":True,"30":True,"50":True,"200":True})
        custom_emas= [int(x) for x in p.get('custom_emas', []) if str(x).isdigit()]
        show_bb    = bool(p.get('show_bb', True))
        show_sigs  = bool(p.get('show_sigs', True))
        dcf_g      = float(p.get('dcf_g',   0.10))
        dcf_wacc   = float(p.get('dcf_wacc', 0.10))
        dcf_tg     = float(p.get('dcf_tg',   0.03))
        dcf_yrs    = int(p.get('dcf_yrs',   5))
        mc_n       = int(p.get('mc_n', 3000))
        sel_chart  = p.get('sel_chart', tickers[0] if tickers else '')
        sel_sim    = p.get('sel_sim',   tickers[0] if tickers else '')

        if not tickers:
            return jsonify({'ok': False, 'error': 'No tickers provided'}), 400

        hist      = fetch_hist(tuple(tickers), start_str, end_str, freq)
        bench_raw = fetch_hist((bench_tick,), start_str, end_str, freq)
        vix_df    = fetch_vix()

        valid = [t for t in tickers if t in hist]
        if not valid:
            return jsonify({'ok': False, 'error': 'No data retrieved. Check tickers.'}), 400

        prices_df  = pd.DataFrame({t: hist[t]["Close"] for t in valid}).dropna(how="all")
        bench_ser  = (bench_raw[bench_tick]["Close"]
                      if bench_tick in bench_raw else pd.Series(dtype=float))
        returns_df = prices_df.pct_change().dropna()
        bench_ret  = bench_ser.pct_change().dropna()

        # Risk metrics table
        rows = []
        for t in valid:
            if t not in returns_df.columns: continue
            r  = returns_df[t].dropna(); pr = prices_df[t].dropna()
            rows.append({
                "Ticker": t,
                "Ann. Return":     _safe(ann_return(r)),
                "Ann. Volatility": _safe(ann_vol(r)),
                "Sharpe Ratio":    _safe(sharpe(r, rfr)),
                "Sortino Ratio":   _safe(sortino(r, rfr)),
                "Max Drawdown":    _safe(max_dd(pr)),
                "Beta":            _safe(beta_calc(r, bench_ret)),
                "Alpha":           _safe(alpha_calc(r, bench_ret, rfr)),
                f"VaR {int(conf*100)}%":  _safe(var_hist(r, conf)),
                f"CVaR {int(conf*100)}%": _safe(cvar_hist(r, conf)),
                "Treynor":         _safe(treynor(r, bench_ret, rfr)),
                "Calmar":          _safe(calmar(r, pr)),
                "Info. Ratio":     _safe(info_ratio(r, bench_ret.reindex(r.index).dropna())),
            })
        risk_df = pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()

        # VIX widget
        vix_data = {}
        if not vix_df.empty:
            cur  = float(vix_df["Close"].iloc[-1])
            prev = float(vix_df["Close"].iloc[-2]) if len(vix_df) > 1 else cur
            chg  = cur - prev
            if   cur < 15: regime, col = "Low Volatility - Risk-On", "#00e87a"
            elif cur < 20: regime, col = "Normal",                    "#00e87a"
            elif cur < 25: regime, col = "Elevated - Caution",        "#ffd700"
            elif cur < 35: regime, col = "High - Risk-Off",           "#ff6b35"
            else:          regime, col = "Extreme Fear",              "#ff4545"
            ch_col = "#ff4545" if chg >= 0 else "#00e87a"
            arrow  = "▲" if chg >= 0 else "▼"
            vix_data = dict(cur=round(cur,2), chg=round(chg,2), regime=regime,
                            col=col, ch_col=ch_col, arrow=arrow)

        # KPI cards
        kpis = []
        for t in valid:
            info_d = fetch_info(t)
            pr     = prices_df[t].dropna()
            rt     = returns_df[t].dropna() if t in returns_df else pd.Series()
            cp     = float(pr.iloc[-1]) if not pr.empty else 0
            pp     = float(pr.iloc[-2]) if len(pr) > 1 else cp
            dc     = (cp/pp-1)*100 if pp else 0
            bt     = _safe(beta_calc(rt, bench_ret)) if len(rt) > 10 else None
            av_v   = _safe(ann_vol(rt))
            kpis.append(dict(
                ticker=t, name=(info_d.get("longName",t) or t)[:26],
                price=round(cp,2), daily_chg=round(dc,2),
                vol_pct=round(av_v*100,1) if av_v else None,
                beta=round(bt,2) if bt else None,
            ))

        # Candle chart for selected asset
        sel_chart = sel_chart if sel_chart in valid else valid[0]
        chart_df  = hist[sel_chart].copy()
        tfk_sigs_chart = trend_follow_signals(chart_df) if show_sigs else None
        combo_sigs     = (ml_combo_score(chart_df, tfk_sigs_chart)
                          if (show_sigs and tfk_sigs_chart is not None) else None)
        fc_data = forecast_3m(chart_df["Close"])
        candle_opt = echarts_candle(chart_df, sel_chart, ema_cfg, custom_emas,
                                    show_bb, show_sigs, combo_sigs, fc_data, tfk_sigs_chart)

        trend_data = {}
        if tfk_sigs_chart:
            ts_s = tfk_sigs_chart["trend_score"].dropna()
            trend_data = dict(
                score=int(ts_s.iloc[-1]) if len(ts_s) else 0,
                weekly_up=bool(tfk_sigs_chart["weekly_up"].dropna().iloc[-1]) if len(tfk_sigs_chart["weekly_up"].dropna()) else False,
                daily_up= bool(tfk_sigs_chart["daily_up"].dropna().iloc[-1])  if len(tfk_sigs_chart["daily_up"].dropna())  else False,
                swing_up= bool(tfk_sigs_chart["swing_up"].dropna().iloc[-1])  if len(tfk_sigs_chart["swing_up"].dropna())  else False,
                short_up= bool(tfk_sigs_chart["short_up"].dropna().iloc[-1])  if len(tfk_sigs_chart["short_up"].dropna())  else False,
                candle=str(tfk_sigs_chart["candle_color"].dropna().iloc[-1])   if len(tfk_sigs_chart["candle_color"].dropna()) else "default",
                rsi=round(float(tfk_sigs_chart["rsi14"].dropna().iloc[-1]),1) if len(tfk_sigs_chart["rsi14"].dropna()) else 50.0,
            )

        # Cumulative return chart
        norm_dict   = {t: (prices_df[t].dropna() / prices_df[t].dropna().iloc[0] * 100).round(2).tolist() for t in valid}
        date_labels = [_date_str(d) for d in prices_df.index]
        cumul_opt   = echarts_area_animated(date_labels, norm_dict)

        # Benchmark tab
        perf_json  = _fig_json(plotly_perf(prices_df, bench_ser, bench_name)) if PLOTLY_OK else None
        rbeta_json = (_fig_json(plotly_rolling_beta(returns_df, bench_ret))
                      if (PLOTLY_OK and len(returns_df) > 60) else None)
        bench_summary = {}
        if not risk_df.empty:
            cols = [c for c in ["Ann. Return","Ann. Volatility","Sharpe Ratio","Beta","Alpha","Max Drawdown"] if c in risk_df.columns]
            bench_summary = risk_df[cols].to_dict(orient="index")

        # Correlation tab
        corr_heatmap = None; corr_table = {}; rolling_corr_json = None
        pair_scatter_json = None; scatter_mat_json = None
        if len(valid) >= 2:
            corr = returns_df.corr()
            corr_heatmap = echarts_heatmap(corr, "Correlation Matrix")
            corr_table   = corr.round(4).to_dict(orient="index")
            if PLOTLY_OK and not bench_ret.empty:
                fig_rc = go.Figure()
                for i, t in enumerate(valid):
                    rc = returns_df[t].rolling(30).corr(bench_ret)
                    fig_rc.add_trace(go.Scatter(x=[_date_str(x) for x in rc.index], y=rc.tolist(),
                                                name=t, line=dict(color=PLOTLY_COLORS[i%len(PLOTLY_COLORS)],width=1.5)))
                fig_rc.add_hline(y=0, line_color="#333", line_dash="dash")
                fig_rc.update_layout(title=f"Rolling 30-Day Correlation vs {bench_name}",
                                     yaxis_range=[-1,1], **_dark_layout(height=280))
                rolling_corr_json = _fig_json(fig_rc)
            if PLOTLY_OK:
                pair_scatter_json = _fig_json(plotly_pair_scatter(returns_df, valid[0],
                                              valid[1] if len(valid)>1 else valid[0]))
                if len(valid) <= 6:
                    scatter_mat_json = _fig_json(plotly_scatter_matrix(returns_df))

        # Covariance tab
        cov_heatmap = None; cov_table = {}; vol_bar_json = None
        mc_frontier_json = None; mc_weights = {}
        if len(valid) >= 2:
            cov = returns_df.cov() * TRADING_DAYS
            cov_opt = echarts_heatmap(cov, "Annualized Covariance",
                                      color_min="#ffd700", color_max="#ff4545")
            cov_opt["visualMap"]["min"] = float(cov.values.min())
            cov_opt["visualMap"]["max"] = float(cov.values.max())
            cov_heatmap = cov_opt; cov_table = cov.round(6).to_dict(orient="index")
            if PLOTLY_OK:
                vol_bar_json = _fig_json(plotly_vol_bar(returns_df))
            mc = monte_carlo_frontier(returns_df, n_portfolios=mc_n, rfr=rfr)
            if mc and PLOTLY_OK:
                mc_frontier_json = _fig_json(plotly_mc_frontier(mc))
                msi = mc["max_sh_idx"]; mvi = mc["min_v_idx"]
                mc_weights = dict(
                    tickers=mc["tickers"],
                    max_sh=dict(weights=[round(w,4) for w in mc["weights"][msi]],
                                vol=round(mc["vols"][msi],2), ret=round(mc["rets"][msi],2),
                                sharpe=round(mc["sharpes"][msi],3)),
                    min_v=dict(weights=[round(w,4) for w in mc["weights"][mvi]],
                               vol=round(mc["vols"][mvi],2),  ret=round(mc["rets"][mvi],2),
                               sharpe=round(mc["sharpes"][mvi],3)),
                )

        # Risk metrics tab
        rr_scatter_json = None; drawdown_json = None; vix_chart_json = None
        risk_table = {}
        if not risk_df.empty:
            risk_table = risk_df.to_dict(orient="index")
            if PLOTLY_OK:
                rr_scatter_json = _fig_json(plotly_rr_scatter(risk_df, rfr, bench_ret))
                drawdown_json   = _fig_json(plotly_drawdown(prices_df))
        if not vix_df.empty and PLOTLY_OK:
            vix_chart_json = _fig_json(plotly_vix(vix_df))

        # Signals
        signals = {}
        for t in valid:
            info_d = fetch_info(t)
            rec    = recommend(hist[t], info_d, bench_ret, rfr)
            strats = []
            try:
                t_sigs = trend_follow_signals(hist[t])
                av_v   = ann_vol(hist[t]["Close"].pct_change().dropna())
                strats = options_strategies(t_sigs, info_d, av_v if av_v else 0)
            except Exception:
                pass
            news = fetch_news(t)
            signals[t] = dict(
                name=(info_d.get("longName",t) or t),
                rec=rec, strats=strats,
                news=[dict(title=a.get("title",""), link=a.get("link","#"),
                           publisher=a.get("publisher","")) for a in news],
            )

        # Simulator
        sel_sim  = sel_sim if sel_sim in valid else valid[0]
        sim_data = _build_simulator(sel_sim, hist, dcf_g, dcf_wacc, dcf_tg, dcf_yrs)

        result = dict(
            ok=True,
            meta=dict(valid=valid, bench_name=bench_name, conf=int(conf*100),
                      sel_chart=sel_chart, sel_sim=sel_sim, capital=capital),
            vix=vix_data, kpis=kpis,
            charts=dict(
                cumulative=cumul_opt, candle=candle_opt, trend=trend_data, sel_chart=sel_chart,
                perf=perf_json, rolling_beta=rbeta_json,
                corr_heatmap=corr_heatmap, rolling_corr=rolling_corr_json,
                pair_scatter=pair_scatter_json, scatter_matrix=scatter_mat_json,
                cov_heatmap=cov_heatmap, vol_bar=vol_bar_json,
                mc_frontier=mc_frontier_json, rr_scatter=rr_scatter_json,
                drawdown=drawdown_json, vix_chart=vix_chart_json,
            ),
            tables=dict(risk=risk_table, corr=corr_table, cov=cov_table,
                        bench_summary=bench_summary),
            signals=signals, mc_weights=mc_weights, simulator=sim_data,
        )
        return app.response_class(
            response=json.dumps(result, cls=NpEncoder),
            mimetype='application/json')

    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


def _build_simulator(ticker: str, hist: dict, dcf_g, dcf_wacc, dcf_tg, dcf_yrs):
    info_s = fetch_info(ticker)
    if not info_s:
        return dict(ticker=ticker, name=ticker, error="No fundamental data")
    nm  = info_s.get("longName", ticker) or ticker
    fund_keys = [
        ("Market Cap",     info_s.get("marketCap"),          "cap"),
        ("P/E (TTM)",      info_s.get("trailingPE"),         "ratio"),
        ("Forward P/E",    info_s.get("forwardPE"),          "ratio"),
        ("P/B",            info_s.get("priceToBook"),        "ratio"),
        ("EV/EBITDA",      info_s.get("enterpriseToEbitda"), "ratio"),
        ("EPS (TTM)",      info_s.get("trailingEps"),        "dollar"),
        ("Forward EPS",    info_s.get("forwardEps"),         "dollar"),
        ("Revenue TTM",    info_s.get("totalRevenue"),       "cap"),
        ("Gross Margin",   info_s.get("grossMargins"),       "pct"),
        ("Op. Margin",     info_s.get("operatingMargins"),   "pct"),
        ("Net Margin",     info_s.get("profitMargins"),      "pct"),
        ("ROE",            info_s.get("returnOnEquity"),     "pct"),
        ("ROA",            info_s.get("returnOnAssets"),     "pct"),
        ("Debt/Equity",    info_s.get("debtToEquity"),       "ratio"),
        ("Current Ratio",  info_s.get("currentRatio"),       "ratio"),
        ("Free Cash Flow", info_s.get("freeCashflow"),       "cap"),
        ("Div. Yield",     info_s.get("dividendYield"),      "pct"),
        ("Beta",           info_s.get("beta"),               "ratio"),
        ("52W High",       info_s.get("fiftyTwoWeekHigh"),   "dollar"),
        ("52W Low",        info_s.get("fiftyTwoWeekLow"),    "dollar"),
    ]
    def _fmt(val, typ):
        if val is None: return "N/A"
        try:
            if typ=="cap":
                v = abs(val)
                if v >= 1e12: return f"${v/1e12:.2f}T"
                if v >= 1e9:  return f"${v/1e9:.2f}B"
                return f"${v/1e6:.2f}M"
            if typ=="pct":    return f"{val:.2%}"
            if typ=="dollar": return f"${val:.2f}"
            return f"{val:.2f}"
        except Exception: return "N/A"
    fund_map = [dict(label=lbl, value=_fmt(val,typ)) for lbl,val,typ in fund_keys]
    dcf = simple_dcf(info_s, dcf_g, dcf_wacc, dcf_tg, dcf_yrs)
    dcf_clean = None
    if dcf:
        up = dcf["upside"] or 0
        if up > 0.15:    tag,cls2 = "UNDERVALUED - DCF signals meaningful upside","buy"
        elif up < -0.15: tag,cls2 = "OVERVALUED - DCF signals downside vs intrinsic value","sell"
        else:            tag,cls2 = "FAIRLY VALUED - Priced near DCF intrinsic value","hold"
        dcf_clean = dict(iv=round(dcf["iv"],2), price=round(dcf["price"] or 0,2),
                         upside=round(up*100,1), pv_flows=round(dcf["pv_flows"]/1e9,2),
                         terminal=round(dcf["terminal"]/1e9,2), total=round(dcf["total"]/1e9,2),
                         tag=tag, cls=cls2)
    eps_chart_json = None
    if PLOTLY_OK:
        try:
            ed = yf.Ticker(ticker).earnings_dates
            if ed is not None and not ed.empty and "Reported EPS" in ed.columns:
                clean = ed.dropna(subset=["Reported EPS"]).head(8)
                if not clean.empty:
                    c_eps = ["#00e87a" if v >= 0 else "#ff4545" for v in clean["Reported EPS"]]
                    fig_eps = go.Figure(go.Bar(
                        x=[str(d.date()) for d in clean.index], y=clean["Reported EPS"],
                        marker_color=c_eps, text=[f"${v:.2f}" for v in clean["Reported EPS"]],
                        textposition="outside"))
                    fig_eps.update_layout(title="Reported EPS by Quarter", **_dark_layout(height=300))
                    eps_chart_json = _fig_json(fig_eps)
        except Exception: pass
    return dict(ticker=ticker, name=nm,
                sector=info_s.get("sector","N/A"), industry=info_s.get("industry","N/A"),
                country=info_s.get("country","N/A"),
                fund_map=fund_map, dcf=dcf_clean, eps_chart=eps_chart_json)


@app.route('/api/candle', methods=['POST'])
def candle_chart():
    try:
        p           = request.get_json(force=True)
        ticker      = p.get('ticker','').strip().upper()
        start_str   = p.get('start'); end_str = p.get('end'); freq = p.get('freq', '1d')
        ema_cfg     = p.get('ema_cfg', {"7":True,"30":True,"50":True,"200":True})
        custom_emas = [int(x) for x in p.get('custom_emas', []) if str(x).isdigit()]
        show_bb     = bool(p.get('show_bb', True)); show_sigs = bool(p.get('show_sigs', True))
        if not ticker: return jsonify({'ok':False,'error':'No ticker'}), 400
        hist = fetch_hist((ticker,), start_str, end_str, freq)
        if ticker not in hist: return jsonify({'ok':False,'error':'No data'}), 400
        df   = hist[ticker]
        tfk  = trend_follow_signals(df) if show_sigs else None
        sigs = ml_combo_score(df, tfk) if (show_sigs and tfk) else None
        fc   = forecast_3m(df["Close"])
        opt  = echarts_candle(df, ticker, ema_cfg, custom_emas, show_bb, show_sigs, sigs, fc, tfk)
        trend_data = {}
        if tfk:
            trend_data = dict(
                score=int(tfk["trend_score"].dropna().iloc[-1]) if len(tfk["trend_score"].dropna()) else 0,
                weekly_up=bool(tfk["weekly_up"].dropna().iloc[-1]) if len(tfk["weekly_up"].dropna()) else False,
                daily_up= bool(tfk["daily_up"].dropna().iloc[-1])  if len(tfk["daily_up"].dropna())  else False,
                swing_up= bool(tfk["swing_up"].dropna().iloc[-1])  if len(tfk["swing_up"].dropna())  else False,
                short_up= bool(tfk["short_up"].dropna().iloc[-1])  if len(tfk["short_up"].dropna())  else False,
                candle=str(tfk["candle_color"].dropna().iloc[-1])   if len(tfk["candle_color"].dropna()) else "default",
                rsi=round(float(tfk["rsi14"].dropna().iloc[-1]),1)  if len(tfk["rsi14"].dropna()) else 50.0,
            )
        return app.response_class(
            response=json.dumps({'ok':True,'candle':opt,'trend':trend_data}, cls=NpEncoder),
            mimetype='application/json')
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)}), 500


@app.route('/api/simulator', methods=['POST'])
def simulator_tab():
    try:
        p       = request.get_json(force=True)
        ticker  = p.get('ticker','').strip().upper()
        start_s = p.get('start'); end_s = p.get('end'); freq = p.get('freq','1d')
        dcf_g   = float(p.get('dcf_g',   0.10)); dcf_wacc = float(p.get('dcf_wacc', 0.10))
        dcf_tg  = float(p.get('dcf_tg',  0.03)); dcf_yrs  = int(p.get('dcf_yrs', 5))
        if not ticker: return jsonify({'ok':False,'error':'No ticker'}), 400
        hist = fetch_hist((ticker,), start_s, end_s, freq)
        data = _build_simulator(ticker, hist, dcf_g, dcf_wacc, dcf_tg, dcf_yrs)
        return app.response_class(
            response=json.dumps({'ok':True,'simulator':data}, cls=NpEncoder),
            mimetype='application/json')
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)}), 500


@app.route('/api/pair_scatter', methods=['POST'])
def pair_scatter_endpoint():
    try:
        p       = request.get_json(force=True)
        t1      = p.get('t1','').upper(); t2 = p.get('t2','').upper()
        start_s = p.get('start'); end_s = p.get('end'); freq = p.get('freq','1d')
        if not t1 or not t2: return jsonify({'ok':False,'error':'Need t1 and t2'}), 400
        hist       = fetch_hist(tuple(sorted([t1,t2])), start_s, end_s, freq)
        prices_df  = pd.DataFrame({t: hist[t]["Close"] for t in [t1,t2] if t in hist}).dropna(how="all")
        returns_df = prices_df.pct_change().dropna()
        fig = plotly_pair_scatter(returns_df, t1, t2) if PLOTLY_OK else None
        return app.response_class(
            response=json.dumps({'ok':True,'pair_scatter':_fig_json(fig)}, cls=NpEncoder),
            mimetype='application/json')
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)}), 500


@app.route('/api/download', methods=['POST'])
def download():
    try:
        p       = request.get_json(force=True)
        tickers = [t.strip().upper() for t in p.get('tickers',[]) if t.strip()]
        start_s = p.get('start'); end_s = p.get('end'); freq = p.get('freq','1d')
        rfr     = float(p.get('rfr', 0.0457)); conf = float(p.get('conf', 0.95))
        bench_tick = p.get('bench_ticker','^GSPC')
        hist      = fetch_hist(tuple(tickers), start_s, end_s, freq)
        bench_raw = fetch_hist((bench_tick,), start_s, end_s, freq)
        valid     = [t for t in tickers if t in hist]
        if not valid: return jsonify({'ok':False,'error':'No data'}), 400
        prices_df  = pd.DataFrame({t: hist[t]["Close"] for t in valid}).dropna(how="all")
        bench_ser  = bench_raw.get(bench_tick, {})
        bench_ser  = bench_ser.get("Close", pd.Series(dtype=float)) if isinstance(bench_ser, dict) else pd.Series(dtype=float)
        bench_ret  = bench_raw[bench_tick]["Close"].pct_change().dropna() if bench_tick in bench_raw else pd.Series(dtype=float)
        returns_df = prices_df.pct_change().dropna()
        rows = []
        for t in valid:
            r = returns_df[t].dropna(); pr = prices_df[t].dropna()
            rows.append({"Ticker":t, "Ann. Return":_safe(ann_return(r)), "Ann. Volatility":_safe(ann_vol(r)),
                "Sharpe Ratio":_safe(sharpe(r,rfr)), "Sortino Ratio":_safe(sortino(r,rfr)),
                "Max Drawdown":_safe(max_dd(pr)), "Beta":_safe(beta_calc(r,bench_ret)),
                "Alpha":_safe(alpha_calc(r,bench_ret,rfr)),
                f"VaR {int(conf*100)}%":_safe(var_hist(r,conf)),
                f"CVaR {int(conf*100)}%":_safe(cvar_hist(r,conf)),
                "Treynor":_safe(treynor(r,bench_ret,rfr)), "Calmar":_safe(calmar(r,pr)),
                "Info. Ratio":_safe(info_ratio(r,bench_ret.reindex(r.index).dropna()))})
        risk_df = pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()
        xls     = build_excel(hist, risk_df, returns_df)
        fname   = f"analysis_{'_'.join(valid[:5])}_{datetime.today().strftime('%Y%m%d')}.xlsx"
        return send_file(io.BytesIO(xls),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True, download_name=fname)
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
