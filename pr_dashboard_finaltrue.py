"""
pr_dashboard.py — CT104-3-M Pattern Recognition Dashboard
Dark terminal aesthetic · 8-phase pipeline · DTW K-Means
Train (2006-2019) / Validation (2020-2026) split
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings, random
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Pattern Recognition",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS — dark terminal aesthetic matching the screenshots
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:        #111214;
  --surface:   #1a1c1f;
  --border:    #2a2d32;
  --accent:    #4ade80;
  --accent2:   #22d3ee;
  --accent3:   #f59e0b;
  --muted:     #6b7280;
  --text:      #e2e8f0;
  --text-dim:  #94a3b8;
  --red:       #f87171;
  --mono:      'IBM Plex Mono', monospace;
  --sans:      'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--sans) !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1400px; }

.dash-header {
  display: flex; align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 1.5rem; padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}
.dash-title {
  font-family: var(--mono) !important; font-size: 1.4rem; font-weight: 600;
  color: var(--text); letter-spacing: -0.02em; margin: 0 0 0.25rem 0;
}
.dash-sub { font-family: var(--mono) !important; font-size: 0.78rem; color: var(--muted); margin: 0; }
.badge-complete {
  background: var(--accent); color: #000;
  font-family: var(--mono) !important; font-size: 0.7rem; font-weight: 600;
  padding: 0.25rem 0.65rem; border-radius: 3px; letter-spacing: 0.08em;
}
.badge-idle {
  background: var(--border); color: var(--muted);
  font-family: var(--mono) !important; font-size: 0.7rem; font-weight: 600;
  padding: 0.25rem 0.65rem; border-radius: 3px; letter-spacing: 0.08em;
}
.control-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;
}
.control-label {
  font-family: var(--mono) !important; font-size: 0.7rem; font-weight: 500;
  color: var(--muted); letter-spacing: 0.1em; margin-bottom: 0.6rem;
}
.ticker-grid { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.25rem; }
.ticker-chip {
  font-family: var(--mono) !important; font-size: 0.78rem; font-weight: 500;
  padding: 0.3rem 0.7rem; border-radius: 4px; cursor: pointer;
  border: 1px solid var(--border); background: var(--bg); color: var(--text-dim);
}
.ticker-chip.active { background: var(--accent); color: #000; border-color: var(--accent); }

div[data-testid="stSlider"] label {
  font-family: var(--mono) !important; font-size: 0.75rem !important;
  color: var(--muted) !important; font-weight: 500 !important; letter-spacing: 0.06em !important;
}
.stMultiSelect span[data-baseweb="tag"] {
  background: var(--accent) !important; color: #000 !important;
  font-family: var(--mono) !important; font-size: 0.75rem !important; font-weight: 500 !important;
}
div[data-testid="stButton"] > button {
  background: var(--surface) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important;
  font-family: var(--mono) !important; font-size: 0.85rem !important;
  font-weight: 500 !important; letter-spacing: 0.05em !important;
  border-radius: 5px !important; padding: 0.65rem 1.5rem !important;
  width: 100% !important; transition: all 0.15s !important;
}
div[data-testid="stButton"] > button:hover {
  border-color: var(--accent) !important; color: var(--accent) !important;
  background: rgba(74,222,128,0.05) !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
  background: rgba(74,222,128,0.08) !important;
  border-color: var(--accent) !important; color: var(--accent) !important;
}
.progress-wrap { margin: 0.75rem 0 0.5rem 0; }
.progress-label {
  font-family: var(--mono) !important; font-size: 0.72rem; color: var(--muted);
  display: flex; justify-content: space-between; margin-bottom: 0.3rem;
}
.progress-bar-outer { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
.progress-bar-inner { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.4s ease; }

.phase-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 1.25rem 1.5rem; margin-top: 0.5rem;
}
.phase-card-title {
  font-family: var(--mono) !important; font-size: 0.7rem; color: var(--muted);
  letter-spacing: 0.1em; margin-bottom: 1rem;
}
.stat-tile { padding: 0; }
.stat-label {
  font-family: var(--mono) !important; font-size: 0.65rem; color: var(--muted);
  letter-spacing: 0.1em; margin-bottom: 0.3rem;
}
.stat-value {
  font-family: var(--mono) !important; font-size: 1.6rem; font-weight: 600;
  color: var(--text); line-height: 1.1; letter-spacing: -0.03em;
}
.stat-sub { font-family: var(--sans) !important; font-size: 0.72rem; color: var(--muted); margin-top: 0.2rem; }
.stat-divider { width: 100%; height: 1px; background: var(--border); margin: 1rem 0; }
.verdict-box {
  background: rgba(74,222,128,0.07); border: 1px solid rgba(74,222,128,0.25);
  border-radius: 5px; padding: 0.85rem 1.1rem;
  font-family: var(--mono) !important; font-size: 0.82rem; color: var(--accent); margin: 0.75rem 0;
}
.verdict-box.warn { background: rgba(245,158,11,0.07); border-color: rgba(245,158,11,0.25); color: var(--accent3); }
.verdict-box.info { background: rgba(34,211,238,0.06); border-color: rgba(34,211,238,0.2); color: var(--accent2); }
.section-head {
  font-family: var(--mono) !important; font-size: 0.68rem; color: var(--muted);
  letter-spacing: 0.12em; text-transform: uppercase;
  margin: 1.25rem 0 0.6rem 0; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);
}
.winner-card {
  background: linear-gradient(135deg, rgba(74,222,128,0.08) 0%, rgba(34,211,238,0.05) 100%);
  border: 1px solid rgba(74,222,128,0.3); border-radius: 6px; padding: 1.1rem 1.4rem; margin: 0.75rem 0;
}
.winner-label { font-family: var(--mono) !important; font-size: 0.65rem; color: var(--accent); letter-spacing: 0.12em; margin-bottom: 0.4rem; }
.winner-value { font-family: var(--mono) !important; font-size: 1.8rem; font-weight: 600; color: var(--text); letter-spacing: -0.03em; }
.winner-sub { font-family: var(--sans) !important; font-size: 0.78rem; color: var(--text-dim); margin-top: 0.3rem; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS — tickers loaded dynamically from CSV (not hardcoded)
# ─────────────────────────────────────────────────────────────────────────────
_CSV_DEFAULT = "top_companies_20y_daily_combined.csv"

@st.cache_data(show_spinner=False)
def _load_all_tickers(csv_path: str):
    """Read only the Ticker column — fast, and cached after first load."""
    try:
        tickers = (
            pd.read_csv(csv_path, usecols=["Ticker"])["Ticker"]
            .str.strip()
            .dropna()
            .unique()
            .tolist()
        )
        return sorted(tickers)
    except Exception:
        return ["AAPL","MSFT","NVDA","AMZN","TSLA","GOOGL"]

ALL_TICKERS     = _load_all_tickers(_CSV_DEFAULT)
DEFAULT_TICKERS = ["AAPL","MSFT","NVDA","AMZN","TSLA","GOOGL"]
PHASE_NAMES = [
    "1. Ingestion","2. Features","3. Windowing","3b. PCA",
    "4. Elbow","5. DTW Clusters","6. Robustness","7. TA-Lib","8. Evaluation",
]

for k, v in {
    "selected_tickers": DEFAULT_TICKERS,
    "pipeline_done": False,
    "active_phase": 0,
    "results": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
badge_html = (
    '<span class="badge-complete">COMPLETE</span>'
    if st.session_state.pipeline_done else
    '<span class="badge-idle">IDLE</span>'
)
st.markdown(f"""
<div class="dash-header">
  <div>
    <div class="dash-title">Stock pattern recognition</div>
    <div class="dash-sub">CT104-3-M &middot; DTW K-Means &middot; SMA vs EMA &middot; Train 2006–2019 · Val 2020–2026</div>
  </div>
  {badge_html}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONTROL PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="control-card">', unsafe_allow_html=True)
st.markdown('<div class="control-label">TICKERS</div>', unsafe_allow_html=True)
selected = st.multiselect(
    "tickers_hidden", options=ALL_TICKERS, default=st.session_state.selected_tickers,
    label_visibility="collapsed", key="ticker_select",
)
st.session_state.selected_tickers = selected

col_w, col_k, col_t, _ = st.columns([2, 2, 2, 0.1])
with col_w:
    WINDOW_SIZE = st.slider("WINDOW", 3, 30, 10, key="window_sz")
with col_k:
    K_MAX = st.slider("K MAX SEARCH", 5, 20, 15, key="k_max")
with col_t:
    WIN_THRESH = st.slider("WIN THRESH", 50.0, 70.0, 55.0, step=0.5, key="win_thresh", format="%.1f%%")

# ── K selection mode toggle ────────────────────────────────────────────────────
tk_col, ks_col = st.columns([1, 3])
with tk_col:
    k_mode = st.toggle("Manual K", value=False, key="k_manual_mode",
                       help="OFF = auto-select K via Elbow + Silhouette  |  ON = you choose K directly")
with ks_col:
    if k_mode:
        MANUAL_K = st.slider(
            "FINAL PATTERNS (K)", min_value=2, max_value=st.session_state.k_max,
            value=min(8, st.session_state.k_max), step=1, key="k_manual_val",
        )
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
            f'color:#f59e0b;margin-top:2px;">'
            f'▶ Will cluster into exactly <b style="color:#f59e0b">{MANUAL_K}</b> patterns '
            f'— elbow/silhouette search skipped</div>',
            unsafe_allow_html=True)
    else:
        MANUAL_K = None
        st.markdown(
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
            'color:#4ade80;margin-top:6px;">'
            '▶ Auto mode — Kneedle elbow + silhouette will determine optimal K</div>',
            unsafe_allow_html=True)

with st.expander("Advanced", expanded=False):
    CSV_PATH = st.text_input("CSV path", value=_CSV_DEFAULT, key="csv_path")
    # Reload ticker list whenever the path changes
    if CSV_PATH != _CSV_DEFAULT:
        ALL_TICKERS = _load_all_tickers(CSV_PATH)
    skip_talib = st.checkbox("Skip TA-Lib scan (Phase 7)", value=False, key="skip_talib")

run_col, _ = st.columns([3, 1])
with run_col:
    run_btn = st.button("\u2197 Run pipeline", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.pipeline_done = False
    st.session_state.results = {}
    R = st.session_state.results
    prog_area = st.empty()

    def update_prog(pct, label):
        prog_area.markdown(f"""
        <div class="progress-wrap">
          <div class="progress-label"><span>{label}</span><span>{pct}%</span></div>
          <div class="progress-bar-outer"><div class="progress-bar-inner" style="width:{pct}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

    # Phase 1 — Ingestion
    update_prog(5, "Phase 1 — Ingestion")
    try:
        df = pd.read_csv(st.session_state.csv_path)
        df["Date"]   = pd.to_datetime(df["Date"])
        df["Ticker"] = df["Ticker"].str.strip()
        df           = df.sort_values(["Ticker","Date"]).reset_index(drop=True)
        init_rows    = len(df)
        df           = df.drop_duplicates()
        dups_removed = init_rows - len(df)
        price_cols   = ["Open","High","Low","Close","Adj Close"]
        for col in price_cols:
            df[col] = df.groupby("Ticker")[col].transform(lambda x: x.ffill().bfill())
        df[price_cols] = df[price_cols].map(lambda x: np.nan if x <= 0 else x)
        for col in price_cols:
            df[col] = df.groupby("Ticker")[col].transform(lambda x: x.ffill().bfill())
        R["p1"] = dict(
            init_rows=init_rows, dups=dups_removed, final_rows=len(df),
            missing=int(df.isnull().sum().sum()), tickers=df["Ticker"].nunique(),
            date_min=str(df["Date"].min().date()), date_max=str(df["Date"].max().date()),
            all_tickers=sorted(df["Ticker"].unique().tolist()),
        )
    except FileNotFoundError:
        prog_area.empty()
        st.error(f"CSV not found: `{st.session_state.csv_path}`")
        st.stop()

    # Phase 2 — Features (with split)
    update_prog(18, "Phase 2 — Feature Engineering")
    tickers = [t for t in st.session_state.selected_tickers if t in R["p1"]["all_tickers"]]
    if not tickers:
        st.error("None of the selected tickers exist in the CSV.")
        st.stop()
    df_f = df[df["Ticker"].isin(tickers)].copy()
    df_f["SMA_10"]           = df_f.groupby("Ticker")["Close"].transform(lambda x: x.rolling(10).mean())
    df_f["SMA_50"]           = df_f.groupby("Ticker")["Close"].transform(lambda x: x.rolling(50).mean())
    df_f["Trend_Signal_SMA"] = df_f["SMA_10"] - df_f["SMA_50"]
    df_f["EMA_10"]           = df_f.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=10,adjust=False).mean())
    df_f["EMA_50"]           = df_f.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=50,adjust=False).mean())
    df_f["Trend_Signal_EMA"] = df_f["EMA_10"] - df_f["EMA_50"]
    df_f["Future_Close_3D"]  = df_f.groupby("Ticker")["Close"].transform(lambda x: x.shift(-3))
    df_final = df_f.dropna().reset_index(drop=True)

    # ── Split into train (2006-2019) and validation (2020-2026) ─────────────
    df_train = df_final[df_final["Date"].dt.year <= 2019].copy()
    df_val   = df_final[df_final["Date"].dt.year >= 2020].copy()
    R["p2"] = dict(
        tickers=tickers,
        clean_rows=len(df_final),
        train_rows=len(df_train),
        val_rows=len(df_val),
        train_date_range=(str(df_train["Date"].min().date()), str(df_train["Date"].max().date())),
        val_date_range=(str(df_val["Date"].min().date()), str(df_val["Date"].max().date())),
    )

    # Phase 3 — Windowing (train & val separately)
    update_prog(32, "Phase 3 — Sequential Windowing")
    ws = st.session_state.window_sz

    def build_windows(data_df):
        X_sma, X_ema, X_ohlc, y_tgt, dates_info = [], [], [], [], []
        for ticker, grp in data_df.groupby("Ticker"):
            sig_s = grp["Trend_Signal_SMA"].values
            sig_e = grp["Trend_Signal_EMA"].values
            o_arr = grp["Open"].values;   h_arr = grp["High"].values
            l_arr = grp["Low"].values;    c_arr = grp["Close"].values
            fut   = grp["Future_Close_3D"].values
            dts   = grp["Date"].dt.strftime("%Y-%m-%d").values
            if len(sig_s) < ws: continue
            for i in range(len(sig_s) - ws):
                X_sma.append(sig_s[i:i+ws])
                X_ema.append(sig_e[i:i+ws])
                X_ohlc.append(np.column_stack((o_arr[i:i+ws],h_arr[i:i+ws],l_arr[i:i+ws],c_arr[i:i+ws])))
                y_tgt.append(1 if fut[i+ws-1] > c_arr[i+ws-1] else 0)
                dates_info.append((ticker, dts[i+ws-1]))
        return (np.array(X_sma).reshape(-1, ws, 1),
                np.array(X_ema).reshape(-1, ws, 1),
                np.array(X_ohlc),
                np.array(y_tgt),
                dates_info)

    X_sma_train, X_ema_train, X_ohlc_train, y_train, dates_train = build_windows(df_train)
    X_sma_val,   X_ema_val,   X_ohlc_val,   y_val,   dates_val   = build_windows(df_val)

    R["p3"] = dict(
    train=dict(sma_shape=X_sma_train.shape, ema_shape=X_ema_train.shape, ohlc_shape=X_ohlc_train.shape,
               total=len(X_sma_train), up_ratio=float(y_train.mean()), bull_count=int(y_train.sum()), ws=ws),
    val=dict(sma_shape=X_sma_val.shape, ema_shape=X_ema_val.shape, ohlc_shape=X_ohlc_val.shape,
             total=len(X_sma_val), up_ratio=float(y_val.mean()), bull_count=int(y_val.sum()), ws=ws,
             dates=dates_val,   # <── add this
             y=y_val),          # <── add this
            )

    # Phase 3b — PCA Dimensionality Reduction (fit on train, transform both)
    update_prog(42, "Phase 3b — PCA Dimensionality Reduction")
    from sklearn.decomposition import PCA
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    # Two separate scalers — fitting EMA scaler on SMA data (or vice-versa) would
    # apply wrong mean/variance statistics; each signal space needs its own fit.
    sma_scaler = TimeSeriesScalerMeanVariance()
    ema_scaler = TimeSeriesScalerMeanVariance()
    X_sma_norm_train = sma_scaler.fit_transform(X_sma_train)
    X_ema_norm_train = ema_scaler.fit_transform(X_ema_train)
    X_sma_norm_val   = sma_scaler.transform(X_sma_val)
    X_ema_norm_val   = ema_scaler.transform(X_ema_val)

    # Flatten for PCA
    X_sma_flat_train = X_sma_norm_train.reshape(X_sma_norm_train.shape[0], -1)
    X_ema_flat_train = X_ema_norm_train.reshape(X_ema_norm_train.shape[0], -1)
    X_sma_flat_val   = X_sma_norm_val.reshape(X_sma_norm_val.shape[0], -1)
    X_ema_flat_val   = X_ema_norm_val.reshape(X_ema_norm_val.shape[0], -1)

    # 95% variance PCA — for silhouette
    pca_sma = PCA(n_components=0.95, random_state=42)
    X_sma_pca_train = pca_sma.fit_transform(X_sma_flat_train)
    X_sma_pca_val   = pca_sma.transform(X_sma_flat_val)

    pca_ema = PCA(n_components=0.95, random_state=42)
    X_ema_pca_train = pca_ema.fit_transform(X_ema_flat_train)
    X_ema_pca_val   = pca_ema.transform(X_ema_flat_val)

    # 2-component PCA — for visualisation
    pca_2d = PCA(n_components=2, random_state=42)
    X_sma_2d_vis_train = pca_2d.fit_transform(X_sma_flat_train)
    X_sma_2d_vis_val   = pca_2d.transform(X_sma_flat_val)
    X_ema_2d_vis_train = pca_2d.transform(X_ema_flat_train)
    X_ema_2d_vis_val   = pca_2d.transform(X_ema_flat_val)

    R["p3b"] = dict(
        n_comp_sma=int(pca_sma.n_components_),
        n_comp_ema=int(pca_ema.n_components_),
        var_total_sma=float(pca_sma.explained_variance_ratio_.sum() * 100),
        var_total_ema=float(pca_ema.explained_variance_ratio_.sum() * 100),
        individual=pca_sma.explained_variance_ratio_.tolist(),
        cumulative=(np.cumsum(pca_sma.explained_variance_ratio_) * 100).tolist(),
        vis_var=(pca_2d.explained_variance_ratio_ * 100).tolist(),
        window_size=ws,
        # Store arrays needed by later phases
        X_sma_pca_train=X_sma_pca_train,
        X_sma_2d_vis_train=X_sma_2d_vis_train,
        X_sma_2d_vis_val=X_sma_2d_vis_val,
        X_ema_2d_vis_train=X_ema_2d_vis_train,
        X_ema_2d_vis_val=X_ema_2d_vis_val,
        pca_2d=pca_2d,
        X_sma_norm_train=X_sma_norm_train,
        X_ema_norm_train=X_ema_norm_train,
        X_sma_norm_val=X_sma_norm_val,
        X_ema_norm_val=X_ema_norm_val,
        X_sma_flat_train=X_sma_flat_train,
    )

    # Phase 4 — Hyperparameter Tuning (on train)
    from tslearn.clustering import TimeSeriesKMeans
    from sklearn.metrics import silhouette_score

    _manual_mode = st.session_state.get("k_manual_mode", False)
    _manual_k    = st.session_state.get("k_manual_val", 8)

    if _manual_mode:
        update_prog(55, f"Phase 4 — Manual K={_manual_k} (search skipped)")
        OPTIMAL_K = _manual_k
        R["p4"] = dict(mode="manual", optimal_k=OPTIMAL_K)
    else:
        update_prog(55, "Phase 4 — Elbow + Silhouette Search (train)")
        try:
            from kneed import KneeLocator; has_kneed = True
        except ImportError:
            has_kneed = False

        K_RANGE    = range(2, st.session_state.k_max + 1)
        inertias, silhouettes = [], []
        for k in K_RANGE:
            km = TimeSeriesKMeans(n_clusters=k, metric="euclidean", n_init=1,
                                  max_iter=30, random_state=42, verbose=False)
            labels = km.fit_predict(X_sma_norm_train)
            inertias.append(km.inertia_)
            sil = silhouette_score(X_sma_pca_train, labels,
                                   sample_size=min(5000, len(X_sma_pca_train)), random_state=42)
            silhouettes.append(sil)

        if has_kneed:
            kl = KneeLocator(list(K_RANGE), inertias, curve="convex", direction="decreasing")
            elbow_k = kl.knee if kl.knee else list(K_RANGE)[int(np.argmax(np.diff(np.diff(inertias)))) + 1]
        else:
            elbow_k = list(K_RANGE)[int(np.argmax(np.diff(np.diff(inertias)))) + 1]
        sil_k = list(K_RANGE)[int(np.argmax(silhouettes))]
        if elbow_k == sil_k:
            OPTIMAL_K = elbow_k; agree = True
        else:
            OPTIMAL_K = max(2, round((elbow_k + sil_k) / 2)); agree = False
        R["p4"] = dict(mode="auto",
                       k_range=list(K_RANGE), inertias=inertias, silhouettes=silhouettes,
                       elbow_k=elbow_k, sil_k=sil_k, optimal_k=OPTIMAL_K, agree=agree)

    # Phase 5 — DTW Clustering (fit on train, predict on val)
    update_prog(72, f"Phase 5 — DTW K-Means (K={OPTIMAL_K})")
    dtw_p = dict(n_clusters=OPTIMAL_K, metric="dtw", init="k-means++", n_init=2, max_iter=10, verbose=False, random_state=42)
    km_sma  = TimeSeriesKMeans(**dtw_p); lbl_sma_train = km_sma.fit_predict(X_sma_norm_train)
    km_ema  = TimeSeriesKMeans(**dtw_p); lbl_ema_train = km_ema.fit_predict(X_ema_norm_train)

    lbl_sma_val = km_sma.predict(X_sma_norm_val)
    lbl_ema_val = km_ema.predict(X_ema_norm_val)

    # Project centroids into 2D PCA space for scatter
    centers_flat = km_sma.cluster_centers_.reshape(OPTIMAL_K, -1)
    centers_2d   = pca_2d.transform(centers_flat)

    R["p5"] = dict(
        lbl_sma_train=lbl_sma_train, lbl_ema_train=lbl_ema_train,
        lbl_sma_val=lbl_sma_val, lbl_ema_val=lbl_ema_val,
        centers_sma=km_sma.cluster_centers_, optimal_k=OPTIMAL_K,
        centers_2d=centers_2d,
    )

    # Phase 6 — Robustness (train only, 2006-2019)
    update_prog(80, "Phase 6 — Robustness Profiling (train set)")
    res_train = pd.DataFrame(dates_train, columns=["Ticker","Date"])
    res_train["Date"]        = pd.to_datetime(res_train["Date"])
    res_train["Year"]        = res_train["Date"].dt.year
    res_train["Target"]      = y_train
    res_train["Cluster_SMA"] = lbl_sma_train
    res_train["Cluster_EMA"] = lbl_ema_train
    years_train = sorted(res_train["Year"].unique())
    clusters_train = sorted(range(OPTIMAL_K))

    def build_matrix(df_in, col):
        rows = []
        for c in clusters_train:
            row  = {"Cluster": c}
            mask = df_in[col] == c
            for y in years_train:
                sub = df_in[mask & (df_in["Year"] == y)]
                row[str(y)] = round(sub["Target"].mean()*100, 1) if len(sub) > 0 else np.nan
            row["Overall%"] = round(df_in[mask]["Target"].mean()*100, 2)
            row["N"] = int(mask.sum())
            rows.append(row)
        return pd.DataFrame(rows).set_index("Cluster").sort_values("Overall%", ascending=False)

    mat_sma_train = build_matrix(res_train, "Cluster_SMA")
    mat_ema_train = build_matrix(res_train, "Cluster_EMA")
    # Validation robustness matrix (built here so Phase 6 display + Phase 8 can use it)
    res_val_rob = pd.DataFrame(dates_val, columns=["Ticker","Date"])
    res_val_rob["Date"]        = pd.to_datetime(res_val_rob["Date"])
    res_val_rob["Year"]        = res_val_rob["Date"].dt.year
    res_val_rob["Target"]      = y_val
    res_val_rob["Cluster_SMA"] = lbl_sma_val
    res_val_rob["Cluster_EMA"] = lbl_ema_val
    years_val_rob = sorted(res_val_rob["Year"].unique())

    def build_matrix_for(df_in, col, yrs, clust):
        rows = []
        for c in clust:
            row = {"Cluster": c}
            mask = df_in[col] == c
            for y in yrs:
                sub = df_in[mask & (df_in["Year"] == y)]
                row[str(y)] = round(sub["Target"].mean()*100, 1) if len(sub) > 0 else np.nan
            row["Overall%"] = round(df_in[mask]["Target"].mean()*100, 2) if mask.sum() > 0 else np.nan
            row["N"] = int(mask.sum())
            rows.append(row)
        return pd.DataFrame(rows).set_index("Cluster").sort_values("Overall%", ascending=False)

    mat_sma_val_rob = build_matrix_for(res_val_rob, "Cluster_SMA", years_val_rob, clusters_train)
    mat_ema_val_rob = build_matrix_for(res_val_rob, "Cluster_EMA", years_val_rob, clusters_train)

    R["p6"] = dict(results_df=res_train, mat_sma=mat_sma_train, mat_ema=mat_ema_train,
                   years=years_train, clusters=clusters_train,
                   results_val=res_val_rob, mat_sma_val=mat_sma_val_rob, mat_ema_val=mat_ema_val_rob,
                   years_val=years_val_rob)

    # Phase 7 — TA-Lib on validation windows using training-winning clusters
    update_prog(90, "Phase 7 — TA-Lib Validation (validation set)")
    R["p7"] = dict(skipped=st.session_state.skip_talib,
                   matches_sma=None, summary_sma=None,
                   matches_ema=None, summary_ema=None)
    if not st.session_state.skip_talib:
        try:
            import talib
            thresh      = st.session_state.win_thresh
            win_ids_sma = list(mat_sma_train[mat_sma_train["Overall%"] > thresh].index)
            win_ids_ema = list(mat_ema_train[mat_ema_train["Overall%"] > thresh].index)
            talib_pats = {
                "Morning Star": talib.CDLMORNINGSTAR, "Engulfing":    talib.CDLENGULFING,
                "Hammer":       talib.CDLHAMMER,      "Doji":         talib.CDLDOJI,
                "Shooting Star":talib.CDLSHOOTINGSTAR,"Harami":       talib.CDLHARAMI,
                "Three White":  talib.CDL3WHITESOLDIERS,
            }
            res_val_t = pd.DataFrame(dates_val, columns=["Ticker","Date"])
            res_val_t["Date"] = pd.to_datetime(res_val_t["Date"])

            def _scan(lbl_val, win_ids, label):
                matches = []
                for idx, lb in enumerate(lbl_val):
                    if lb not in win_ids: continue
                    win = X_ohlc_val[idx]
                    o2, h2, l2, c2 = (win[:, j].astype(float) for j in range(4))
                    for name, fn in talib_pats.items():
                        try:
                            r2 = fn(o2, h2, l2, c2)
                            if r2[-1] != 0:
                                matches.append({"Cluster":int(lb),"Pattern":name,
                                                "Signal":int(r2[-1]),
                                                "Ticker":res_val_t.iloc[idx]["Ticker"]})
                        except Exception:
                            pass
                return matches

            sma_hits = _scan(lbl_sma_val, win_ids_sma, "SMA")
            ema_hits = _scan(lbl_ema_val, win_ids_ema, "EMA")

            R["p7"].update(
                win_ids_sma=win_ids_sma, win_ids_ema=win_ids_ema,
                matches_sma=pd.DataFrame(sma_hits) if sma_hits else None,
                summary_sma=pd.DataFrame(sma_hits).groupby(["Cluster","Pattern"]).size().unstack(fill_value=0) if sma_hits else None,
                matches_ema=pd.DataFrame(ema_hits) if ema_hits else None,
                summary_ema=pd.DataFrame(ema_hits).groupby(["Cluster","Pattern"]).size().unstack(fill_value=0) if ema_hits else None,
            )
        except ImportError:
            R["p7"]["skipped"] = "no_talib"

    # Phase 8 — Evaluation on validation set
    update_prog(98, "Phase 8 — Final Evaluation (validation set)")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    res_val = pd.DataFrame(dates_val, columns=["Ticker","Date"])
    res_val["Date"]        = pd.to_datetime(res_val["Date"])
    res_val["Target"]      = y_val
    res_val["Cluster_SMA"] = lbl_sma_val
    res_val["Cluster_EMA"] = lbl_ema_val

    def eval_variant(df_in, col, mat_train, label, thresh):
        wids   = list(mat_train[mat_train["Overall%"] > thresh].index)
        y_pred = df_in[col].apply(lambda x: 1 if x in wids else 0)
        y_true = df_in["Target"]
        return dict(
            label=label, winning_ids=wids, coverage=float(y_pred.mean()),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
        )
    thresh = st.session_state.win_thresh
    sma_m  = eval_variant(res_val, "Cluster_SMA", mat_sma_train, "SMA Baseline", thresh)
    ema_m  = eval_variant(res_val, "Cluster_EMA", mat_ema_train, "EMA Freshness", thresh)
    # Also evaluate on training set — delta vs val exposes overfitting
    sma_m_train = eval_variant(res_train, "Cluster_SMA", mat_sma_train, "SMA Baseline (train)", thresh)
    ema_m_train = eval_variant(res_train, "Cluster_EMA", mat_ema_train, "EMA Freshness (train)", thresh)
    overfit_sma = round(sma_m_train["f1"] - sma_m["f1"], 4)
    overfit_ema = round(ema_m_train["f1"] - ema_m["f1"], 4)
    R["p8"] = dict(
        sma=sma_m, ema=ema_m,
        sma_train=sma_m_train, ema_train=ema_m_train,
        overfit_sma=overfit_sma, overfit_ema=overfit_ema,
        winner="SMA Baseline" if sma_m["f1"] >= ema_m["f1"] else "EMA Freshness",
    )

    update_prog(100, "Pipeline complete")
    st.session_state.pipeline_done = True
    st.session_state.active_phase  = 0
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  IDLE STATE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.pipeline_done:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#6b7280;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;">
      Configure settings above and click &#8599; Run pipeline to begin.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  PERSISTENT PROGRESS BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="progress-wrap">
  <div class="progress-label"><span>Pipeline complete</span><span>100%</span></div>
  <div class="progress-bar-outer"><div class="progress-bar-inner" style="width:100%"></div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE NAVIGATION BUTTONS
# ─────────────────────────────────────────────────────────────────────────────
btn_cols = st.columns(9)
for i, (col, name) in enumerate(zip(btn_cols, PHASE_NAMES)):
    with col:
        if st.button(name, key=f"pbtn_{i}", use_container_width=True):
            st.session_state.active_phase = i
            st.rerun()

ap = st.session_state.active_phase
R  = st.session_state.results

# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY BASE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px

PL = dict(
    paper_bgcolor="#1a1c1f", plot_bgcolor="#1a1c1f",
    font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#2a2d32", linecolor="#2a2d32"),
    yaxis=dict(gridcolor="#2a2d32", linecolor="#2a2d32"),
)

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — INGESTION
# ─────────────────────────────────────────────────────────────────────────────
if ap == 0:
    p1 = R["p1"]
    st.markdown('<div class="phase-card">', unsafe_allow_html=True)
    st.markdown('<div class="phase-card-title">PHASE 1 — DATA INGESTION & PRE-PROCESSING</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        (c1, "INITIAL ROWS",      f"{p1['init_rows']:,}",    "raw CSV records"),
        (c2, "DUPLICATES REMOVED",f"{p1['dups']:,}",         "exact duplicates dropped"),
        (c3, "FINAL ROWS",        f"{p1['final_rows']:,}",   "after cleaning"),
        (c4, "MISSING VALUES",    f"{p1['missing']}",        "after ffill / bfill"),
    ]
    for col, label, val, sub in tiles:
        with col:
            st.markdown(f'<div class="stat-tile"><div class="stat-label">{label}</div><div class="stat-value">{val}</div><div class="stat-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="stat-divider"></div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">TICKERS</div><div class="stat-value">{p1["tickers"]}</div><div class="stat-sub">unique companies in dataset</div></div>', unsafe_allow_html=True)
    with c6:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">DATE RANGE</div><div class="stat-value" style="font-size:1.05rem">{p1["date_min"]} → {p1["date_max"]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head">CLEANING PIPELINE</div>', unsafe_allow_html=True)
    for label, desc in [
        ("Date parsing",      "Parsed to datetime, sorted Ticker → Date chronologically"),
        ("Duplicate removal", f"{p1['dups']:,} exact duplicate rows dropped"),
        ("Missing values",    "Per-ticker forward-fill then backward-fill on OHLCV"),
        ("Zero/neg prices",   "Converted to NaN, re-bridged with ffill/bfill"),
        ("Final check",       f"{p1['missing']} missing values remain after imputation"),
    ]:
        st.markdown(f'<div style="display:flex;gap:1rem;align-items:flex-start;margin-bottom:0.6rem;"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#4ade80;min-width:130px">✓ {label}</span><span style="font-size:0.78rem;color:#94a3b8">{desc}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — FEATURES
# ─────────────────────────────────────────────────────────────────────────────
elif ap == 1:
    p2 = R["p2"]; p1 = R["p1"]
    st.markdown('<div class="phase-card">', unsafe_allow_html=True)
    st.markdown('<div class="phase-card-title">PHASE 2 — FEATURE ENGINEERING & TARGET DEFINITION (TRAIN/VAL SPLIT)</div>', unsafe_allow_html=True)

    st.markdown('<div class="control-label">SELECTED TICKERS</div>', unsafe_allow_html=True)
    chips  = "".join(f'<span class="ticker-chip active">{t}</span>' for t in p2["tickers"])
    bad    = [t for t in st.session_state.selected_tickers if t not in p2["tickers"]]
    bchips = "".join(f'<span class="ticker-chip" style="border-color:#f87171;color:#f87171">{t} ✗</span>' for t in bad)
    st.markdown(f'<div class="ticker-grid">{chips}{bchips}</div>', unsafe_allow_html=True)
    st.markdown('<div class="stat-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">CLEAN ROWS</div><div class="stat-value">{p2["clean_rows"]:,}</div><div class="stat-sub">after dropna</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">TRAIN ROWS</div><div class="stat-value">{p2["train_rows"]:,}</div><div class="stat-sub">2006–2019</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">VAL ROWS</div><div class="stat-value">{p2["val_rows"]:,}</div><div class="stat-sub">2020–2026</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head">FEATURES ENGINEERED</div>', unsafe_allow_html=True)
    for feat, desc in [
        ("SMA_10 / SMA_50",      "10-day and 50-day simple moving averages"),
        ("Trend_Signal_SMA",     "SMA_10 − SMA_50 — baseline momentum spread"),
        ("EMA_10 / EMA_50",      "10-day and 50-day exponential moving averages"),
        ("Trend_Signal_EMA",     "EMA_10 − EMA_50 — freshness-variant spread"),
        ("Future_Close_3D",      "Close shifted −3 days (target source)"),
        ("Target (binary)",      "1 if Future_Close_3D > current Close else 0"),
    ]:
        st.markdown(f'<div style="display:flex;gap:1rem;margin-bottom:0.55rem;"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;color:#22d3ee;min-width:160px">{feat}</span><span style="font-size:0.78rem;color:#94a3b8">{desc}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3 — WINDOWING
# ─────────────────────────────────────────────────────────────────────────────
elif ap == 2:
    p3 = R["p3"]
    train_info = p3["train"]
    val_info   = p3["val"]
    st.markdown('<div class="phase-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="phase-card-title">PHASE 3 — SEQUENTIAL WINDOWING (W={train_info["ws"]})</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head">TRAINING SET (2006–2019)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">SMA TENSOR (train)</div><div class="stat-value">({train_info["sma_shape"][0]:,}, {train_info["sma_shape"][1]}, {train_info["sma_shape"][2]})</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">UP-DAY RATIO (train)</div><div class="stat-value">{train_info["up_ratio"]*100:.1f}%</div><div class="stat-sub">{train_info["bull_count"]:,} bullish</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head">VALIDATION SET (2020–2026)</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">SMA TENSOR (val)</div><div class="stat-value">({val_info["sma_shape"][0]:,}, {val_info["sma_shape"][1]}, {val_info["sma_shape"][2]})</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">UP-DAY RATIO (val)</div><div class="stat-value">{val_info["up_ratio"]*100:.1f}%</div><div class="stat-sub">{val_info["bull_count"]:,} bullish</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head">WINDOWING STRATEGY</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.8rem;color:#94a3b8;line-height:1.75">Each window is a contiguous slice of {train_info["ws"]} consecutive trading days. Ticker boundaries are respected. The binary label is determined by the closing price 3 days after the window ends. Train and validation sets are windowed separately.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3b — PCA DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────────────────────
elif ap == 3:
    p3b = R["p3b"]
    st.markdown('<div class="phase-card">', unsafe_allow_html=True)
    st.markdown('<div class="phase-card-title">PHASE 3b — PCA DIMENSIONALITY REDUCTION (fitted on train)</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">ORIGINAL DIMS</div><div class="stat-value">{p3b["window_size"]}</div><div class="stat-sub">timesteps per window</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">PCA DIMS (95%)</div><div class="stat-value">{p3b["n_comp_sma"]}</div><div class="stat-sub">components retained — SMA</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">VARIANCE KEPT</div><div class="stat-value">{p3b["var_total_sma"]:.1f}%</div><div class="stat-sub">of total SMA variance</div></div>', unsafe_allow_html=True)
    with c4:
        vis_v = p3b["vis_var"]
        st.markdown(f'<div class="stat-tile"><div class="stat-label">2D VIS COVERAGE</div><div class="stat-value">{sum(vis_v):.1f}%</div><div class="stat-sub">PC1 {vis_v[0]:.1f}% + PC2 {vis_v[1]:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="verdict-box info">→ PCA reduces each {p3b["window_size"]}-dim window to {p3b["n_comp_sma"]} principal components. Fitted on training data only; validation data is transformed using the same projection.</div>', unsafe_allow_html=True)

    from plotly.subplots import make_subplots
    individual  = p3b["individual"]
    cumulative  = p3b["cumulative"]
    n_pc        = len(individual)
    pc_labels   = [f"PC{i+1}" for i in range(n_pc)]

    fig_pca = make_subplots(rows=1, cols=2,
                            subplot_titles=["Scree — Variance per PC", "Cumulative Variance"],
                            horizontal_spacing=0.1)
    fig_pca.add_trace(go.Bar(x=pc_labels, y=[v*100 for v in individual],
                             marker_color="#22d3ee", text=[f"{v*100:.1f}%" for v in individual],
                             textposition="outside", textfont=dict(size=9, color="#94a3b8"),
                             showlegend=False), row=1, col=1)
    fig_pca.add_trace(go.Scatter(x=pc_labels, y=cumulative, mode="lines+markers",
                                 line=dict(color="#4ade80", width=2.2),
                                 marker=dict(size=7, color="#4ade80"),
                                 fill="tozeroy", fillcolor="rgba(74,222,128,0.06)",
                                 showlegend=False), row=1, col=2)
    fig_pca.add_hline(y=95, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                      annotation_text="95%", annotation_font_color="#f59e0b", row=1, col=2)
    fig_pca.update_layout(**PL, height=340,
                          title=dict(text=f"SMA Windows — {p3b['window_size']} dims → {n_pc} PCs retain {p3b['var_total_sma']:.1f}% variance",
                                     font=dict(size=12, color="#94a3b8")))
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown('<div class="section-head">HOW PCA WORKS HERE</div>', unsafe_allow_html=True)
    for step, desc in [
        ("Normalise",   "Each window is z-scored via TimeSeriesScalerMeanVariance — removes price-level differences between tickers"),
        ("Flatten",     f"3D tensor (N, {p3b['window_size']}, 1) → 2D matrix (N, {p3b['window_size']})"),
        ("Fit PCA",     f"Eigen-decompose the covariance matrix. Keep first {p3b['n_comp_sma']} PCs that explain ≥ 95% of variance"),
        ("Silhouette",  "Phase 4 computes silhouette in this reduced space — inter-cluster distances are cleaner without noise dims"),
        ("2D vis",      "A separate 2-component PCA projects all windows onto a plane for the Phase 5 scatter plot"),
    ]:
        st.markdown(f'<div style="display:flex;gap:1rem;margin-bottom:0.5rem;"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;color:#f59e0b;min-width:100px">→ {step}</span><span style="font-size:0.78rem;color:#94a3b8">{desc}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 4 — HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
elif ap == 4:
    p4 = R["p4"]
    st.markdown('<div class="phase-card">', unsafe_allow_html=True)
    st.markdown('<div class="phase-card-title">PHASE 4 — HYPERPARAMETER TUNING (on training set)</div>', unsafe_allow_html=True)

    if p4["mode"] == "manual":
        st.markdown(
            f'<div class="verdict-box warn" style="margin-bottom:1rem;">'
            f'⚙ Manual mode — elbow/silhouette search was skipped.<br>'
            f'<span style="font-size:0.78rem">K was set directly to '
            f'<strong>{p4["optimal_k"]}</strong> before running DTW.</span>'
            f'</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="stat-tile"><div class="stat-label">K MODE</div><div class="stat-value" style="color:#f59e0b;font-size:1.2rem">MANUAL</div><div class="stat-sub">User-specified</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-tile"><div class="stat-label">FINAL K</div><div class="stat-value" style="color:#f59e0b">{p4["optimal_k"]}</div><div class="stat-sub">Passed directly to DTW</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="stat-tile"><div class="stat-label">SEARCH RANGE</div><div class="stat-value" style="color:#6b7280;font-size:1.2rem">—</div><div class="stat-sub">Skipped</div></div>', unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="stat-tile"><div class="stat-label">ELBOW K</div><div class="stat-value">{p4["elbow_k"]}</div><div class="stat-sub">Kneedle inflection point</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-tile"><div class="stat-label">SILHOUETTE K</div><div class="stat-value">{p4["sil_k"]}</div><div class="stat-sub">Peak cluster tightness</div></div>', unsafe_allow_html=True)
        with c3:
            agree_txt = "agreement" if p4["agree"] else "averaged"
            st.markdown(f'<div class="stat-tile"><div class="stat-label">SELECTED K</div><div class="stat-value" style="color:#4ade80">{p4["optimal_k"]}</div><div class="stat-sub">{agree_txt}</div></div>', unsafe_allow_html=True)

        if p4["agree"]:
            st.markdown(f'<div class="verdict-box">✓ Both methods independently agree → K = {p4["optimal_k"]} is statistically robust &nbsp;·&nbsp; <span style="color:#22d3ee;font-size:0.75rem">Silhouette computed in PCA space (Phase 3b) for cleaner signal</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-box warn">⚖ Methods diverge (Elbow→{p4["elbow_k"]}, Silhouette→{p4["sil_k"]}) — averaged to K={p4["optimal_k"]} &nbsp;·&nbsp; <span style="font-size:0.75rem">Silhouette uses PCA space (Phase 3b)</span></div>', unsafe_allow_html=True)

        ks      = p4["k_range"]
        norm_in = np.array(p4["inertias"]) / max(p4["inertias"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=norm_in, name="Inertia (norm.)", mode="lines+markers",
                                 line=dict(color="#4ade80",width=2), marker=dict(size=6,color="#4ade80")))
        fig.add_trace(go.Scatter(x=ks, y=p4["silhouettes"], name="Silhouette", mode="lines+markers", yaxis="y2",
                                 line=dict(color="#22d3ee",width=2,dash="dash"), marker=dict(size=6,color="#22d3ee",symbol="square")))
        fig.add_vline(x=p4["elbow_k"],   line_color="#4ade80", line_dash="dot", opacity=0.6,
                      annotation_text=f"Elbow K={p4['elbow_k']}", annotation_font_color="#4ade80", annotation_font_size=10)
        fig.add_vline(x=p4["sil_k"],     line_color="#22d3ee", line_dash="dot", opacity=0.6,
                      annotation_text=f"Sil K={p4['sil_k']}", annotation_font_color="#22d3ee", annotation_font_size=10)
        fig.add_vline(x=p4["optimal_k"], line_color="#f59e0b", line_width=2,
                      annotation_text=f"Selected K={p4['optimal_k']}", annotation_font_color="#f59e0b", annotation_font_size=11)
        fig.update_layout(**PL)
        fig.update_layout(
            title="Elbow + Silhouette Dual Search (on training data)",
            xaxis=dict(title="K", tickvals=p4["k_range"]),
            yaxis=dict(title=dict(text="Inertia (norm.)", font=dict(color="royalblue"))),
            yaxis2=dict(title=dict(text="Silhouette", font=dict(color="tomato")),
                        overlaying="y", side="right", gridcolor="#2a2d32"),
            height=380,
            legend=dict(orientation="h", y=1.12),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        tdf = pd.DataFrame({"K": ks, "Inertia": [round(v,2) for v in p4["inertias"]],
                            "Silhouette": [round(v,4) for v in p4["silhouettes"]]}).set_index("K")
        st.dataframe(tdf.style
            .applymap(lambda _: "background-color:#1a2d1a;color:#4ade80;font-weight:600",
                      subset=pd.IndexSlice[[p4["optimal_k"]], :])
            .format({"Inertia":"{:.2f}","Silhouette":"{:.4f}"}),
            use_container_width=True, height=220)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 5 — DTW CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
elif ap == 5:
    from plotly.subplots import make_subplots
    p5 = R["p5"]; K = p5["optimal_k"]
    st.markdown('<div class="phase-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="phase-card-title">PHASE 5 — DTW K-MEANS CLUSTERING (K={K}) — Fit on train, predict on val</div>', unsafe_allow_html=True)

    sizes   = [int(np.sum(p5["lbl_sma_train"] == c)) for c in range(K)]
    biggest = int(np.argmax(sizes))
    c1, c2  = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">PATTERNS DISCOVERED</div><div class="stat-value">{K}</div><div class="stat-sub">unique shape archetypes (SMA, train)</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-tile"><div class="stat-label">LARGEST CLUSTER</div><div class="stat-value">Pattern {biggest}</div><div class="stat-sub">{sizes[biggest]:,} train windows</div></div>', unsafe_allow_html=True)

    fig_sz = go.Figure(go.Bar(
        x=[f"P{i}" for i in range(K)], y=sizes,
        marker_color=["#4ade80" if i==biggest else "#2a3a2a" for i in range(K)],
        marker_line_color="#2a2d32", marker_line_width=1,
        text=sizes, textposition="outside", textfont=dict(size=9, color="#94a3b8"),
    ))
    fig_sz.update_layout(**PL, height=190, showlegend=False)
    fig_sz.update_yaxes(title="Window count (train)", gridcolor="#2a2d32")
    fig_sz.update_xaxes(title="Pattern", gridcolor="#2a2d32")
    st.plotly_chart(fig_sz, use_container_width=True)

    # Centroid grid
    n_cols_c = min(4, K)
    n_rows_c = int(np.ceil(K / n_cols_c))
    subtitles = [f"Pattern {i}  (n={sizes[i]:,})" for i in range(K)]
    fig_c = make_subplots(rows=n_rows_c, cols=n_cols_c, subplot_titles=subtitles,
                          vertical_spacing=0.14, horizontal_spacing=0.05)
    for i in range(K):
        r = i // n_cols_c + 1
        c = i % n_cols_c + 1
        centroid = p5["centers_sma"][i].ravel()
        color = "#4ade80" if i == biggest else "#22d3ee"
        fig_c.add_trace(go.Scatter(y=centroid, mode="lines", line=dict(color=color, width=2), showlegend=False), row=r, col=c)
        fig_c.update_xaxes(showticklabels=False, gridcolor="#2a2d32", row=r, col=c)
        fig_c.update_yaxes(showticklabels=False, gridcolor="#2a2d32", zeroline=True,
                           zerolinecolor="#3a3a3a", zerolinewidth=1, row=r, col=c)
    fig_c.update_layout(**PL, height=max(200, 170*n_rows_c),
                        title=dict(text="SMA DTW Centroids — Discovered Pattern Shapes (from training)", font=dict(size=12,color="#94a3b8")))
    for ann in fig_c.layout.annotations:
        ann.font = dict(size=9, color="#94a3b8", family="IBM Plex Mono")
    st.plotly_chart(fig_c, use_container_width=True)

    # ── PCA Cluster Scatter (with train/val toggle) ─────────────────────────
    st.markdown('<div class="section-head">PCA CLUSTER SCATTER — DTW ASSIGNMENTS IN 2D PCA SPACE</div>', unsafe_allow_html=True)
    p3b = R["p3b"]
    vis_var = p3b["vis_var"]
    st.markdown(f'<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.75rem;">Each point = one {p3b["window_size"]}-day window projected onto 2 principal components (PC1 {vis_var[0]:.1f}% + PC2 {vis_var[1]:.1f}% = {sum(vis_var):.1f}% of SMA variance). Colour = DTW cluster assignment. ★ = cluster centroid projected into PCA space.</div>', unsafe_allow_html=True)

    dataset_choice = st.selectbox("Select dataset", ["Training (2006-2019)", "Validation (2020-2026)"], key="p5_dataset")
    if dataset_choice.startswith("Training"):
        vis_sma = p3b["X_sma_2d_vis_train"]
        vis_ema = p3b["X_ema_2d_vis_train"]
        lbl_sma = p5["lbl_sma_train"]
        lbl_ema = p5["lbl_ema_train"]
        label_suffix = "train"
    else:
        vis_sma = p3b["X_sma_2d_vis_val"]
        vis_ema = p3b["X_ema_2d_vis_val"]
        lbl_sma = p5["lbl_sma_val"]
        lbl_ema = p5["lbl_ema_val"]
        label_suffix = "val"

    tab_sma, tab_ema = st.tabs(["SMA Baseline", "EMA Freshness"])

    _COLORS = ["#4ade80","#22d3ee","#f59e0b","#f87171","#a78bfa",
               "#fb923c","#34d399","#60a5fa","#f472b6","#94a3b8",
               "#fbbf24","#6ee7b7","#818cf8","#fca5a5","#c4b5fd"]

    def _pca_scatter(vis_coords, labels, centers_2d, K, title):
        N_PLOT = min(8000, len(vis_coords))
        rng    = np.random.default_rng(42)
        idx    = rng.choice(len(vis_coords), N_PLOT, replace=False)
        xs, ys = vis_coords[idx, 0], vis_coords[idx, 1]
        cs     = labels[idx]

        fig_sc = go.Figure()
        for c_id in range(K):
            mask = cs == c_id
            color = _COLORS[c_id % len(_COLORS)]
            fig_sc.add_trace(go.Scatter(
                x=xs[mask], y=ys[mask], mode="markers", name=f"P{c_id}",
                marker=dict(size=4, color=color, opacity=0.4, line=dict(width=0)),
                hovertemplate=f"Pattern {c_id}<br>PC1=%{{x:.3f}}<br>PC2=%{{y:.3f}}<extra></extra>",
            ))

        # Centroids as stars
        for c_id in range(K):
            color = _COLORS[c_id % len(_COLORS)]
            fig_sc.add_trace(go.Scatter(
                x=[centers_2d[c_id, 0]], y=[centers_2d[c_id, 1]],
                mode="markers+text", name=f"P{c_id} ★",
                marker=dict(size=16, color=color, symbol="star",
                            line=dict(color="#111214", width=1.2)),
                text=[f"P{c_id}"], textposition="top center",
                textfont=dict(size=9, color=color, family="IBM Plex Mono"),
                showlegend=False,
                hovertemplate=f"Centroid P{c_id}<br>PC1=%{{x:.3f}}<br>PC2=%{{y:.3f}}<extra></extra>",
            ))

        fig_sc.update_layout(**PL)
        fig_sc.update_layout(
            height=420,
            title=dict(text=f"{title} — {dataset_choice}  ({N_PLOT:,} sampled)", font=dict(size=11, color="#94a3b8")),
            xaxis=dict(title=f"PC1 ({vis_var[0]:.1f}%)", gridcolor="#2a2d32", zeroline=False),
            yaxis=dict(title=f"PC2 ({vis_var[1]:.1f}%)", gridcolor="#2a2d32", zeroline=False),
            legend=dict(orientation="h", y=-0.15, font=dict(size=9), itemsizing="constant"),
            hovermode="closest",
        )
        return fig_sc

    with tab_sma:
        st.plotly_chart(_pca_scatter(np.array(vis_sma), lbl_sma,
                                     np.array(p5["centers_2d"]), K, "SMA Baseline"), use_container_width=True)
    with tab_ema:
        st.plotly_chart(_pca_scatter(np.array(vis_ema), lbl_ema,
                                     np.array(p5["centers_2d"]), K, "EMA Freshness"), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 6 — ROBUSTNESS (on training data)
# ─────────────────────────────────────────────────────────────────────────────
elif ap == 6:
    p6    = R["p6"]
    res   = p6["results_df"]
    years = p6["years"]
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    cmap    = {c: palette[c % len(palette)] for c in p6["clusters"]}

    st.markdown('<div class="phase-card">', unsafe_allow_html=True)
    st.markdown('<div class="phase-card-title">PHASE 6 — EMPIRICAL ROBUSTNESS PROFILING (training set, 2006–2019)</div>', unsafe_allow_html=True)

    # Yearly dominance bar
    st.markdown('<div class="section-head">YEARLY PATTERN DOMINANCE (SMA, train)</div>', unsafe_allow_html=True)
    yearly_rows = []
    for y in years:
        best_c, best_wr = -1, -1.0
        for c in p6["clusters"]:
            sub = res[(res["Cluster_SMA"]==c) & (res["Year"]==y)]
            if len(sub) == 0: continue
            wr = sub["Target"].mean() * 100
            if wr > best_wr: best_c, best_wr = c, wr
        if best_c >= 0:
            yearly_rows.append({"Year":y, "Dominant":f"Pattern {best_c}", "Win Rate":round(best_wr,1), "_cid":best_c})
    yd = pd.DataFrame(yearly_rows)

    fig_y = go.Figure()
    for pid in sorted(yd["_cid"].unique()):
        sub = yd[yd["_cid"]==pid]
        fig_y.add_trace(go.Bar(
            x=sub["Year"], y=sub["Win Rate"], name=f"Pattern {pid}",
            marker_color=cmap[pid],
            text=sub["Win Rate"].astype(str)+"%", textposition="outside", textfont=dict(size=9),
        ))
    fig_y.add_hline(y=50, line_dash="dot", line_color="#6b7280", opacity=0.7,
                    annotation_text="50% baseline", annotation_font_size=9, annotation_font_color="#6b7280")
    fig_y.update_layout(**PL, height=300, barmode="stack",
                    legend=dict(orientation="h", y=1.12, font=dict(size=9)),
                    hovermode="x unified")
    fig_y.update_xaxes(title="Year", dtick=1)
    fig_y.update_yaxes(title="Win Rate (%)", range=[40,75])
    st.plotly_chart(fig_y, use_container_width=True)

    yr_tbl = yd[["Year","Dominant","Win Rate"]].set_index("Year")
    st.dataframe(yr_tbl.style.background_gradient(subset=["Win Rate"], cmap="RdYlGn", vmin=45, vmax=68)
                              .format({"Win Rate":"{:.1f}%"}), use_container_width=True, height=220)

    # Aggregate winner
    st.markdown('<div class="section-head">AGGREGATE WINNER — SELECT RANGE</div>', unsafe_allow_html=True)
    min_y, max_y = int(min(years)), int(max(years))
    yr_range = st.slider("Year range", min_y, max_y, (min_y, max_y), key="p6_yr")
    sub_res  = res[(res["Year"]>=yr_range[0]) & (res["Year"]<=yr_range[1])]
    agg = []
    for c in p6["clusters"]:
        m = sub_res["Cluster_SMA"] == c; n = int(m.sum())
        if n > 0: agg.append({"Pattern":f"Pattern {c}","Win Rate":round(sub_res[m]["Target"].mean()*100,2),"Windows":n,"_cid":c})
    agg_df = pd.DataFrame(agg).sort_values("Win Rate", ascending=False).reset_index(drop=True)

    if not agg_df.empty:
        w = agg_df.iloc[0]
        st.markdown(f"""
        <div class="winner-card">
          <div class="winner-label">AGGREGATE WINNER (train) · {yr_range[0]}–{yr_range[1]}</div>
          <div class="winner-value">{w['Pattern']}</div>
          <div class="winner-sub">{w['Win Rate']}% win rate · {w['Windows']:,} windows</div>
        </div>
        """, unsafe_allow_html=True)
        fig_agg = go.Figure(go.Bar(
            x=agg_df["Pattern"], y=agg_df["Win Rate"],
            marker_color=[cmap[r["_cid"]] for _, r in agg_df.iterrows()],
            text=agg_df["Win Rate"].round(1).astype(str)+"%",
            textposition="outside", textfont=dict(size=10),
        ))
        fig_agg.add_hline(y=50, line_dash="dot", line_color="#6b7280", opacity=0.7)
        fig_agg.update_layout(**PL, height=280, showlegend=False)
        fig_agg.update_yaxes(range=[40,72], title="Win Rate (%)")
        st.plotly_chart(fig_agg, use_container_width=True)

    # Full heatmap
    st.markdown('<div class="section-head">WIN RATE HEATMAP — ALL CLUSTERS × YEARS (SMA, train)</div>', unsafe_allow_html=True)
    mat = p6["mat_sma"]
    year_cols  = [str(y) for y in years]
    heat_data  = mat[year_cols].values
    K_clusters = len(mat)
    fig_h = go.Figure(go.Heatmap(
        z=heat_data, x=year_cols, y=[f"P{c}" for c in mat.index],
        colorscale=[[0,"#f87171"],[0.5,"#1a1c1f"],[1,"#4ade80"]], zmid=50,
        text=np.where(np.isnan(heat_data), "\u2014", heat_data.astype(str)),
        texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title=dict(text="Win %", font=dict(color="#94a3b8", size=9)),
                      tickfont=dict(color="#94a3b8", size=9), len=0.8),
    ))
    fig_h.update_layout(**PL, height=max(220, 30*K_clusters+60))
    fig_h.update_xaxes(tickfont=dict(size=9))
    fig_h.update_yaxes(tickfont=dict(size=9))
    st.plotly_chart(fig_h, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # ── Validation robustness heatmap (from stored pipeline results) ─────────
    st.markdown('<div class="section-head">VALIDATION WIN RATE HEATMAP (SMA, 2020–2026) — out-of-sample</div>', unsafe_allow_html=True)
    mat_sma_val    = p6["mat_sma_val"]
    years_val_disp = p6["years_val"]
    year_cols_val  = [str(y) for y in years_val_disp]
    heat_data_val  = mat_sma_val[year_cols_val].values

    fig_h_val = go.Figure(go.Heatmap(
        z=heat_data_val, x=year_cols_val,
        y=[f"P{c}" for c in mat_sma_val.index],
        colorscale=[[0,"#f87171"],[0.5,"#1a1c1f"],[1,"#4ade80"]], zmid=50,
        text=np.where(np.isnan(heat_data_val), "—", heat_data_val.astype(str)),
        texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title=dict(text="Win %", font=dict(color="#94a3b8", size=9)),
                      tickfont=dict(color="#94a3b8", size=9), len=0.8),
    ))
    fig_h_val.update_layout(**PL, height=max(220, 30*len(mat_sma_val)+60))
    fig_h_val.update_xaxes(tickfont=dict(size=9))
    fig_h_val.update_yaxes(tickfont=dict(size=9))
    st.plotly_chart(fig_h_val, use_container_width=True)

    # ── Overall% train vs val comparison (pattern stability check) ──────────
    st.markdown('<div class="section-head">PATTERN STABILITY — TRAIN vs VALIDATION (OVERALL WIN %)</div>', unsafe_allow_html=True)
    train_overall = p6["mat_sma"]["Overall%"].to_dict()
    val_overall   = mat_sma_val["Overall%"].to_dict()
    all_patterns  = sorted(set(train_overall) | set(val_overall))
    stab_fig = go.Figure()
    stab_fig.add_trace(go.Bar(
        x=[f"P{c}" for c in all_patterns],
        y=[train_overall.get(c, 0) for c in all_patterns],
        name="Train (2006–2019)", marker_color="#4ade80", marker_line_width=0,
    ))
    stab_fig.add_trace(go.Bar(
        x=[f"P{c}" for c in all_patterns],
        y=[val_overall.get(c, 0) for c in all_patterns],
        name="Validation (2020–2026)", marker_color="#22d3ee", marker_line_width=0,
    ))
    stab_fig.add_hline(y=50, line_dash="dot", line_color="#6b7280", opacity=0.6)
    stab_fig.add_hline(y=st.session_state.win_thresh, line_dash="dot", line_color="#f59e0b",
                       annotation_text=f"Threshold {st.session_state.win_thresh:.1f}%",
                       annotation_font_color="#f59e0b", annotation_font_size=9, opacity=0.7)
    stab_fig.update_layout(**PL, barmode="group", height=260, showlegend=True,
                           legend=dict(orientation="h", y=1.12, font=dict(size=9)))
    stab_fig.update_yaxes(range=[40, 72], title="Win Rate (%)")
    st.plotly_chart(stab_fig, use_container_width=True)

    
# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 7 — TA-LIB (on validation windows with training-winning clusters)
# ─────────────────────────────────────────────────────────────────────────────
elif ap == 7:
    p7 = R["p7"]
    st.markdown('<div class="phase-card">', unsafe_allow_html=True)
    st.markdown('<div class="phase-card-title">PHASE 7 — TA-LIB CANDLESTICK VALIDATION (validation set)</div>', unsafe_allow_html=True)

    if p7["skipped"] is True:
        st.markdown('<div class="verdict-box warn">⚠ TA-Lib scan skipped (checkbox enabled in Advanced settings)</div>', unsafe_allow_html=True)
    elif p7["skipped"] == "no_talib":
        st.markdown('<div class="verdict-box warn">⚠ TA-Lib not installed — run <code>pip install ta-lib</code> and rerun the pipeline</div>', unsafe_allow_html=True)
    else:
        thresh = st.session_state.win_thresh

        # ── SMA / EMA toggle ──────────────────────────────────────────────
        variant_choice = st.radio(
            "Heatmap variant", ["SMA Baseline", "EMA Freshness"],
            horizontal=True, key="p7_variant",
            label_visibility="collapsed",
        )
        is_sma    = variant_choice == "SMA Baseline"
        win_ids   = p7.get("win_ids_sma" if is_sma else "win_ids_ema", [])
        mdf       = p7["matches_sma" if is_sma else "matches_ema"]
        summary   = p7["summary_sma"  if is_sma else "summary_ema"]
        v_color   = "#4ade80" if is_sma else "#22d3ee"
        mid_color = "rgba(74,222,128,0.5)" if is_sma else "rgba(34,211,238,0.5)"

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat-tile"><div class="stat-label">WIN THRESHOLD</div><div class="stat-value">{thresh:.1f}%</div><div class="stat-sub">clusters above this are validated</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-tile"><div class="stat-label">WINNING CLUSTERS ({variant_choice.split()[0]}, train)</div><div class="stat-value" style="color:{v_color}">{len(win_ids)}</div><div class="stat-sub">IDs: {win_ids}</div></div>', unsafe_allow_html=True)

        if summary is not None and mdf is not None:
            st.markdown(f'<div class="verdict-box info">ℹ {len(mdf):,} candlestick pattern hits on validation set · {variant_choice}</div>', unsafe_allow_html=True)

            fig_hm = go.Figure(go.Heatmap(
                z=summary.values, x=list(summary.columns),
                y=[f"Cluster {c}" for c in summary.index],
                colorscale=[[0,"#1a1c1f"], [0.5, mid_color], [1, v_color]],
                text=summary.values, texttemplate="%{text}", textfont=dict(size=10, color="#fff"),
                colorbar=dict(title=dict(text="Hits", font=dict(color="#94a3b8", size=9)),
                              tickfont=dict(color="#94a3b8", size=9)),
            ))
            fig_hm.update_layout(**PL, height=max(220, 40*len(win_ids)+80),
                                 title=dict(text=f"Pattern Hits: Cluster × Candlestick — {variant_choice} (validation)",
                                            font=dict(size=12, color="#94a3b8")))
            st.plotly_chart(fig_hm, use_container_width=True)

            st.markdown('<div class="section-head">SIGNAL POLARITY</div>', unsafe_allow_html=True)
            bb = mdf.groupby(["Pattern","Signal"]).size().reset_index(name="Count")
            bb["Direction"] = bb["Signal"].map({100:"Bullish (+100)",-100:"Bearish (−100)"})
            fig_bb = px.bar(bb, x="Pattern", y="Count", color="Direction",
                            color_discrete_map={"Bullish (+100)":v_color,"Bearish (−100)":"#f87171"},
                            barmode="group")
            fig_bb.update_layout(**PL, height=260,
                                 legend=dict(orientation="h", y=1.12, font=dict(size=10)))
            st.plotly_chart(fig_bb, use_container_width=True)

            # ── Side-by-side SMA vs EMA hit count comparison ──────────────
            st.markdown('<div class="section-head">SMA vs EMA — TOTAL HIT COMPARISON</div>', unsafe_allow_html=True)
            sma_total = len(p7["matches_sma"]) if p7["matches_sma"] is not None else 0
            ema_total = len(p7["matches_ema"]) if p7["matches_ema"] is not None else 0
            cmp_fig = go.Figure(go.Bar(
                x=["SMA Baseline", "EMA Freshness"],
                y=[sma_total, ema_total],
                marker_color=["#4ade80", "#22d3ee"],
                text=[sma_total, ema_total], textposition="outside",
                textfont=dict(size=13, color="#e2e8f0"),
            ))
            cmp_fig.update_layout(**PL, height=220, showlegend=False)
            cmp_fig.update_yaxes(title="Total Pattern Hits")
            st.plotly_chart(cmp_fig, use_container_width=True)
        else:
            st.markdown(f'<div class="verdict-box warn">No textbook candlestick patterns detected in {variant_choice} validation windows of winning clusters — DTW may have discovered non-standard shapes.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 8 — EVALUATION (on validation set)
# ─────────────────────────────────────────────────────────────────────────────
elif ap == 8:
    p8 = R["p8"]
    sma_m = p8["sma"]; ema_m = p8["ema"]
    sma_tr = p8["sma_train"]; ema_tr = p8["ema_train"]
    overfit_sma = p8["overfit_sma"]; overfit_ema = p8["overfit_ema"]

    st.markdown('<div class="phase-card">', unsafe_allow_html=True)
    st.markdown('<div class="phase-card-title">PHASE 8 — FINAL EVALUATION: SMA vs EMA · TRAIN vs VALIDATION</div>', unsafe_allow_html=True)

    winner_color = "#4ade80" if p8["winner"] == "SMA Baseline" else "#22d3ee"
    st.markdown(f"""
    <div class="winner-card">
      <div class="winner-label">SUPERIOR VARIANT ON VALIDATION (F1 SCORE)</div>
      <div class="winner-value" style="color:{winner_color}">{p8['winner']}</div>
      <div class="winner-sub">Validation F1: {sma_m['f1']:.4f} (SMA) &nbsp; vs &nbsp; {ema_m['f1']:.4f} (EMA)</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-metric comparison cards (SMA train/val, EMA train/val) ──────────
    st.markdown('<div class="section-head">METRICS — TRAIN (2006–2019) vs VALIDATION (2020–2026)</div>', unsafe_allow_html=True)
    metrics_order = ["accuracy","precision","recall","f1","coverage"]
    metric_labels = {"accuracy":"Accuracy","precision":"Precision","recall":"Recall","f1":"F1 Score","coverage":"Coverage"}
    m_cols = st.columns(5)
    for col, m in zip(m_cols, metrics_order):
        sv_tr = sma_tr[m]; sv_v = sma_m[m]
        ev_tr = ema_tr[m]; ev_v = ema_m[m]
        # Best val variant highlighted green
        sma_col = "#4ade80" if sv_v >= ev_v else "#94a3b8"
        ema_col = "#22d3ee" if ev_v >  sv_v else "#94a3b8"
        col.markdown(f"""
        <div class="stat-tile">
          <div class="stat-label">{metric_labels[m]}</div>
          <div style="margin-top:0.5rem;border-bottom:1px solid #2a2d32;padding-bottom:0.4rem;margin-bottom:0.4rem">
            <div style="font-size:0.62rem;color:#6b7280;font-family:'IBM Plex Mono',monospace;margin-bottom:2px">SMA</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#6b7280">tr {sv_tr:.4f}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:{sma_col};font-weight:600">val {sv_v:.4f}</div>
          </div>
          <div>
            <div style="font-size:0.62rem;color:#6b7280;font-family:'IBM Plex Mono',monospace;margin-bottom:2px">EMA</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#6b7280">tr {ev_tr:.4f}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:{ema_col};font-weight:600">val {ev_v:.4f}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Overfitting indicators ───────────────────────────────────────────────
    st.markdown('<div class="section-head">OVERFITTING CHECK — TRAIN F1 minus VAL F1 (lower = better generalisation)</div>', unsafe_allow_html=True)
    def _overfit_color(delta):
        if delta < 0.02: return "#4ade80"   # negligible
        if delta < 0.06: return "#f59e0b"   # moderate
        return "#f87171"                     # high — concern

    oc1, oc2 = st.columns(2)
    with oc1:
        clr = _overfit_color(overfit_sma)
        verdict = "✓ Generalises well" if overfit_sma < 0.02 else ("⚠ Moderate overfit" if overfit_sma < 0.06 else "✗ Significant overfit")
        st.markdown(f'<div class="stat-tile"><div class="stat-label">SMA OVERFIT DELTA (F1)</div>'
                    f'<div class="stat-value" style="color:{clr};font-size:1.4rem">+{overfit_sma:.4f}</div>'
                    f'<div class="stat-sub">{verdict}</div></div>', unsafe_allow_html=True)
    with oc2:
        clr = _overfit_color(overfit_ema)
        verdict = "✓ Generalises well" if overfit_ema < 0.02 else ("⚠ Moderate overfit" if overfit_ema < 0.06 else "✗ Significant overfit")
        st.markdown(f'<div class="stat-tile"><div class="stat-label">EMA OVERFIT DELTA (F1)</div>'
                    f'<div class="stat-value" style="color:{clr};font-size:1.4rem">+{overfit_ema:.4f}</div>'
                    f'<div class="stat-sub">{verdict}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="stat-divider"></div>', unsafe_allow_html=True)

    # ── Grouped bar chart: 4 bars per metric (SMA-train, SMA-val, EMA-train, EMA-val) ──
    st.markdown('<div class="section-head">GROUPED COMPARISON CHART</div>', unsafe_allow_html=True)
    bar_metrics = ["accuracy","precision","recall","f1","coverage"]
    bar_labels  = ["Accuracy","Precision","Recall","F1","Coverage"]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="SMA Train",      x=bar_labels, y=[sma_tr[m] for m in bar_metrics],
                             marker_color="rgba(74,222,128,0.45)", marker_line_color="#4ade80", marker_line_width=1))
    fig_bar.add_trace(go.Bar(name="SMA Validation", x=bar_labels, y=[sma_m[m]  for m in bar_metrics],
                             marker_color="#4ade80", marker_line_width=0))
    fig_bar.add_trace(go.Bar(name="EMA Train",      x=bar_labels, y=[ema_tr[m] for m in bar_metrics],
                             marker_color="rgba(34,211,238,0.45)", marker_line_color="#22d3ee", marker_line_width=1))
    fig_bar.add_trace(go.Bar(name="EMA Validation", x=bar_labels, y=[ema_m[m]  for m in bar_metrics],
                             marker_color="#22d3ee", marker_line_width=0))
    # 1. Apply the global dark theme first
    fig_bar.update_layout(**PL)
    
    # 2. Add the specific bar chart layout settings
    fig_bar.update_layout(
        barmode="group", height=300,
        legend=dict(orientation="h", y=1.12, font=dict(size=9))
    )
    
    # 3. Safely set the Y-axis range without breaking the dark theme grid lines
    fig_bar.update_yaxes(range=[0, 1.15])
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Radar chart (validation only, SMA vs EMA) ───────────────────────────
    st.markdown('<div class="section-head">RADAR — VALIDATION SET SMA vs EMA</div>', unsafe_allow_html=True)
    cats = ["Accuracy","Precision","Recall","F1","Coverage"]
    sv   = [sma_m[m] for m in ["accuracy","precision","recall","f1","coverage"]]
    ev_  = [ema_m[m] for m in ["accuracy","precision","recall","f1","coverage"]]
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=sv+[sv[0]], theta=cats+[cats[0]], fill="toself", name="SMA Validation",
                                    line=dict(color="#4ade80",width=2), fillcolor="rgba(74,222,128,0.1)"))
    fig_r.add_trace(go.Scatterpolar(r=ev_+[ev_[0]], theta=cats+[cats[0]], fill="toself", name="EMA Validation",
                                    line=dict(color="#22d3ee",width=2), fillcolor="rgba(34,211,238,0.1)"))
    fig_r.update_layout(
        paper_bgcolor="#1a1c1f", height=320,
        polar=dict(bgcolor="#1a1c1f",
                   radialaxis=dict(visible=True,range=[0,1],gridcolor="#2a2d32",tickfont=dict(size=8,color="#6b7280"),tickformat=".2f"),
                   angularaxis=dict(gridcolor="#2a2d32",tickfont=dict(size=10,color="#94a3b8"))),
        legend=dict(orientation="h",x=0.15,y=-0.12,font=dict(size=10,color="#94a3b8")),
        margin=dict(l=50,r=50,t=20,b=50),
        font=dict(family="IBM Plex Mono"),
    )
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown('<div class="section-head">WINNING CLUSTER MEMBERSHIP (identified on train, applied on val)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="verdict-box">SMA winning clusters (train): {sma_m["winning_ids"]}<br>Signal coverage (val): {sma_m["coverage"]:.1%}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="verdict-box info">EMA winning clusters (train): {ema_m["winning_ids"]}<br>Signal coverage (val): {ema_m["coverage"]:.1%}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="verdict-box" style="margin-top:0.5rem">Pipeline complete ✓ — Winning variant on validation: <strong>{p8["winner"]}</strong></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)