# -*- coding: utf-8 -*-
"""
株価モンテカルロ予測 — Streamlit アプリ
=========================================
ブラウザで銘柄を入力 → ボタンを押すだけで、データ取得・モンテカルロ・グラフ表示まで自動。

【使い方】
1. 初回だけライブラリを入れる:
       pip install streamlit yfinance numpy pandas matplotlib
2. このファイルがあるフォルダで実行:
       streamlit run montecarlo_app.py
3. 自動でブラウザが開きます(開かなければ http://localhost:8501 )。
   左側で銘柄を選び「分析する」を押すだけ。

【銘柄コード(Yahoo Finance形式)の例】
   日経平均 ^N225 / トヨタ 7203.T / ソニーG 6758.T / アップル AAPL / S&P500 ^GSPC

※ グラフの軸ラベルは、フォント設定なしでも文字化けしないよう英語にしています。
   説明は日本語で表示されます。
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NAVY = "#1a2744"
PINK = "#ec4899"
SUB  = "#3d4f6e"
BAR  = "#9bb4dd"

st.set_page_config(page_title="株価モンテカルロ予測", layout="wide")
st.title("株価モンテカルロ予測")
st.caption("過去の値動きから、1年後の株価がどのあたりに着地しそうかを「確率の範囲」で示します。")

# ---------- サイドバー(設定) ----------
with st.sidebar:
    st.header("設定")
    ticker  = st.text_input("銘柄コード (Yahoo Finance形式)", "^N225")
    display = st.text_input("表示名", "日経225")
    years   = st.slider("取得する年数", 1, 10, 5)
    n_runs  = st.select_slider("シミュレーション回数",
                               options=[1000, 5000, 10000, 20000, 50000], value=10000)
    run     = st.button("分析する", type="primary")
    st.caption("例: 日経平均 ^N225 / トヨタ 7203.T / アップル AAPL / S&P500 ^GSPC")

HORIZON = 252  # 約1年(営業日)

def _to_stooq(t):
    """Yahoo形式の銘柄コードをStooq形式に変換(予備データ源用)"""
    t = t.strip()
    m = {"^N225": "^NKX", "^GSPC": "^SPX", "^IXIC": "^NDQ", "^DJI": "^DJI", "^FTSE": "^FTM"}
    if t in m:
        return m[t]
    if t.endswith(".T"):            # 東証(例 7203.T → 7203.JP)
        return t[:-2] + ".JP"
    if "." not in t and not t.startswith("^"):   # 米国株とみなす(例 AAPL → AAPL.US)
        return t + ".US"
    return t

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(ticker, years):
    # 主: Yahoo Finance (yfinance)
    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{years}y", interval="1d",
                         auto_adjust=True, progress=False)
        if df is not None and len(df) > 0:
            if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                df.columns = [c[0] for c in df.columns]
            s = df["Close"].dropna()
            if len(s) >= 60:
                return s, "Yahoo Finance"
    except Exception:
        pass
    # 予備: Stooq (クラウドでYahooが混み合った時の保険)
    try:
        import datetime as _dt
        from pandas_datareader import data as pdr
        end = _dt.date.today()
        start = _dt.date(end.year - years, end.month, end.day)
        df = pdr.DataReader(_to_stooq(ticker), "stooq", start, end)
        if df is not None and len(df) > 0:
            s = df.sort_index()["Close"].dropna()
            if len(s) >= 60:
                return s, "Stooq"
    except Exception:
        pass
    return None, None

def monte_carlo(closes, n_runs, horizon=HORIZON):
    logr = np.diff(np.log(closes))
    m   = logr.mean()
    var = logr.var(ddof=1)
    sd  = np.sqrt(var)
    cur = closes[-1]
    rng = np.random.default_rng()
    steps = rng.normal(m - 0.5 * var, sd, size=(n_runs, horizon))
    paths = cur * np.exp(np.cumsum(steps, axis=1))
    paths = np.hstack([np.full((n_runs, 1), cur), paths])
    return cur, m, var, sd, paths

def style_axes(ax):
    ax.grid(alpha=.25)
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("#d0d8e8")

# ---------- メイン ----------
if not run:
    st.info("左の設定で銘柄を選び、「分析する」を押してください。")
    st.stop()

with st.spinner("データを取得中..."):
    s, source = fetch_prices(ticker, years)

if s is None or len(s) < 60:
    st.error("データを取得できませんでした。銘柄コードをご確認のうえ、少し時間をおいて再実行してください。"
             "（クラウドでは取得元が混み合うと一時的に失敗することがあります）")
    st.stop()

closes = s.values.astype(float)
dates  = s.index
cur, m, var, sd, paths = monte_carlo(closes, n_runs)
ann_ret = (np.exp(m * 252) - 1) * 100
ann_vol = sd * np.sqrt(252) * 100
term = paths[:, -1]
p = np.percentile(term, [2.5, 25, 50, 75, 97.5])  # p[0]=2.5% ... p[4]=97.5%

# 統計サマリ
c1, c2, c3 = st.columns(3)
c1.metric("現在の水準", f"{cur:,.0f}")
c2.metric("年率リターン (平均的な伸び)", f"{ann_ret:.1f} %/年")
c3.metric("年率ボラティリティ (ブレ幅)", f"{ann_vol:.1f} %/年")
st.caption(f"対象期間: {str(dates[0])[:10]} 〜 {str(dates[-1])[:10]} / {len(closes):,}営業日 ・ データ取得元: {source}")

# 過去チャート
st.subheader(f"{display}：過去{years}年の値動き")
fig1, ax1 = plt.subplots(figsize=(11, 3.2))
ax1.plot(dates, closes, color=NAVY, lw=1.5)
style_axes(ax1)
st.pyplot(fig1)

# 予測ファンチャート
st.subheader("1年後の株価予測グラフ（モンテカルロ・シミュレーション）")
xs = np.arange(paths.shape[1])
bands = np.percentile(paths, [2.5, 25, 50, 75, 97.5], axis=0)
fig2, ax2 = plt.subplots(figsize=(11, 4.0))
ax2.fill_between(xs, bands[0], bands[4], color=PINK, alpha=.15, label="95% CI")
for i in range(min(100, n_runs)):
    ax2.plot(xs, paths[i], color=NAVY, lw=.7, alpha=.12)
ax2.plot(xs, bands[2], color=NAVY, lw=2.6, label="Median")
ax2.set_xlabel("Trading days (0 = now, 252 ≈ 1 year)")
ax2.set_xlim(0, HORIZON)
style_axes(ax2)
ax2.legend(loc="upper left")
st.pyplot(fig2)
st.caption(f"薄いピンクが95%信頼区間、細い線が1本1本のシナリオ（{min(100, n_runs)}本表示）、太い線が中央値です。先に行くほど範囲が広がります。")

# 分布(ヒストグラム)
st.subheader("1年後の分布")
fig3, ax3 = plt.subplots(figsize=(11, 3.4))
ax3.hist(term, bins=60, range=(p[0] * 0.9, p[4] * 1.05), color=BAR)
labels = [(p[0], "2.5%", SUB, 1.2, "--"), (p[1], "25%", SUB, 1.2, "--"),
          (p[2], "Median", PINK, 2.6, "-"), (p[3], "75%", SUB, 1.2, "--"),
          (p[4], "97.5%", SUB, 1.2, "--")]
for val, lab, col, lw, ls in labels:
    ax3.axvline(val, color=col, lw=lw, ls=ls)
    ax3.text(val, ax3.get_ylim()[1] * 0.96, lab, color=col, ha="center", va="top", fontsize=9)
ax3.set_xlabel("Price after 1 year")
style_axes(ax3)
st.pyplot(fig3)
st.caption("中央の赤い線が中央値。両端(2.5%と97.5%)の間が95%信頼区間です。")

# まとめ
st.subheader("まとめ：1年後の予測")
r1, r2 = st.columns(2)
r1.metric("1年後の中央値（最も起こりやすい着地点）",
          f"{p[2]:,.0f}", f"{(p[2] / cur - 1) * 100:+.1f} % vs 現在")
r2.metric("95%信頼区間（20回に19回はこの範囲）",
          f"{p[0]:,.0f} 〜 {p[4]:,.0f}")
st.info("「過去の値動きのクセがこれからも続く」と仮定した計算です。"
        "決算・事件・暴落など過去になかった出来事は反映されにくく、投資助言ではありません。"
        "ご自身の判断と他の情報も合わせてご活用ください。")
