# -*- coding: utf-8 -*-
"""
株価モンテカルロ予測アプリ(Streamlit版・データ取得2系統対応)
================================================================
銘柄コードを入力すると、過去5年分の株価を自動取得し、
モンテカルロシミュレーションで1年後の株価分布を予測します。

データ取得は2段構え:
  1. Yahoo Finance (yfinance)  … まずこちらを試す
  2. Stooq (stooq.com)         … Yahooがブロックされた場合の保険
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# ----------------------------------------------------------
# 設定値
# ----------------------------------------------------------
N_SIMULATIONS = 10_000   # シミュレーション回数(1万通り)
N_DAYS = 252             # 予測日数(1年 ≒ 252営業日)
HISTORY_YEARS = 5        # 取得する過去データの年数

st.set_page_config(page_title="株価モンテカルロ予測", layout="wide")
st.title("📈 株価モンテカルロ予測(1年後)")
st.caption("過去5年の値動きパターンをもとに、1年後の株価を1万通りシミュレーションします")

# ----------------------------------------------------------
# 入力部分(サイドバー)
# ----------------------------------------------------------
with st.sidebar:
    st.header("銘柄の指定")
    ticker = st.text_input(
        "銘柄コード(ティッカー)",
        value="7203.T",
        help="日本株は「4桁コード.T」(例: トヨタ=7203.T)、米国株はそのまま(例: AAPL)、日経平均は ^N225",
    )
    st.markdown(
        """
        **入力例**
        - `7203.T` … トヨタ自動車
        - `9984.T` … ソフトバンクG
        - `^N225` … 日経平均
        - `AAPL` … アップル
        - `^IXIC` … NASDAQ総合
        """
    )
    run = st.button("分析を実行する", type="primary", use_container_width=True)


def currency_label(ticker_code: str) -> str:
    """ティッカーから通貨表示を判定(東証・日経系は円、それ以外はドル)"""
    t = ticker_code.upper()
    if t.endswith(".T") or t in ("^N225",):
        return "円"
    return "ドル"


def to_stooq_symbol(ticker_code: str) -> str:
    """Yahoo形式のティッカーをStooq形式に変換する"""
    t = ticker_code.upper().strip()
    index_map = {
        "^N225": "^NKX",   # 日経平均
        "^IXIC": "^NDQ",   # NASDAQ総合
        "^GSPC": "^SPX",   # S&P500
        "^DJI": "^DJI",    # NYダウ
    }
    if t in index_map:
        return index_map[t]
    if t.endswith(".T"):              # 東証銘柄: 7203.T → 7203.JP
        return t.replace(".T", ".JP")
    if "." not in t and not t.startswith("^"):
        return t + ".US"              # 米国株: AAPL → AAPL.US
    return t


def fetch_from_yahoo(ticker_code: str) -> pd.Series:
    """Yahoo Financeから取得(クラウド環境ではブロックされることがある)"""
    data = yf.download(
        ticker_code, period=f"{HISTORY_YEARS}y",
        auto_adjust=True, progress=False,
    )
    if data is None or data.empty:
        return pd.Series(dtype=float)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def fetch_from_stooq(ticker_code: str) -> pd.Series:
    """Stooq(無料の株価データサービス)からCSVを直接取得"""
    symbol = to_stooq_symbol(ticker_code)
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    df = pd.read_csv(url)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    cutoff = df.index.max() - pd.DateOffset(years=HISTORY_YEARS)
    return df.loc[df.index >= cutoff, "Close"].dropna()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(ticker_code: str):
    """Yahoo→Stooqの順に試し、(株価データ, データ源の名前) を返す"""
    try:
        prices = fetch_from_yahoo(ticker_code)
        if len(prices) > 0:
            return prices, "Yahoo Finance"
    except Exception:
        pass
    try:
        prices = fetch_from_stooq(ticker_code)
        if len(prices) > 0:
            return prices, "Stooq"
    except Exception:
        pass
    return pd.Series(dtype=float), None


def run_monte_carlo(prices: pd.Series):
    """幾何ブラウン運動(GBM)によるモンテカルロシミュレーション"""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean()      # 1日あたりの平均リターン
    sigma = log_returns.std()    # 1日あたりのばらつき(リスク)

    last_price = float(prices.iloc[-1])
    rng = np.random.default_rng(seed=42)

    daily = rng.normal(mu, sigma, size=(N_DAYS, N_SIMULATIONS))
    paths = last_price * np.exp(np.cumsum(daily, axis=0))
    paths = np.vstack([np.full(N_SIMULATIONS, last_price), paths])
    return paths, last_price, mu, sigma


# ----------------------------------------------------------
# メイン処理
# ----------------------------------------------------------
if run:
    with st.spinner(f"{ticker} の株価データを取得中…"):
        prices, source = fetch_prices(ticker.strip())

    if prices.empty:
        st.error(
            "データを取得できませんでした。銘柄コードを確認してください(例: 7203.T)。"
            "コードが正しい場合は、データ提供元が混雑している可能性があるので、"
            "数分おいて再実行してみてください。"
        )
        st.stop()

    st.caption(f"データ取得元: {source}")

    if len(prices) < 250:
        st.warning("取得できたデータが1年分未満です。予測の信頼性が下がる点にご注意ください。")

    cur = currency_label(ticker)
    paths, last_price, mu, sigma = run_monte_carlo(prices)
    final_prices = paths[-1]

    p10 = np.percentile(final_prices, 10)
    p50 = np.percentile(final_prices, 50)
    p90 = np.percentile(final_prices, 90)
    prob_up = (final_prices > last_price).mean() * 100
    annual_vol = sigma * np.sqrt(252) * 100

    st.subheader(f"分析結果:{ticker}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("現在値", f"{last_price:,.0f} {cur}")
    c2.metric("悲観シナリオ(下位10%)", f"{p10:,.0f} {cur}", f"{(p10/last_price-1)*100:+.1f}%")
    c3.metric("中央値", f"{p50:,.0f} {cur}", f"{(p50/last_price-1)*100:+.1f}%")
    c4.metric("楽観シナリオ(上位10%)", f"{p90:,.0f} {cur}", f"{(p90/last_price-1)*100:+.1f}%")
    c5.metric("1年後に上昇している確率", f"{prob_up:.1f}%")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### 過去5年の実績と、シミュレーションの例(100本)")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=prices.index, y=prices.values,
            name="過去の実績", line=dict(color="#1a2744", width=2),
        ))
        future_dates = pd.bdate_range(start=prices.index[-1], periods=N_DAYS + 1)
        for i in range(100):
            fig1.add_trace(go.Scatter(
                x=future_dates, y=paths[:, i],
                line=dict(color="rgba(37,99,235,0.08)", width=1),
                showlegend=False, hoverinfo="skip",
            ))
        fig1.add_trace(go.Scatter(
            x=future_dates, y=np.percentile(paths, 50, axis=1),
            name="予測の中央値", line=dict(color="#2563eb", width=2, dash="dash"),
        ))
        fig1.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis_title=f"株価({cur})", legend=dict(orientation="h"))
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        st.markdown("##### 1年後の株価はどこに落ち着きそうか(1万通りの分布)")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=final_prices, nbinsx=80, marker_color="#2563eb", opacity=0.75,
            name="1年後の株価",
        ))
        for val, label, color in [
            (last_price, "現在値", "#1a2744"),
            (p10, "悲観(下位10%)", "#dc2626"),
            (p90, "楽観(上位10%)", "#16a34a"),
        ]:
            fig2.add_vline(x=val, line_dash="dash", line_color=color,
                           annotation_text=label, annotation_position="top")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                           xaxis_title=f"株価({cur})", yaxis_title="シナリオ数",
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("計算の前提を見る"):
        st.markdown(f"""
        - データ取得元:{source}
        - 取得データ:過去 **{len(prices)} 営業日**({prices.index[0]:%Y/%m/%d} 〜 {prices.index[-1]:%Y/%m/%d})
        - 日次平均リターン:{mu*100:.4f}% / 年率ボラティリティ:{annual_vol:.1f}%
        - 手法:幾何ブラウン運動(GBM)。過去の値動きの「平均」と「ばらつき」が今後も続くという前提のシミュレーションです
        - 決算・金利変更・地政学イベントなどの個別要因は考慮していません
        - 本結果は情報提供であり、投資判断はご自身の責任で行ってください
        """)
else:
    st.info("← 左のサイドバーで銘柄コードを入力し、「分析を実行する」を押してください")
