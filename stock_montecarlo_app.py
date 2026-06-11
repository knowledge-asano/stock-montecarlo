# -*- coding: utf-8 -*-
"""
株価モンテカルロ予測アプリ(Streamlit版)
========================================
銘柄コードを入力すると、過去5年分の株価を自動取得し、
モンテカルロシミュレーションで1年後の株価分布を予測します。

【事前準備(初回のみ、コマンドプロンプトで実行)】
    pip install streamlit yfinance plotly pandas numpy

【起動方法】
    streamlit run stock_montecarlo_app.py
    → 自動でブラウザが開きます(開かない場合は http://localhost:8501 へ)
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# ----------------------------------------------------------
# 設定値(必要に応じて変更可能)
# ----------------------------------------------------------
N_SIMULATIONS = 10_000   # シミュレーション回数(1万通り)
N_DAYS = 252             # 予測日数(1年 ≒ 252営業日)
HISTORY_PERIOD = "5y"    # 取得する過去データの期間

# ----------------------------------------------------------
# ページ設定
# ----------------------------------------------------------
st.set_page_config(page_title="株価モンテカルロ予測", layout="wide")
st.title("📈 株価モンテカルロ予測(1年後)")
st.caption("過去5年の値動きパターンをもとに、1年後の株価を1万通りシミュレーションします")

# ----------------------------------------------------------
# 入力部分(画面左のサイドバー)
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
    if t.endswith(".T") or t in ("^N225", "998407.O"):
        return "円"
    return "ドル"


@st.cache_data(show_spinner=False)
def fetch_prices(ticker_code: str) -> pd.Series:
    """過去5年分の終値を取得(同じ銘柄は再取得せずキャッシュを使う)"""
    data = yf.download(ticker_code, period=HISTORY_PERIOD, auto_adjust=True, progress=False)
    if data is None or data.empty:
        return pd.Series(dtype=float)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # 複数列で返る場合への対応
        close = close.iloc[:, 0]
    return close.dropna()


def run_monte_carlo(prices: pd.Series):
    """
    幾何ブラウン運動(GBM)によるモンテカルロシミュレーション。
    過去の日次リターンの「平均」と「ばらつき(標準偏差)」を測り、
    そのクセを保ったままサイコロを振って、1年分の値動きを1万通り作る。
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean()      # 1日あたりの平均リターン
    sigma = log_returns.std()    # 1日あたりのばらつき(リスク)

    last_price = float(prices.iloc[-1])
    rng = np.random.default_rng(seed=42)  # 再現性のため乱数を固定

    # (日数 × シミュレーション数) のランダムな日次リターンを一括生成
    daily = rng.normal(mu, sigma, size=(N_DAYS, N_SIMULATIONS))
    paths = last_price * np.exp(np.cumsum(daily, axis=0))
    paths = np.vstack([np.full(N_SIMULATIONS, last_price), paths])  # 初日=現在値

    return paths, last_price, mu, sigma


# ----------------------------------------------------------
# メイン処理
# ----------------------------------------------------------
if run:
    with st.spinner(f"{ticker} の株価データを取得中…"):
        prices = fetch_prices(ticker.strip())

    if prices.empty:
        st.error("データを取得できませんでした。銘柄コードを確認してください(例: 7203.T)。")
        st.stop()

    if len(prices) < 250:
        st.warning("取得できたデータが1年分未満です。予測の信頼性が下がる点にご注意ください。")

    cur = currency_label(ticker)
    paths, last_price, mu, sigma = run_monte_carlo(prices)
    final_prices = paths[-1]  # 1年後(252営業日後)の株価 1万通り

    # ---- 主要な統計値 ----
    p10 = np.percentile(final_prices, 10)    # 悲観シナリオ(下位10%)
    p50 = np.percentile(final_prices, 50)    # 中央値
    p90 = np.percentile(final_prices, 90)    # 楽観シナリオ(上位10%)
    prob_up = (final_prices > last_price).mean() * 100  # 現在値を上回る確率
    annual_vol = sigma * np.sqrt(252) * 100             # 年率ボラティリティ

    # ---- サマリーカード ----
    st.subheader(f"分析結果:{ticker}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("現在値", f"{last_price:,.0f} {cur}")
    c2.metric("悲観シナリオ(下位10%)", f"{p10:,.0f} {cur}", f"{(p10/last_price-1)*100:+.1f}%")
    c3.metric("中央値", f"{p50:,.0f} {cur}", f"{(p50/last_price-1)*100:+.1f}%")
    c4.metric("楽観シナリオ(上位10%)", f"{p90:,.0f} {cur}", f"{(p90/last_price-1)*100:+.1f}%")
    c5.metric("1年後に上昇している確率", f"{prob_up:.1f}%")

    st.divider()
    col_left, col_right = st.columns(2)

    # ---- 左:過去5年+シミュレーション例 ----
    with col_left:
        st.markdown("##### 過去5年の実績と、シミュレーションの例(100本)")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=prices.index, y=prices.values,
            name="過去の実績", line=dict(color="#1a2744", width=2),
        ))
        future_dates = pd.bdate_range(start=prices.index[-1], periods=N_DAYS + 1)
        for i in range(100):  # 1万本全部描くと重いので100本だけ見せる
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

    # ---- 右:1年後の株価分布(ヒストグラム) ----
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

    # ---- 補足情報 ----
    with st.expander("計算の前提を見る"):
        st.markdown(f"""
        - 取得データ:過去 **{len(prices)} 営業日**({prices.index[0]:%Y/%m/%d} 〜 {prices.index[-1]:%Y/%m/%d})
        - 日次平均リターン:{mu*100:.4f}% / 年率ボラティリティ:{annual_vol:.1f}%
        - 手法:幾何ブラウン運動(GBM)。過去の値動きの「平均」と「ばらつき」が今後も続くという前提のシミュレーションです
        - 決算・金利変更・地政学イベントなどの個別要因は考慮していません
        - 本結果は情報提供であり、投資判断はご自身の責任で行ってください
        """)
else:
    st.info("← 左のサイドバーで銘柄コードを入力し、「分析を実行する」を押してください")
