# -*- coding: utf-8 -*-
"""
株価モンテカルロ予測スライド ジェネレーター
=================================================
銘柄コードを指定して実行すると、5年分の株価を取得し、
そのデータを埋め込んだHTMLスライド(単一ファイル)を自動生成します。
モンテカルロ(10,000回)とグラフ描画はHTMLを開いた時にブラウザ側で実行されます。

【使い方】
1. 必要なライブラリを一度だけインストール:
       pip install yfinance pandas
2. 下の「設定」の TICKER と DISPLAY_NAME を変えて、このファイルを実行:
       python stock_montecarlo_generator.py
3. デスクトップに montecarlo_slides.html ができるので、ダブルクリックで開く。

【銘柄コード(Yahoo Finance形式)の例】
   日経平均   : ^N225
   トヨタ      : 7203.T     (日本株は「証券コード.T」)
   ソニーG    : 6758.T
   アップル    : AAPL
   S&P500     : ^GSPC
   NASDAQ総合 : ^IXIC

※ ネットにつながらない/取得が不安定なときは、LOCAL_CSV に
   ダウンロード済みCSVのパスを入れると、それを使って生成します。
"""

# ============ 設定(ここだけ変えればOK) ============
TICKER       = "^N225"          # 取得する銘柄コード(Yahoo Finance形式)
DISPLAY_NAME = "日経225"         # スライドに表示する名前
YEARS        = 5                # 取得する年数
OUTPUT_NAME  = "montecarlo_slides.html"   # 出力ファイル名(デスクトップに自動保存。どのPCでもOK)
LOCAL_CSV    = ""               # CSVから作る場合はパスを指定(例 "n225.csv")。空ならネット取得
# ====================================================

import json, sys, datetime, os

def _desktop_dir():
    """WindowsでもMac/Linuxでも、英語/日本語表記でもデスクトップを探す。無ければホーム。"""
    home = os.path.expanduser("~")
    for _n in ("Desktop", "デスクトップ"):
        _p = os.path.join(home, _n)
        if os.path.isdir(_p):
            return _p
    return home

OUTPUT = os.path.join(_desktop_dir(), OUTPUT_NAME)

def rows_from_dataframe(df):
    """pandas DataFrame から [["YYYY-MM-DD", 終値], ...] を作る(昇順)"""
    import pandas as pd
    df = df.copy()
    # 列名を小文字化して終値列を探す
    cols = {str(c).lower(): c for c in df.columns}
    close_col = None
    for key in ("close", "adj close", "price", "終値"):
        if key in cols:
            close_col = cols[key]; break
    if close_col is None:
        close_col = df.columns[-1]
    s = pd.to_numeric(df[close_col].astype(str).str.replace(",", ""), errors="coerce")
    out = []
    for idx, val in s.items():
        if pd.isna(val):
            continue
        # 日付の取り出し(インデックスが日付 or Date列)
        if isinstance(idx, (datetime.date, datetime.datetime, pd.Timestamp)):
            d = pd.Timestamp(idx).strftime("%Y-%m-%d")
        else:
            d = str(idx)[:10]
        out.append([d, round(float(val), 2)])
    out.sort(key=lambda r: r[0])
    return out

def fetch_online(ticker, years):
    import yfinance as yf
    print("取得中: %s (%d年分) ..." % (ticker, years))
    df = yf.download(ticker, period="%dy" % years, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("データが空でした")
    # 単一銘柄でもMultiIndex列になる場合への対応
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = [c[0] for c in df.columns]
    return rows_from_dataframe(df)

def read_local(path):
    import pandas as pd
    print("CSV読み込み: %s" % path)
    df = pd.read_csv(path)
    date_col = None
    for dc in ("Date", "date", "日付", "Datetime"):
        if dc in df.columns:
            date_col = dc; break
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)
    return rows_from_dataframe(df)

def main():
    try:
        rows = read_local(LOCAL_CSV) if LOCAL_CSV else fetch_online(TICKER, YEARS)
    except ImportError:
        print("！ ライブラリが足りません。先に  pip install yfinance pandas  を実行してください。")
        sys.exit(1)
    except Exception as e:
        print("！ データ取得に失敗しました:", e)
        print("  ネット制限の可能性があります。Yahoo/StooqでCSVを保存し、")
        print("  LOCAL_CSV にそのパスを書いて再実行してください。")
        sys.exit(1)

    if len(rows) < 60:
        print("！ データが少なすぎます(%d件)。" % len(rows)); sys.exit(1)

    print("取得OK: %s 〜 %s / %d営業日" % (rows[0][0], rows[-1][0], len(rows)))

    # 参考: Python側でも中央値・95%区間をざっくり計算して表示
    try:
        import math, random
        closes = [r[1] for r in rows]
        lr = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
        m = sum(lr)/len(lr); var = sum((x-m)**2 for x in lr)/(len(lr)-1); sd = var**0.5
        cur = closes[-1]; term = []
        for _ in range(10000):
            p = cur
            for _ in range(252):
                p *= math.exp((m-0.5*var)+sd*random.gauss(0, 1))
            term.append(p)
        term.sort()
        def q(pp):
            i = (len(term)-1)*pp; lo = int(i); hi = min(lo+1, len(term)-1)
            return term[lo]+(term[hi]-term[lo])*(i-lo)
        print("  現在値 %s / 1年後 中央値 %s / 95%%区間 %s 〜 %s"
              % (round(cur), round(q(.5)), round(q(.025)), round(q(.975))))
    except Exception:
        pass

    data_js = "const SAMPLE_N225=" + json.dumps(rows, ensure_ascii=False) + ";"
    html = TEMPLATE.replace("/*SAMPLE_DATA*/", data_js)

    # 表示ラベルの差し替え(日経225以外でも正しく表示されるように)
    label = DISPLAY_NAME
    repl = [
        ("日経225 (サンプル)", label + "（取得データ）"),
        ("日経225 モンテカルロ予測", label + " モンテカルロ予測"),
        (">日経225 <span", ">" + label + " <span"),
        ("MONTE CARLO FORECAST ・ NIKKEI 225", "MONTE CARLO FORECAST ・ " + label),
        ("1年後の日経平均が", "1年後の" + label + "が"),
        ("サンプル(日経225・直近5年)で分析しました。", label + "の取得データで分析しました。"),
        ("サンプル(日経225)を読み込み中…", label + "を読み込み中…"),
        ("サンプル(日経225)で試す", label + "を再計算"),
    ]
    for a, b in repl:
        html = html.replace(a, b)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    print("生成しました → %s (ダブルクリックで開けます)" % OUTPUT)

# ====== HTMLテンプレート(編集不要) ======
TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>日経225 モンテカルロ予測</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family:'Hiragino Kaku Gothic ProN','Noto Sans JP',sans-serif;
    background:#888888; min-height:100vh;
    display:flex; align-items:center; justify-content:center;
  }
  .slide-wrapper { width:1280px; height:720px; transform-origin:center center; }
  .slide-container { position:relative; width:1280px; height:720px; }
  .slide {
    position:absolute; inset:0; width:1280px; height:720px;
    background:#ffffff; opacity:0; pointer-events:none;
    transition:opacity .35s ease; overflow:hidden;
  }
  .slide.active { opacity:1; pointer-events:auto; }
  /* 左端バーは全スライド濃紺で統一 */
  .slide::before { content:""; position:absolute; left:0; top:0; bottom:0; width:10px; background:#1a2744; }
  .slide-inner { position:absolute; inset:0; padding:46px 76px 44px 78px; display:flex; flex-direction:column; }

  .category { font-size:14px; font-weight:800; letter-spacing:.22em; color:#1a2744; margin-bottom:14px; }
  .slide-title { font-size:40px; font-weight:800; color:#1a2744; line-height:1.25; margin-bottom:22px; }
  .footer { position:absolute; left:78px; right:76px; bottom:20px; display:flex; justify-content:space-between; font-size:14px; color:#4a5e78; }

  /* 表紙 */
  .cover-big { font-size:52px; font-weight:800; color:#1a2744; line-height:1.2; margin-bottom:12px; }
  .cover-big span { border-bottom:5px solid #1a2744; padding-bottom:2px; }
  .cover-sub { font-size:18px; color:#3d4f6e; line-height:1.8; margin-bottom:24px; max-width:780px; }

  /* コントロールパネル(白背景用) */
  .panel { background:#dce4f0; border:1px solid #d0d8e8; border-radius:12px; padding:22px 24px; max-width:900px; }
  .panel-row { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:14px; }
  .panel label { font-size:14px; color:#1a2744; font-weight:800; min-width:96px; }
  .panel input[type=text], .panel textarea {
    font-family:inherit; font-size:15px; color:#1a2744; background:#ffffff;
    border:1px solid #c0cbdd; border-radius:7px; padding:9px 12px;
  }
  .panel input[type=text] { width:190px; }
  .panel textarea { width:100%; height:62px; resize:none; line-height:1.4; }
  .btn {
    font-family:inherit; font-size:14px; font-weight:800; letter-spacing:.04em;
    color:#ffffff; background:#1a2744; border:none; border-radius:7px;
    padding:10px 18px; cursor:pointer; transition:filter .15s;
  }
  .btn:hover { filter:brightness(1.25); }
  .btn-ghost { color:#1a2744; background:#ffffff; border:1px solid #1a2744; }
  .btn-ghost:hover { background:#eef1f7; }
  .hint { font-size:14px; color:#4a5e78; line-height:1.7; margin-top:4px; }
  .status { font-size:14px; color:#1a2744; font-weight:700; margin-top:10px; min-height:20px; }
  .status.err { color:#1a2744; font-weight:800; }

  /* 統計カードグリッド */
  .stat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; margin-top:6px; }
  .stat-card { background:#dce4f0; border-radius:10px; padding:20px 22px; border-top:4px solid #1a2744; }
  .stat-card .lab { font-size:14px; font-weight:800; letter-spacing:.06em; color:#4a5e78; margin-bottom:8px; }
  .stat-card .val { font-size:34px; font-weight:900; color:#1a2744; line-height:1; }
  .stat-card .unit { font-size:16px; font-weight:700; color:#1a2744; margin-left:4px; }
  .stat-card .note { font-size:14px; color:#3d4f6e; line-height:1.6; margin-top:10px; }

  /* チャート領域 */
  .chart-box { flex:1; display:flex; align-items:center; justify-content:center; min-height:0; }
  canvas { max-width:100%; max-height:100%; }
  .chart-cap { font-size:14px; color:#4a5e78; line-height:1.7; margin-top:6px; }

  /* ステップフロー */
  .step-flow { display:flex; align-items:stretch; gap:14px; margin-top:10px; }
  .step-box { flex:1; background:#dce4f0; border-radius:10px; padding:22px 20px; border-top:4px solid #1a2744; }
  .step-num { font-size:30px; font-weight:900; color:#1a2744; line-height:1; margin-bottom:12px; }
  .step-box h3 { font-size:18px; font-weight:800; color:#1a2744; line-height:1.4; margin-bottom:8px; }
  .step-box p { font-size:14px; color:#3d4f6e; line-height:1.7; }
  .step-arrow { display:flex; align-items:center; font-size:26px; color:#1a2744; font-weight:800; }

  /* 結論パネル(濃紺ハイライト・白文字) */
  .result-panel { display:flex; gap:18px; margin-top:4px; }
  .result-main { flex:1.3; background:#1a2744; border-radius:12px; padding:26px 30px; color:#fff; text-align:center; }
  .result-ci { flex:1; background:#1a2744; border-radius:12px; padding:26px 30px; color:#fff; text-align:center; }
  .rlab { font-size:17px; font-weight:700; color:#c0cde3; letter-spacing:.04em; margin-bottom:12px; }
  .rval { font-size:62px; font-weight:900; color:#ffffff; line-height:1; }
  .rval2 { font-size:38px; font-weight:900; color:#ffffff; line-height:1.25; }
  .result-cur { font-size:17px; color:#3d4f6e; margin-top:16px; font-weight:700; text-align:center; }

  .caveat-list { margin-top:22px; display:flex; flex-direction:column; gap:12px; }
  .caveat { display:flex; gap:12px; align-items:flex-start; }
  .caveat .ct { font-size:15px; font-weight:800; background:#1a2744; color:#fff; padding:3px 10px; border-radius:4px; white-space:nowrap; margin-top:2px; }
  .caveat p { font-size:17px; color:#3d4f6e; line-height:1.6; }

  /* ナビ */
  .nav { position:fixed; bottom:18px; left:50%; transform:translateX(-50%); display:flex; align-items:center; gap:18px; z-index:50; }
  .nav button { font-family:inherit; font-size:14px; font-weight:700; white-space:nowrap; background:#1a2744; color:#fff; border:none; border-radius:6px; padding:9px 16px; cursor:pointer; }
  .nav button:hover { filter:brightness(1.3); }
  .dots { display:flex; gap:8px; }
  .dot { width:10px; height:10px; border-radius:50%; background:#c8cdd6; cursor:pointer; }
  .dot.active { background:#1a2744; }

  @media print {
    @page { size:A4 landscape; margin:0; }
    body { background:#fff; padding:0; display:block; }
    .slide-wrapper { transform:none !important; }
    .slide-container { position:static; }
    .nav { display:none !important; }
    .slide { position:relative; inset:auto; opacity:1 !important; display:flex; page-break-after:always; }
    .slide:last-child { page-break-after:auto; }
  }
</style>
</head>
<body>
<div class="slide-wrapper" id="slideWrapper">
  <div class="slide-container">

    <!-- 1. 表紙(コントロール) -->
    <div class="slide active">
      <div class="slide-inner">
        <div class="category">MONTE CARLO FORECAST ・ NIKKEI 225</div>
        <div class="cover-big">日経225 <span>モンテカルロ</span>予測</div>
        <div class="cover-sub">過去5年の値動きから「平均的な伸び」と「ブレ幅」を測り、ランダムな試行を1万回繰り返して、1年後の日経平均がどのあたりに着地しそうかを<b style="color:#1a2744">確率の範囲</b>で示します。</div>

        <div class="panel">
          <div class="panel-row">
            <label>銘柄コード</label>
            <input type="text" id="symInput" value="^nkx" spellcheck="false">
            <button class="btn" id="fetchBtn">ネットから取得</button>
            <button class="btn btn-ghost" id="sampleBtn">サンプル(日経225)で試す</button>
          </div>
          <div class="hint">Stooq形式 → 日経平均: <b>^nkx</b> / 日本株: <b>7203.jp</b>(トヨタ) / 米国株: <b>aapl.us</b> / S&amp;P500: <b>^spx</b></div>
          <div class="panel-row" style="margin-top:14px;">
            <label style="align-self:flex-start;">CSVを貼付</label>
            <textarea id="csvInput" placeholder="StooqやYahooでダウンロードしたCSVをここに貼り付け(Date と Close の列があればOK)"></textarea>
          </div>
          <div class="panel-row">
            <button class="btn" id="csvBtn">貼り付けたCSVで分析</button>
            <span class="hint" style="margin:0;">↑ 最新の数値で確実に動かしたいときはこちら</span>
          </div>
          <div class="status" id="status"></div>
        </div>

        <div style="flex:1;"></div>
        <div class="footer"><span>Knowledge Assist Ltd.</span><span>1 / 7</span></div>
      </div>
    </div>

    <!-- 2. 過去チャート -->
    <div class="slide">
      <div class="slide-inner">
        <div class="category" id="cat2">HISTORICAL PRICE</div>
        <div class="slide-title">過去5年の値動き</div>
        <div class="chart-box"><canvas id="histCanvas"></canvas></div>
        <div class="chart-cap" id="histCap">— サンプル(日経225)を読み込み中…</div>
        <div class="footer"><span>Knowledge Assist Ltd.</span><span>2 / 7</span></div>
      </div>
    </div>

    <!-- 3. 統計サマリ -->
    <div class="slide">
      <div class="slide-inner">
        <div class="category">KEY STATISTICS</div>
        <div class="slide-title">過去データから読み取った数字</div>
        <div class="stat-grid">
          <div class="stat-card"><div class="lab">現在の水準</div><div class="val"><span id="sCur">—</span></div><div class="note" id="sCurNote">いちばん新しい終値です。</div></div>
          <div class="stat-card"><div class="lab">年率リターン (平均的な伸び)</div><div class="val"><span id="sRet">—</span><span class="unit">%/年</span></div><div class="note">1年あたり平均で何%動いてきたか。プラスなら上昇傾向。</div></div>
          <div class="stat-card"><div class="lab">年率ボラティリティ (ブレ幅)</div><div class="val"><span id="sVol">—</span><span class="unit">%/年</span></div><div class="note">値動きの激しさ。大きいほど予測の幅も広がります。</div></div>
        </div>
        <div class="stat-grid" style="margin-top:18px;">
          <div class="stat-card"><div class="lab">対象期間</div><div class="val" style="font-size:21px;line-height:1.3;" id="sPeriod">—</div></div>
          <div class="stat-card"><div class="lab">使用した営業日数</div><div class="val"><span id="sDays">—</span><span class="unit">日</span></div></div>
          <div class="stat-card"><div class="lab">シミュレーション回数</div><div class="val"><span id="sRuns">—</span><span class="unit">回</span></div><div class="note">この回数だけ「ありうる1年」を試します。</div></div>
        </div>
        <div style="flex:1;"></div>
        <div class="footer"><span>Knowledge Assist Ltd.</span><span>3 / 7</span></div>
      </div>
    </div>

    <!-- 4. しくみ -->
    <div class="slide">
      <div class="slide-inner">
        <div class="category">HOW IT WORKS</div>
        <div class="slide-title">モンテカルロのしくみ（3ステップ）</div>
        <div class="step-flow">
          <div class="step-box"><div class="step-num">01</div><h3>クセを測る</h3><p>過去5年の毎日の値動きから「平均的にどれくらい動くか(伸び)」と「どれくらいブレるか(ブレ幅)」を計算します。</p></div>
          <div class="step-arrow">→</div>
          <div class="step-box"><div class="step-num">02</div><h3>1万回試す</h3><p>そのクセに従って、サイコロを振るように1日ずつ動かし、1年分を1本の未来として描きます。これを1万本作ります。</p></div>
          <div class="step-arrow">→</div>
          <div class="step-box"><div class="step-num">03</div><h3>範囲で示す</h3><p>1万本の到達点を集計し、「真ん中はこのあたり」「だいたいこの範囲に収まる」を<b>確率</b>で表します。</p></div>
        </div>
        <div class="chart-cap" style="margin-top:22px;">ポイント：1つの数字で「いくらになる」と当てるのではなく、<b>起こりうる範囲</b>を見せるのがモンテカルロの考え方です。過去のクセが続くと仮定した場合の話であり、当たることを保証するものではありません。</div>
        <div style="flex:1;"></div>
        <div class="footer"><span>Knowledge Assist Ltd.</span><span>4 / 7</span></div>
      </div>
    </div>

    <!-- 5. ファンチャート -->
    <div class="slide">
      <div class="slide-inner">
        <div class="category">SIMULATION PATHS</div>
        <div class="slide-title" style="margin-bottom:6px;">1年後の株価予測グラフ</div>
        <div style="font-size:20px;font-weight:700;color:#3d4f6e;margin-bottom:18px;">モンテカルロ・シミュレーション 10,000回</div>
        <div class="chart-box"><canvas id="fanCanvas"></canvas></div>
        <div class="chart-cap">薄いピンクの帯が95%信頼区間（2.5〜97.5%）。細い折れ線1本1本が「ありうる1年」のシナリオ（1万回のうち100本を表示）で、太い線が中央値です。先に行くほど道すじが広がっていくのが分かります。</div>
        <div class="footer"><span>Knowledge Assist Ltd.</span><span>5 / 7</span></div>
      </div>
    </div>

    <!-- 6. ヒストグラム -->
    <div class="slide">
      <div class="slide-inner">
        <div class="category">1-YEAR DISTRIBUTION</div>
        <div class="slide-title">1年後の予測（分布）</div>
        <div class="chart-box"><canvas id="histoCanvas"></canvas></div>
        <div class="chart-cap" id="histoCap">縦線は左から 2.5% / 25% / 中央値 / 75% / 97.5% の目安。両端の外側(2.5%と97.5%)の間が95%信頼区間です。</div>
        <div class="footer"><span>Knowledge Assist Ltd.</span><span>6 / 7</span></div>
      </div>
    </div>

    <!-- 7. 結論 -->
    <div class="slide">
      <div class="slide-inner">
        <div class="category">CONCLUSION &amp; CAUTIONS</div>
        <div class="slide-title">まとめ：1年後の予測</div>
        <div class="result-panel">
          <div class="result-main">
            <div class="rlab">1年後の中央値（最も起こりやすい着地点）</div>
            <div class="rval" id="cMed">—</div>
          </div>
          <div class="result-ci">
            <div class="rlab">95%信頼区間（20回に19回はこの範囲）</div>
            <div class="rval2"><span id="cLo">—</span> 〜 <span id="cHi">—</span></div>
          </div>
        </div>
        <div class="result-cur">現在の水準 <span id="cCur">—</span> ／ 中央50%の範囲 <span id="cMid">—</span> ／ シミュレーション <span id="cRuns">—</span>回</div>
        <div class="caveat-list">
          <div class="caveat"><span class="ct">前提</span><p>「過去5年の値動きのクセが、これからも同じように続く」と仮定した計算です。</p></div>
          <div class="caveat"><span class="ct">弱点</span><p>決算・事件・暴落など、過去になかった大きな出来事は反映されにくいモデルです。</p></div>
          <div class="caveat"><span class="ct">注意</span><p>これは投資助言ではありません。意思決定はご自身の判断と他の情報も合わせて行ってください。</p></div>
        </div>
        <div style="flex:1;"></div>
        <div class="footer"><span>Knowledge Assist Ltd.</span><span>7 / 7</span></div>
      </div>
    </div>

  </div>
</div>

<div class="nav">
  <button onclick="prevSlide()">← 前へ</button>
  <div class="dots" id="dots"></div>
  <button onclick="nextSlide()">次へ →</button>
</div>

<script>
/* ===== サンプルデータ(日経225・直近5年・実データ) ===== */
/*SAMPLE_DATA*/

function scaleSlide(){
  const w=document.getElementById('slideWrapper');
  const s=Math.min(window.innerWidth/1280, window.innerHeight/720)*0.96;
  w.style.transform='scale('+s+')';
}
scaleSlide(); window.addEventListener('resize', scaleSlide);

let current=0;
const slides=document.querySelectorAll('.slide');
const dotsWrap=document.getElementById('dots');
slides.forEach((_,i)=>{ const d=document.createElement('div'); d.className='dot'+(i===0?' active':''); d.onclick=()=>goTo(i); dotsWrap.appendChild(d); });
const dots=document.querySelectorAll('.dot');
function update(){ slides.forEach((s,i)=>s.classList.toggle('active',i===current)); dots.forEach((d,i)=>d.classList.toggle('active',i===current)); }
function nextSlide(){ if(current<slides.length-1){current++;update();} }
function prevSlide(){ if(current>0){current--;update();} }
function goTo(i){ current=i; update(); }
document.addEventListener('keydown', e=>{
  const t=document.activeElement.tagName;
  if(t==='INPUT'||t==='TEXTAREA') return;
  if(e.key==='ArrowRight'||e.key===' '){ e.preventDefault(); nextSlide(); }
  if(e.key==='ArrowLeft'){ e.preventDefault(); prevSlide(); }
});

function gaussian(){ let u=0,v=0; while(u===0)u=Math.random(); while(v===0)v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
function pct(sortedArr, p){ const idx=(sortedArr.length-1)*p; const lo=Math.floor(idx), hi=Math.ceil(idx); if(lo===hi) return sortedArr[lo]; return sortedArr[lo]+(sortedArr[hi]-sortedArr[lo])*(idx-lo); }
function fmt(n){ if(n>=1000) return Math.round(n).toLocaleString('ja-JP'); if(n>=100) return n.toFixed(1); return n.toFixed(2); }

function parseCSV(text){
  const lines=text.trim().split(/\r?\n/).filter(l=>l.trim()!=='');
  if(lines.length<2) throw new Error('データが足りません');
  const delim = lines[0].indexOf(';')>=0 && lines[0].indexOf(',')<0 ? ';' : ',';
  const header=lines[0].split(delim).map(h=>h.replace(/"/g,'').trim().toLowerCase());
  let di=header.indexOf('date');
  let ci=header.indexOf('close'); if(ci<0) ci=header.indexOf('adj close'); if(ci<0) ci=header.indexOf('price'); if(ci<0) ci=header.indexOf('終値');
  if(di<0) di=0;
  if(ci<0) ci=header.length-1;
  const out=[];
  for(let i=1;i<lines.length;i++){
    const c=lines[i].split(delim);
    const d=c[di]?c[di].replace(/"/g,'').trim():'';
    const v=parseFloat((c[ci]||'').replace(/[^0-9.\-]/g,''));
    if(d && isFinite(v) && v>0) out.push([d,v]);
  }
  if(out.length<30) throw new Error('使えるデータが少なすぎます(30日分以上必要)');
  out.sort((a,b)=>a[0]<b[0]?-1:1);
  return out;
}

const N_RUNS=10000, HORIZON=252;
function analyze(series, labelName){
  const closes=series.map(r=>r[1]);
  const dates=series.map(r=>r[0]);
  const logr=[];
  for(let i=1;i<closes.length;i++) logr.push(Math.log(closes[i]/closes[i-1]));
  const m=logr.reduce((a,b)=>a+b,0)/logr.length;
  const variance=logr.reduce((a,b)=>a+(b-m)*(b-m),0)/(logr.length-1);
  const sd=Math.sqrt(variance);
  const cur=closes[closes.length-1];
  const annRet=(Math.exp(m*252)-1)*100;
  const annVol=sd*Math.sqrt(252)*100;

  const paths=new Float32Array(N_RUNS*(HORIZON+1));
  for(let k=0;k<N_RUNS;k++){
    let p=cur; paths[k*(HORIZON+1)]=p;
    for(let t=1;t<=HORIZON;t++){ p=p*Math.exp((m-0.5*variance)+sd*gaussian()); paths[k*(HORIZON+1)+t]=p; }
  }
  const ps=[0.025,0.25,0.5,0.75,0.975];
  const bands=ps.map(()=>new Float64Array(HORIZON+1));
  const col=new Float64Array(N_RUNS);
  for(let t=0;t<=HORIZON;t++){
    for(let k=0;k<N_RUNS;k++) col[k]=paths[k*(HORIZON+1)+t];
    const s=Array.from(col).sort((a,b)=>a-b);
    ps.forEach((pp,pi)=>bands[pi][t]=pct(s,pp));
  }
  const term=new Float64Array(N_RUNS);
  for(let k=0;k<N_RUNS;k++) term[k]=paths[k*(HORIZON+1)+HORIZON];
  const ts=Array.from(term).sort((a,b)=>a-b);
  const tp={p025:pct(ts,0.025),p25:pct(ts,0.25),p50:pct(ts,0.5),p75:pct(ts,0.75),p975:pct(ts,0.975)};

  document.getElementById('sCur').textContent=fmt(cur);
  document.getElementById('sRet').textContent=annRet.toFixed(1);
  document.getElementById('sVol').textContent=annVol.toFixed(1);
  document.getElementById('sPeriod').textContent=dates[0]+' 〜 '+dates[dates.length-1];
  document.getElementById('sDays').textContent=closes.length.toLocaleString('ja-JP');
  document.getElementById('sRuns').textContent=N_RUNS.toLocaleString('ja-JP');
  document.getElementById('cMed').textContent=fmt(tp.p50);
  document.getElementById('cCur').textContent=fmt(cur);
  document.getElementById('cLo').textContent=fmt(tp.p025);
  document.getElementById('cHi').textContent=fmt(tp.p975);
  document.getElementById('cMid').textContent=fmt(tp.p25)+' 〜 '+fmt(tp.p75);
  document.getElementById('cRuns').textContent=N_RUNS.toLocaleString('ja-JP');
  document.getElementById('cat2').textContent='HISTORICAL PRICE ・ '+labelName;
  document.getElementById('histCap').textContent=labelName+'：'+dates[0]+'〜'+dates[dates.length-1]+'（'+closes.length+'営業日）の終値';

  const NS=100; const sample=[];
  for(let k=0;k<NS;k++){ const arr=new Float64Array(HORIZON+1); for(let t=0;t<=HORIZON;t++) arr[t]=paths[k*(HORIZON+1)+t]; sample.push(arr); }
  drawHist(closes); drawFan(cur, bands, sample); drawHisto(ts, cur, tp);
}

function setup(canvas, w, h){ const dpr=Math.min(window.devicePixelRatio||1,2); canvas.width=w*dpr; canvas.height=h*dpr; canvas.style.width=w+'px'; canvas.style.height=h+'px'; const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr); return ctx; }
const NAVY='#1a2744', SUB='#3d4f6e', GRID='#d0d8e8', BAND='#9bb4dd', BANDLT='#cdd9ee';
function axes(ctx,w,h,pad){ ctx.strokeStyle=GRID; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,h-pad.b); ctx.lineTo(w-pad.r,h-pad.b); ctx.stroke(); }

function drawHist(closes){
  const c=document.getElementById('histCanvas'); const w=1040,h=430; const ctx=setup(c,w,h);
  ctx.clearRect(0,0,w,h); const pad={l:86,r:24,t:18,b:40};
  const mn=Math.min(...closes), mx=Math.max(...closes); const range=mx-mn||1;
  const X=i=>pad.l+(w-pad.l-pad.r)*i/(closes.length-1);
  const Y=v=>h-pad.b-(h-pad.t-pad.b)*(v-mn)/range;
  ctx.font='14px sans-serif'; ctx.fillStyle=SUB; ctx.textAlign='right'; ctx.textBaseline='middle';
  for(let g=0;g<=4;g++){ const v=mn+range*g/4; const y=Y(v); ctx.strokeStyle=GRID; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke(); ctx.fillText(fmt(v),pad.l-8,y); }
  axes(ctx,w,h,pad);
  ctx.strokeStyle=NAVY; ctx.lineWidth=2; ctx.beginPath();
  closes.forEach((v,i)=>{ const x=X(i),y=Y(v); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }); ctx.stroke();
}

function drawFan(cur, bands, sample){
  const c=document.getElementById('fanCanvas'); const w=1040,h=420; const ctx=setup(c,w,h);
  ctx.clearRect(0,0,w,h); const pad={l:86,r:24,t:18,b:40}; const H=bands[0].length;
  // y範囲は95%帯(2.5〜97.5%)を基準にし、極端なパスは枠でクリップ
  let mn=Infinity,mx=-Infinity;
  for(let t=0;t<H;t++){ if(bands[0][t]<mn)mn=bands[0][t]; if(bands[4][t]>mx)mx=bands[4][t]; }
  mn=Math.min(mn,cur); mx=Math.max(mx,cur);
  const sp=mx-mn; mn-=sp*0.04; mx+=sp*0.04; const range=mx-mn||1;
  const X=t=>pad.l+(w-pad.l-pad.r)*t/(H-1);
  const Y=v=>h-pad.b-(h-pad.t-pad.b)*(v-mn)/range;
  ctx.font='14px sans-serif'; ctx.fillStyle=SUB; ctx.textAlign='right'; ctx.textBaseline='middle';
  for(let g=0;g<=4;g++){ const v=mn+range*g/4; const y=Y(v); ctx.strokeStyle=GRID; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke(); ctx.fillText(fmt(v),pad.l-8,y); }
  // チャート領域にクリップし、1本ずつの折れ線(シナリオ)を描く
  ctx.save();
  ctx.beginPath(); ctx.rect(pad.l,pad.t,w-pad.l-pad.r,h-pad.t-pad.b); ctx.clip();
  // 95%信頼区間(2.5〜97.5%)を薄いピンクで塗る(折れ線の背面)
  ctx.fillStyle='rgba(236,72,153,0.15)'; ctx.beginPath();
  for(let t=0;t<H;t++){ const x=X(t),y=Y(bands[0][t]); t===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }
  for(let t=H-1;t>=0;t--){ ctx.lineTo(X(t),Y(bands[4][t])); }
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle='rgba(26,39,68,0.16)'; ctx.lineWidth=1.4;
  sample.forEach(p=>{ ctx.beginPath(); for(let t=0;t<H;t++){ const x=X(t),y=Y(p[t]); t===0?ctx.moveTo(x,y):ctx.lineTo(x,y); } ctx.stroke(); });
  // 中央値(太線)
  ctx.strokeStyle=NAVY; ctx.lineWidth=2.8; ctx.beginPath(); for(let t=0;t<H;t++){const v=bands[2][t]; t===0?ctx.moveTo(X(t),Y(v)):ctx.lineTo(X(t),Y(v));} ctx.stroke();
  ctx.restore();
  axes(ctx,w,h,pad);
  ctx.fillStyle=SUB; ctx.textAlign='center'; ctx.textBaseline='top';
  [0,63,126,189,252].forEach(t=>{ ctx.fillText(t===0?'現在':(Math.round(t/21)+'ヶ月後'), X(t), h-pad.b+8); });
}

function drawHisto(sorted, cur, tp){
  const c=document.getElementById('histoCanvas'); const w=1040,h=410; const ctx=setup(c,w,h);
  ctx.clearRect(0,0,w,h); const pad={l:84,r:30,t:62,b:54};
  const lo=tp.p025*0.9, hi=tp.p975*1.05; const range=hi-lo||1; const bins=48;
  const counts=new Array(bins).fill(0);
  sorted.forEach(v=>{ if(v>=lo&&v<=hi){ let b=Math.floor((v-lo)/range*bins); if(b>=bins)b=bins-1; if(b<0)b=0; counts[b]++; } });
  const cmax=Math.max(...counts)||1;
  const X=v=>pad.l+(w-pad.l-pad.r)*(v-lo)/range;
  const bw=(w-pad.l-pad.r)/bins;
  ctx.fillStyle=BAND;
  counts.forEach((n,i)=>{ const bh=(h-pad.t-pad.b)*n/cmax; ctx.fillRect(pad.l+i*bw+1, h-pad.b-bh, bw-2, bh); });
  axes(ctx,w,h,pad);
  const RED='#dc2626';
  const marks=[['2.5%',tp.p025],['25%',tp.p25],['中央値',tp.p50],['75%',tp.p75],['97.5%',tp.p975]];
  marks.forEach((mk,i)=>{ const x=X(mk[1]); const isMed=(i===2);
    ctx.strokeStyle=(isMed?RED:SUB); ctx.lineWidth=(isMed?4:1.4); ctx.setLineDash(isMed?[]:[5,4]);
    ctx.beginPath(); ctx.moveTo(x,isMed?(pad.t-12):pad.t); ctx.lineTo(x,h-pad.b); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle=(isMed?RED:SUB); ctx.font=(isMed?'bold 20px sans-serif':'16px sans-serif');
    ctx.textAlign='center'; ctx.textBaseline='top';
    ctx.fillText(mk[0]+' '+fmt(mk[1]), x, isMed?6:34); });
  if(cur>=lo&&cur<=hi){ const x=X(cur); ctx.strokeStyle='#8da6c9'; ctx.lineWidth=1.5; ctx.setLineDash([2,3]); ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,h-pad.b); ctx.stroke(); ctx.setLineDash([]); }
  ctx.fillStyle=SUB; ctx.font='16px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='top';
  for(let g=0;g<=4;g++){ const v=lo+range*g/4; ctx.fillText(fmt(v), pad.l+(w-pad.l-pad.r)*g/4, h-pad.b+10); }
}

function setStatus(msg, err){ const s=document.getElementById('status'); s.textContent=msg; s.className='status'+(err?' err':''); }
async function tryFetch(url, ms){ const ctrl=new AbortController(); const id=setTimeout(()=>ctrl.abort(), ms); try{ const r=await fetch(url,{signal:ctrl.signal}); clearTimeout(id); if(!r.ok) throw new Error('HTTP '+r.status); return await r.text(); } catch(e){ clearTimeout(id); throw e; } }
async function fetchStooq(sym){
  const stooq='https://stooq.com/q/d/l/?s='+encodeURIComponent(sym.toLowerCase())+'&i=d';
  const proxies=[ 'https://api.allorigins.win/raw?url='+encodeURIComponent(stooq), 'https://corsproxy.io/?url='+encodeURIComponent(stooq), 'https://thingproxy.freeboard.io/fetch/'+stooq ];
  for(const p of proxies){ try{ const txt=await tryFetch(p, 12000); if(/date/i.test(txt) && /\d{4}-\d{2}-\d{2}/.test(txt) && /[0-9]+\.[0-9]+/.test(txt)){ return parseCSV(txt); } }catch(e){} }
  throw new Error('取得できませんでした');
}

document.getElementById('sampleBtn').onclick=()=>{ setStatus('計算中…'); setTimeout(()=>{ analyze(SAMPLE_N225.slice(), '日経225 (サンプル)'); setStatus('サンプル(日経225・直近5年)で分析しました。スライドを進めて結果をご覧ください。'); goTo(1); },20); };
document.getElementById('csvBtn').onclick=()=>{ const t=document.getElementById('csvInput').value; if(!t.trim()){ setStatus('CSVが空です。', true); return; } try{ const s=parseCSV(t); setStatus('計算中…'); setTimeout(()=>{ analyze(s, '貼り付けデータ'); setStatus('貼り付けたCSVで分析しました。'); goTo(1); },20); } catch(e){ setStatus('解析できませんでした：'+e.message, true); } };
document.getElementById('fetchBtn').onclick=async ()=>{ const sym=document.getElementById('symInput').value.trim(); if(!sym){ setStatus('銘柄コードを入れてください。',true); return; } setStatus('取得中…（ブラウザの制限で失敗することがあります）'); try{ const s=await fetchStooq(sym); analyze(s, sym.toUpperCase()); setStatus('「'+sym+'」を取得して分析しました。'); goTo(1); } catch(e){ setStatus('自動取得に失敗しました。Stooq/Yahooで「CSVダウンロード」して貼り付ける方法をお試しください。サンプルは下のボタンで動きます。', true); } };

/* 起動時：サンプルで全スライドを埋めておく */
analyze(SAMPLE_N225.slice(), '日経225 (サンプル)');
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()