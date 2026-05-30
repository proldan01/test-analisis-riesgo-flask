// ============================================================
// Financial Intelligence Platform — Frontend JS
// ============================================================

'use strict';

// ── Global state ──────────────────────────────────────────
let _lastParams = null;
let _lastResult = null;
let _echartsInstances = {};   // id → echarts instance
let _numAssets = parseInt(document.getElementById('numAssets').value, 10);
let _activeTab = 'overview';

// ── Sidebar toggle ────────────────────────────────────────
function toggleSidebar() {
  const sb       = document.getElementById('sidebar');
  const floatBtn = document.getElementById('sbFloatBtn');
  const colBtn   = sb.querySelector('.sb-collapse-btn');
  const isNowCollapsed = sb.classList.toggle('collapsed');

  // Update button icons
  if (colBtn)   colBtn.textContent = isNowCollapsed ? '▶' : '◀';
  if (floatBtn) floatBtn.style.display = isNowCollapsed ? 'flex' : 'none';

  // Persist preference
  localStorage.setItem('sbCollapsed', isNowCollapsed ? '1' : '0');

  // Resize charts after transition finishes
  setTimeout(resizeAllCharts, 300);
}

// Restore sidebar state from localStorage on load
window.addEventListener('DOMContentLoaded', () => {
  if (localStorage.getItem('sbCollapsed') === '1') {
    // Collapse without animation on first load
    const sb = document.getElementById('sidebar');
    sb.style.transition = 'none';
    sb.classList.add('collapsed');
    const floatBtn = document.getElementById('sbFloatBtn');
    if (floatBtn) floatBtn.style.display = 'flex';
    const colBtn = sb.querySelector('.sb-collapse-btn');
    if (colBtn) colBtn.textContent = '▶';
    setTimeout(() => { sb.style.transition = ''; }, 50);
  }
});

// ── Tab switching ─────────────────────────────────────────
function switchTab(name) {
  _activeTab = name;
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  const idx = ['overview','benchmark','correlation','covariance','risk','simulator','export'].indexOf(name);
  document.querySelectorAll('.tab-btn')[idx].classList.add('active');
  // Trigger ECharts resize on reveal
  setTimeout(resizeAllCharts, 50);
}

function resizeAllCharts() {
  Object.values(_echartsInstances).forEach(c => { try { c.resize(); } catch(e){} });
}

window.addEventListener('resize', resizeAllCharts);

// ── Ticker row management ─────────────────────────────────
function pickTicker(sel, idx) {
  if (sel.value) {
    document.getElementById('asset_' + idx).value = sel.value.toUpperCase();
    sel.value = '';
  }
}

function addAsset() {
  if (_numAssets >= 10) return;
  const i = _numAssets;
  const opts = POPULAR.map(s => `<option value="${s}">${s}</option>`).join('');
  const row = `
    <div class="sb-ticker-row" id="tickerRowExtra${i}">
      <input class="sb-input ticker-input" placeholder="Type ticker" id="asset_${i}"
             style="text-transform:uppercase">
      <select class="sb-select ticker-pick" onchange="pickTicker(this,${i})" id="pick_${i}">
        <option value="">or pick</option>${opts}
      </select>
    </div>`;
  document.getElementById('extraTickerRows').insertAdjacentHTML('beforeend', row);
  _numAssets++;
  document.getElementById('numAssets').value = _numAssets;
}

function removeAsset() {
  if (_numAssets <= 1) return;
  _numAssets--;
  document.getElementById('numAssets').value = _numAssets;
  const extra = document.getElementById('tickerRowExtra' + _numAssets);
  if (extra) { extra.remove(); return; }
  const base = document.getElementById('tickerRow' + _numAssets);
  if (base) base.remove();
}

// ── Date helpers ──────────────────────────────────────────
function updateDates() {
  const val = document.getElementById('period').value;
  const div = document.getElementById('customDates');
  div.style.display = val === 'custom' ? 'block' : 'none';
}

function getDateRange() {
  const val = document.getElementById('period').value;
  const today = new Date();
  const fmt = d => d.toISOString().split('T')[0];
  if (val === 'custom') {
    return {
      start: document.getElementById('customStart').value,
      end:   document.getElementById('customEnd').value || fmt(today),
    };
  }
  const days = parseInt(val, 10);
  const start = new Date(today);
  start.setDate(start.getDate() - days);
  return { start: fmt(start), end: fmt(today) };
}

// ── Build params object ───────────────────────────────────
function buildParams() {
  const tickers = [];
  for (let i = 0; i < _numAssets; i++) {
    const v = (document.getElementById('asset_' + i) || {}).value || '';
    if (v.trim()) tickers.push(v.trim().toUpperCase());
  }
  const { start, end } = getDateRange();
  const benchSel = document.getElementById('benchmark');
  const benchOpt = benchSel.options[benchSel.selectedIndex];
  const customRaw = document.getElementById('customEmas').value || '';
  const customEmas = customRaw.split(',').map(s=>s.trim()).filter(s=>/^\d+$/.test(s)).map(Number);

  return {
    tickers,
    start, end,
    freq:        document.getElementById('freq').value,
    bench_ticker: benchSel.value,
    bench_name:   benchOpt.dataset.name || benchOpt.text,
    rfr:          parseFloat(document.getElementById('rfr').value) || 0.0457,
    conf:         parseInt(document.getElementById('conf').value, 10) / 100,
    capital:      parseFloat(document.getElementById('capital').value) || 10000,
    ema_cfg: {
      "7":   document.getElementById('ema7').checked,
      "30":  document.getElementById('ema30').checked,
      "50":  document.getElementById('ema50').checked,
      "200": document.getElementById('ema200').checked,
    },
    custom_emas: customEmas,
    show_bb:   document.getElementById('showBB').checked,
    show_sigs: document.getElementById('showSigs').checked,
    dcf_g:    parseFloat(document.getElementById('dcfG').value) / 100,
    dcf_wacc: parseFloat(document.getElementById('dcfWacc').value) / 100,
    dcf_tg:   parseFloat(document.getElementById('dcfTg').value) / 100,
    dcf_yrs:  parseInt(document.getElementById('dcfYrs').value, 10),
    mc_n:     parseInt(document.getElementById('mcN').value, 10),
    sel_chart: tickers[0] || '',
    sel_sim:   tickers[0] || '',
  };
}

// ── Loading overlay ───────────────────────────────────────
function setLoading(active, msg) {
  const ov = document.getElementById('loadingOverlay');
  const txt = document.getElementById('loadingText');
  if (active) {
    ov.classList.add('active');
    if (msg) txt.textContent = msg;
    document.getElementById('runBtn').disabled = true;
  } else {
    ov.classList.remove('active');
    document.getElementById('runBtn').disabled = false;
  }
}

// ── Main: run analysis ────────────────────────────────────
async function runAnalysis() {
  const params = buildParams();
  if (!params.tickers.length) {
    alert('Enter at least one ticker symbol.');
    return;
  }
  _lastParams = params;

  // Animated loading messages so user knows it's working (can take 30-90 sec)
  const messages = [
    'Fetching market data from Yahoo Finance...',
    'Computing technical indicators...',
    'Running ML signals (Random Forest)...',
    'Building charts...',
    'Almost done...',
  ];
  let msgIdx = 0;
  setLoading(true, messages[0]);
  const msgInterval = setInterval(() => {
    msgIdx = Math.min(msgIdx + 1, messages.length - 1);
    document.getElementById('loadingText').textContent = messages[msgIdx];
  }, 18000);

  try {
    const resp = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });

    clearInterval(msgInterval);

    // Handle non-JSON responses (e.g. 504 Gateway Timeout HTML from Render)
    const contentType = resp.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const text = await resp.text();
      setLoading(false);
      showError(`Server error ${resp.status}: ${text.substring(0, 300)}`);
      return;
    }

    const data = await resp.json();
    if (!data.ok) {
      setLoading(false);
      showError(data.error || 'Analysis failed');
      return;
    }
    _lastResult = data;
    setLoading(true, 'Rendering charts...');
    await new Promise(r => setTimeout(r, 10));
    renderAll(data, params);
    setLoading(false);
  } catch (err) {
    clearInterval(msgInterval);
    setLoading(false);
    showError('Connection error: ' + err.message +
      '\n\nTip: The free Render server may have timed out. Try again with fewer tickers or a shorter date range.');
  }
}

function showError(msg) {
  const box = document.getElementById('welcomeBox');
  box.innerHTML = `<div class="error-box" style="text-align:left;white-space:pre-wrap">⚠️ ${escHtml(msg)}</div>`;
  box.style.display = 'block';
}

// ── Render all sections ───────────────────────────────────
function renderAll(data, params) {
  document.getElementById('welcomeBox').style.display = 'none';
  document.getElementById('results').style.display = 'block';

  renderVix(data.vix);
  renderKPIs(data.kpis);
  renderOverviewHeader(data.meta);
  renderCumulativeChart(data.charts.cumulative);
  renderCandleSelector(data.meta.valid, data.meta.sel_chart);
  renderTrendBar(data.charts.trend, data.meta.sel_chart);
  renderCandleChart(data.charts.candle);
  renderSignalCards(data.signals, data.meta.valid);

  renderBenchTab(data.charts, data.tables.bench_summary, data.meta);
  renderCorrTab(data.charts, data.tables.corr, data.meta.valid, params);
  renderCovTab(data.charts, data.tables.cov, data.mc_weights, data.meta);
  renderRiskTab(data.charts, data.tables.risk, data.meta);
  renderSimulatorSelector(data.meta.valid, data.meta.sel_sim);
  renderSimulator(data.simulator);
  renderExportTab(data.meta, params);
}

// ── VIX bar ───────────────────────────────────────────────
function renderVix(vix) {
  const el = document.getElementById('vixBar');
  if (!vix || !vix.cur) { el.innerHTML = ''; return; }
  el.innerHTML = `
    <div class="vix-bar">
      <div>
        <span style="color:#7090b0;font-size:.7em;text-transform:uppercase;letter-spacing:.08em">CBOE VIX</span><br>
        <span style="color:${vix.col};font-size:1.8em;font-weight:700;font-family:'Share Tech Mono',monospace">${vix.cur.toFixed(2)}</span>
        <span style="color:${vix.ch_col};font-size:.88em;margin-left:8px">${vix.arrow} ${Math.abs(vix.chg).toFixed(2)}</span>
      </div>
      <div style="text-align:right">
        <span style="color:#7090b0;font-size:.7em">MARKET REGIME</span><br>
        <span style="color:${vix.col};font-weight:600;font-size:.88em">${vix.regime}</span>
      </div>
    </div>`;
}

// ── KPI cards ─────────────────────────────────────────────
function renderKPIs(kpis) {
  const el = document.getElementById('kpiGrid');
  el.innerHTML = kpis.map(k => {
    const dc = k.daily_chg;
    const dcCls = dc >= 0 ? 'kpi-pos' : 'kpi-neg';
    const dcStr = (dc >= 0 ? '+' : '') + dc.toFixed(2) + '%';
    const volStr = k.vol_pct != null ? k.vol_pct.toFixed(1) + '%' : 'N/A';
    const betaStr = k.beta != null ? k.beta.toFixed(2) : 'N/A';
    return `<div class="kpi-card">
      <div class="kpi-label">${escHtml(k.name)}</div>
      <div class="kpi-ticker">${escHtml(k.ticker)}</div>
      <div class="kpi-price">$${fmtNum(k.price)}</div>
      <div class="${dcCls}">${dcStr}</div>
      <div class="kpi-sub">Vol: ${volStr} &nbsp;|&nbsp; β: ${betaStr}</div>
    </div>`;
  }).join('');
}

function renderOverviewHeader(meta) {
  document.getElementById('overviewHeader').textContent =
    `Selected Assets — ${meta.valid.length}`;
}

// ── ECharts rendering ─────────────────────────────────────
function renderEChart(domId, option) {
  const el = document.getElementById(domId);
  if (!el) return;
  let chart = _echartsInstances[domId];
  if (!chart) {
    chart = echarts.init(el, 'dark');
    _echartsInstances[domId] = chart;
  }
  chart.setOption(option, true);
}

function renderCumulativeChart(opt) {
  if (opt) renderEChart('chartCumulative', opt);
}

function renderCandleSelector(valid, sel) {
  const el = document.getElementById('chartAssetSel');
  el.innerHTML = valid.map(t =>
    `<option value="${t}" ${t===sel?'selected':''}>${t}</option>`
  ).join('');
}

function renderTrendBar(trend, ticker) {
  const el = document.getElementById('trendBar');
  if (!trend || !trend.score === undefined) { el.innerHTML=''; return; }
  const flag = b => b ? '✅' : '⬜';
  const cc = trend.candle || 'default';
  const col = cc === 'green' ? '#00e87a' : cc === 'purple' ? '#9b59b6' : '#888';
  const rsiCol = trend.rsi < 30 ? '#4da6ff' : trend.rsi > 70 ? '#ff4545' : '#94a3b8';
  el.innerHTML = `
    <div class="trend-bar">
      <span class="label">TREND SCORE</span>
      <span class="val" style="color:${col}">${trend.score}/4</span>
      <span class="flags">
        ${flag(trend.weekly_up)} Weekly &nbsp;
        ${flag(trend.daily_up)} Daily &nbsp;
        ${flag(trend.swing_up)} Swing &nbsp;
        ${flag(trend.short_up)} Short
      </span>
      <span class="label">CANDLE</span>
      <span style="color:${col};font-size:.83em;font-weight:600">${cc.toUpperCase()}</span>
      <span class="label">RSI</span>
      <span style="color:${rsiCol};font-size:.92em;font-weight:600">${trend.rsi}</span>
    </div>`;
}

function renderCandleChart(opt) {
  if (opt) renderEChart('chartCandle', opt);
}

async function switchCandleChart(ticker) {
  if (!_lastParams) return;
  setLoading(true, `Loading ${ticker} chart...`);
  try {
    const resp = await fetch('/api/candle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ticker,
        start: _lastParams.start, end: _lastParams.end, freq: _lastParams.freq,
        ema_cfg: _lastParams.ema_cfg, custom_emas: _lastParams.custom_emas,
        show_bb: _lastParams.show_bb, show_sigs: _lastParams.show_sigs,
      }),
    });
    const data = await resp.json();
    if (data.ok) {
      renderTrendBar(data.trend, ticker);
      renderCandleChart(data.candle);
    }
  } catch(e) { /* silent */ }
  setLoading(false);
}

// ── Signal cards ──────────────────────────────────────────
function renderSignalCards(signals, valid) {
  const el = document.getElementById('signalCards');
  el.innerHTML = valid.map((t, idx) => {
    const s = signals[t];
    if (!s) return '';
    const r = s.rec;
    const tc_colors = {bullish:'#00e87a',income:'#ffd700',bearish:'#ff4545',neutral:'#94a3b8','bullish-neutral':'#40e0d0'};

    const strats = (s.strats || []).map(st => {
      const tc = tc_colors[st.type] || '#94a3b8';
      return `<div class="strat-card" style="border-left-color:${tc}">
        <div style="display:flex;justify-content:space-between">
          <span class="strat-name" style="color:${tc}">${escHtml(st.name)}</span>
          <span class="strat-conf">${escHtml(st.conf)}</span>
        </div>
        <div class="strat-why">${escHtml(st.why)}</div>
        <div class="strat-entry"><span class="strat-entry-label">Entry: </span>${escHtml(st.entry)}</div>
      </div>`;
    }).join('');

    const news = (s.news || []).map(n =>
      `<div class="news-card">
         <a class="news-link" href="${escHtml(n.link)}" target="_blank" rel="noopener">${escHtml(n.title)}</a>
         <div class="news-pub">${escHtml(n.publisher)}</div>
       </div>`
    ).join('');

    const pros = (r.pros||[]).map(p=>`<li>${escHtml(p)}</li>`).join('');
    const cons = (r.cons||[]).map(c=>`<li>${escHtml(c)}</li>`).join('');
    const risks = (r.risks||[]).map(ri=>`<li>${escHtml(ri)}</li>`).join('');
    const valCls = r.valuation==='undervalued'?'sig-buy':r.valuation==='overvalued'?'sig-sell':'sig-hold';
    const expanded = idx === 0 ? 'open' : '';
    const chevron = idx === 0 ? '▲' : '▼';

    return `<div class="signal-card">
      <div class="signal-card-header" onclick="toggleSignalCard(this)">
        <span class="signal-card-title">${escHtml(t)} — ${escHtml(s.name)}</span>
        <span class="signal-card-toggle">${chevron}</span>
      </div>
      <div class="signal-card-body ${expanded}">
        <div class="sig-recs">
          <div class="sig-rec-col"><div class="sig-rec-label">1 Month</div><div class="sig-${r.cls_1m}">${r.rec_1m}</div></div>
          <div class="sig-rec-col"><div class="sig-rec-label">3 Months</div><div class="sig-${r.cls_3m}">${r.rec_3m}</div></div>
          <div class="sig-rec-col"><div class="sig-rec-label">6 Months</div><div class="sig-${r.cls_6m}">${r.rec_6m}</div></div>
        </div>
        <div class="pros-cons">
          <div>${pros?'<strong style="color:#b8cce0;font-size:.8em">Factors in favor:</strong><ul class="pros-list">'+pros+'</ul>':''}</div>
          <div>${cons?'<strong style="color:#b8cce0;font-size:.8em">Factors against:</strong><ul class="cons-list">'+cons+'</ul>':''}</div>
        </div>
        ${risks?'<strong style="color:#b8cce0;font-size:.8em">Key risks:</strong><ul class="risks-list">'+risks+'</ul>':''}
        <div style="margin:8px 0"><span class="${valCls}">Valuation: ${escHtml(r.valuation.toUpperCase())}</span></div>
        ${strats?'<strong style="color:#b8cce0;font-size:.8em;display:block;margin:10px 0 4px">Options Strategy Recommendations:</strong>'+strats:''}
        ${news?'<strong style="color:#b8cce0;font-size:.8em;display:block;margin:10px 0 4px">Recent News:</strong>'+news:''}
      </div>
    </div>`;
  }).join('');
}

function toggleSignalCard(header) {
  const body = header.nextElementSibling;
  const tog  = header.querySelector('.signal-card-toggle');
  if (body.classList.contains('open')) {
    body.classList.remove('open'); tog.textContent = '▼';
  } else {
    body.classList.add('open'); tog.textContent = '▲';
  }
}

// ── Benchmark tab ─────────────────────────────────────────
function renderBenchTab(charts, benchSummary, meta) {
  document.getElementById('benchHeader').textContent =
    'Performance vs ' + meta.bench_name;
  renderPlotly('chartPerf', charts.perf);
  if (charts.rolling_beta) {
    document.getElementById('rollingBetaBox').style.display = 'block';
    renderPlotly('chartRollingBeta', charts.rolling_beta);
  }
  renderDictTable('benchTable', benchSummary, {
    'Ann. Return': pct, 'Ann. Volatility': pct, 'Max Drawdown': pct,
    'Sharpe Ratio': ratio, 'Beta': ratio, 'Alpha': ratio,
  });
}

// ── Correlation tab ───────────────────────────────────────
function renderCorrTab(charts, corrTable, valid, params) {
  if (charts.corr_heatmap) renderEChart('chartCorrHeatmap', charts.corr_heatmap);
  renderSymMatrix('corrTable', corrTable, v => v != null ? v.toFixed(3) : 'N/A');
  if (charts.rolling_corr) renderPlotly('chartRollingCorr', charts.rolling_corr);
  if (charts.pair_scatter) renderPlotly('chartPairScatter', charts.pair_scatter);

  // Populate pair selects
  const px = document.getElementById('pairX');
  const py = document.getElementById('pairY');
  px.innerHTML = py.innerHTML = valid.map(t => `<option value="${t}">${t}</option>`).join('');
  if (valid.length > 1) py.selectedIndex = 1;

  if (charts.scatter_matrix) {
    document.getElementById('scatterMatBox').style.display = 'block';
    renderPlotly('chartScatterMatrix', charts.scatter_matrix);
  }
}

async function loadPairScatter() {
  if (!_lastParams) return;
  const t1 = document.getElementById('pairX').value;
  const t2 = document.getElementById('pairY').value;
  if (t1 === t2) { alert('Select two different assets.'); return; }
  setLoading(true, 'Computing pair scatter...');
  try {
    const resp = await fetch('/api/pair_scatter', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ t1, t2, start: _lastParams.start, end: _lastParams.end, freq: _lastParams.freq }),
    });
    const data = await resp.json();
    if (data.ok && data.pair_scatter) renderPlotly('chartPairScatter', data.pair_scatter);
  } catch(e) { /* silent */ }
  setLoading(false);
}

// ── Covariance tab ────────────────────────────────────────
function renderCovTab(charts, covTable, mcWeights, meta) {
  if (charts.cov_heatmap) renderEChart('chartCovHeatmap', charts.cov_heatmap);
  renderSymMatrix('covTable', covTable, v => v != null ? v.toFixed(6) : 'N/A');
  if (charts.vol_bar) renderPlotly('chartVolBar', charts.vol_bar);
  if (charts.mc_frontier) {
    renderPlotly('chartMCFrontier', charts.mc_frontier);
    document.getElementById('mcExplain').style.display = 'block';
    renderMCWeights(mcWeights);
  }
}

function renderMCWeights(mc) {
  if (!mc || !mc.tickers) return;
  const el = document.getElementById('mcWeights');
  function weightCard(title, wdata, col) {
    const rows = mc.tickers.map((t,i) =>
      `<div class="mc-weight-row"><span>${t}</span><span>${(wdata.weights[i]*100).toFixed(1)}%</span></div>`
    ).join('');
    return `<div class="mc-weight-card">
      <div class="mc-weight-title" style="color:${col}">${title}</div>
      ${rows}
      <div class="mc-caption">Vol: ${wdata.vol}% | Ret: ${wdata.ret}% | Sharpe: ${wdata.sharpe}</div>
    </div>`;
  }
  el.innerHTML = weightCard('⭐ Max Sharpe Portfolio', mc.max_sh, '#00e87a') +
                 weightCard('◆ Min Volatility Portfolio', mc.min_v, '#4da6ff');
}

// ── Risk Metrics tab ──────────────────────────────────────
function renderRiskTab(charts, riskTable, meta) {
  const confPct = meta.conf;
  const fmt = {
    'Ann. Return': pct, 'Ann. Volatility': pct, 'Max Drawdown': pct,
    [`VaR ${confPct}%`]: pct, [`CVaR ${confPct}%`]: pct,
    'Sharpe Ratio': ratio, 'Sortino Ratio': ratio, 'Beta': ratio,
    'Alpha': ratio, 'Treynor': ratio, 'Calmar': ratio, 'Info. Ratio': ratio,
  };
  renderDictTable('riskTable', riskTable, fmt);
  if (charts.rr_scatter) renderPlotly('chartRRScatter', charts.rr_scatter);
  if (charts.drawdown)   renderPlotly('chartDrawdown', charts.drawdown);
  if (charts.vix_chart)  renderPlotly('chartVIX', charts.vix_chart);
}

// ── Simulator tab ─────────────────────────────────────────
function renderSimulatorSelector(valid, sel) {
  const el = document.getElementById('simAssetSel');
  el.innerHTML = valid.map(t =>
    `<option value="${t}" ${t===sel?'selected':''}>${t}</option>`
  ).join('');
}

function renderSimulator(sim) {
  const el = document.getElementById('simContent');
  if (!sim) { el.innerHTML = '<div class="warn-box">No data available.</div>'; return; }
  if (sim.error) { el.innerHTML = `<div class="warn-box">${escHtml(sim.error)}</div>`; return; }

  const header = `
    <div class="kpi-card" style="margin-bottom:12px">
      <h3 style="color:#ffd700;margin:0;font-size:1em">${escHtml(sim.name)}</h3>
      <div style="color:#7090b0;font-size:.78em">${escHtml(sim.sector||'')} · ${escHtml(sim.industry||'')} · ${escHtml(sim.country||'')}</div>
    </div>`;

  const funds = `
    <div class="sec-hdr">Key Fundamentals</div>
    <div class="fund-grid">
      ${(sim.fund_map||[]).map(f =>
        `<div class="fund-card">
           <div class="fund-label">${escHtml(f.label)}</div>
           <div class="fund-value">${escHtml(f.value)}</div>
         </div>`
      ).join('')}
    </div>`;

  let dcfHtml = '';
  if (sim.dcf) {
    const d = sim.dcf;
    const upSign = d.upside >= 0 ? '+' : '';
    dcfHtml = `
      <hr style="border-color:#1e3550;margin:14px 0">
      <div class="sec-hdr">Discounted Cash Flow Valuation</div>
      <div class="dcf-metrics">
        <div class="dcf-metric">
          <div class="dcf-label">DCF Intrinsic Value</div>
          <div class="dcf-value">$${d.iv.toFixed(2)}</div>
        </div>
        <div class="dcf-metric">
          <div class="dcf-label">Current Price</div>
          <div class="dcf-value">$${d.price.toFixed(2)}</div>
        </div>
        <div class="dcf-metric">
          <div class="dcf-label">Upside / Downside</div>
          <div class="dcf-value" style="color:${d.upside>=0?'#00e87a':'#ff4545'}">${upSign}${d.upside.toFixed(1)}%</div>
        </div>
      </div>
      <div style="margin-bottom:12px"><span class="sig-${d.cls}">${escHtml(d.tag)}</span></div>`;
  }

  const epsChart = sim.eps_chart
    ? `<div class="sec-hdr" style="margin-top:14px">EPS History</div>
       <div class="chart-box"><div class="plotly-container" id="chartEPS" style="height:300px"></div></div>`
    : '';

  el.innerHTML = header + funds + dcfHtml + epsChart;

  if (sim.eps_chart) renderPlotly('chartEPS', sim.eps_chart);
}

async function loadSimulator(ticker) {
  if (!_lastParams) return;
  setLoading(true, `Loading ${ticker} fundamentals...`);
  try {
    const resp = await fetch('/api/simulator', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ticker,
        start: _lastParams.start, end: _lastParams.end, freq: _lastParams.freq,
        dcf_g: _lastParams.dcf_g, dcf_wacc: _lastParams.dcf_wacc,
        dcf_tg: _lastParams.dcf_tg, dcf_yrs: _lastParams.dcf_yrs,
      }),
    });
    const data = await resp.json();
    if (data.ok) renderSimulator(data.simulator);
  } catch(e) { /* silent */ }
  setLoading(false);
}

// ── Export tab ────────────────────────────────────────────
function renderExportTab(meta, params) {
  // nothing dynamic needed — button already in HTML
}

async function downloadExcel() {
  if (!_lastParams) return;
  const btn = document.getElementById('downloadBtn');
  btn.disabled = true; btn.textContent = '⏳ Generating...';
  try {
    const resp = await fetch('/api/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(_lastParams),
    });
    if (!resp.ok) { throw new Error('Download failed'); }
    const blob = await resp.blob();
    const cd   = resp.headers.get('Content-Disposition') || '';
    const fnMatch = cd.match(/filename="?([^"]+)"?/);
    const filename = fnMatch ? fnMatch[1] : 'analysis.xlsx';
    const url = URL.createObjectURL(blob);
    const a   = document.createElement('a');
    a.href = url; a.download = filename; document.body.appendChild(a);
    a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
  } catch(e) {
    alert('Download error: ' + e.message);
  }
  btn.disabled = false; btn.textContent = '📥 Build & Download Excel Report';
}

// ── Plotly rendering ──────────────────────────────────────
function renderPlotly(domId, figJson) {
  const el = document.getElementById(domId);
  if (!el || !figJson) return;
  try {
    const fig = typeof figJson === 'string' ? JSON.parse(figJson) : figJson;
    Plotly.react(el, fig.data, fig.layout, { responsive: true, displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d','select2d'], displaylogo: false });
  } catch(e) { console.error('Plotly error', domId, e); }
}

// ── Table helpers ─────────────────────────────────────────
function pct(v)   { return v != null ? (v*100).toFixed(2)+'%' : 'N/A'; }
function ratio(v) { return v != null ? parseFloat(v).toFixed(3) : 'N/A'; }

function renderDictTable(domId, dict, fmtMap) {
  const el = document.getElementById(domId);
  if (!el || !dict || !Object.keys(dict).length) { el.innerHTML=''; return; }
  const tickers = Object.keys(dict);
  const cols    = Object.keys(dict[tickers[0]] || {});
  const hdr = `<thead><tr><th>Ticker</th>${cols.map(c=>`<th>${escHtml(c)}</th>`).join('')}</tr></thead>`;
  const rows = tickers.map(t => {
    const row = dict[t] || {};
    return `<tr><td style="color:#ffd700;font-family:'Share Tech Mono',monospace">${escHtml(t)}</td>` +
      cols.map(c => {
        const v   = row[c];
        const fmt = fmtMap && fmtMap[c];
        const str = fmt ? fmt(v) : (v != null ? v : 'N/A');
        return `<td>${escHtml(String(str))}</td>`;
      }).join('') + '</tr>';
  }).join('');
  el.innerHTML = `<table>${hdr}<tbody>${rows}</tbody></table>`;
}

function renderSymMatrix(domId, dict, fmtFn) {
  const el = document.getElementById(domId);
  if (!el || !dict || !Object.keys(dict).length) { el.innerHTML=''; return; }
  const keys = Object.keys(dict);
  const hdr  = `<thead><tr><th></th>${keys.map(k=>`<th>${escHtml(k)}</th>`).join('')}</tr></thead>`;
  const rows = keys.map(r =>
    `<tr><td style="color:#ffd700;font-family:'Share Tech Mono',monospace">${escHtml(r)}</td>` +
    keys.map(c => {
      const v = dict[r] && dict[r][c] != null ? dict[r][c] : null;
      return `<td>${escHtml(fmtFn(v))}</td>`;
    }).join('') + '</tr>'
  ).join('');
  el.innerHTML = `<table>${hdr}<tbody>${rows}</tbody></table>`;
}

// ── Utilities ─────────────────────────────────────────────
function escHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function fmtNum(n) {
  if (n == null) return 'N/A';
  return parseFloat(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
