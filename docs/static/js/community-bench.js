/* OpenFlowLM Community Benchmarks Frontend: Leaderboard, ECharts Analytics, Detail & Submit
 * Preserves the Cloudflare Worker API contract (/api/benchmarks, /api/benchmarks/:id, /api/token)
 * and falls back cleanly to verified hardware seed & community runs when /api is unreachable on static hosts.
 */
(async function () {
  'use strict';

  const API = document.documentElement.dataset.cbApi || '/api';
  const pageEl = document.querySelector('[data-cb-page]');
  if (!pageEl) return;
  const kind = pageEl.dataset.cbPage;

  const $ = function (s) { return document.querySelector(s); };
  const esc = function (s) {
    return String(s ?? '').replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  };

  // Verified hardware runs (community Framework 13 Ryzen AI 7 350 w/ Radeon 860M + repo HX 370 sweeps)
  const SEED_RUNS = [
    {
      id: 'seed-qwen35-2b-fw13',
      model: 'qwen3.5:2b',
      task: 'LLM',
      hardware: 'Framework Laptop 13 (AMD Ryzen AI 7 350 w/ Radeon 860M)',
      cpu: 'AMD Ryzen AI 7 350 w/ Radeon 860M',
      ram_gb: 32,
      ram_spec: '32 GB DDR5-5600',
      backend: 'open_qwen35',
      quantization: 'Q4_K_M',
      oflm_version: '0.1.0',
      os: 'Linux (Performance power mode)',
      runs: 3,
      status: 'validated',
      source: 'community',
      date: '2026-11-18',
      headline: { ctx_k: 1, prefill: 366.24, decode: 20.79, ttft_ms: 3115 },
      stages: [
        { ctx_k: 1, ttft_avg_s: 3.115, ttft_std_s: 0.258, prefill_avg: 366.24, prefill_std: 30.78, decode_avg: 20.79, decode_std: 0.07 },
        { ctx_k: 2, ttft_avg_s: 5.789, ttft_std_s: 0.182, prefill_avg: 363.34, prefill_std: 11.69, decode_avg: 19.40, decode_std: 0.06 },
        { ctx_k: 4, ttft_avg_s: 10.547, ttft_std_s: 0.124, prefill_avg: 383.07, prefill_std: 4.49, decode_avg: 16.99, decode_std: 0.12 },
        { ctx_k: 8, ttft_avg_s: 21.573, ttft_std_s: 0.301, prefill_avg: 366.97, prefill_std: 5.16, decode_avg: 13.68, decode_std: 0.02 },
        { ctx_k: 16, ttft_avg_s: 48.022, ttft_std_s: 1.450, prefill_avg: 326.47, prefill_std: 9.86, decode_avg: 9.78, decode_std: 0.02 },
        { ctx_k: 32, ttft_avg_s: 109.297, ttft_std_s: 2.097, prefill_avg: 285.03, prefill_std: 5.47, decode_avg: 6.21, decode_std: 0.06 }
      ]
    },
    {
      id: 'seed-minilm-hx370',
      model: 'all-minilm:l6-v2',
      task: 'Embedding',
      hardware: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      cpu: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      ram_gb: 32,
      backend: 'open_npue',
      quantization: 'BF16',
      oflm_version: '0.1.0',
      os: 'Linux (Performance power mode)',
      runs: 3,
      status: 'seed',
      source: 'verified_seed',
      date: '2026-09-13',
      headline: { ctx_k: 64, prefill: 28791, decode: 1305.0, ttft_ms: 49.1 },
      stages: [
        { ctx_k: 1, ttft_avg_s: 0.0093, ttft_std_s: 0.0006, prefill_avg: 3240, decode_avg: 108.0, decode_std: 7.2, looped_avg_s: 0.0092, speedup: 0.99 },
        { ctx_k: 2, ttft_avg_s: 0.0092, ttft_std_s: 0.0006, prefill_avg: 5448, decode_avg: 217.9, decode_std: 13.1, looped_avg_s: 0.0170, speedup: 1.84 },
        { ctx_k: 4, ttft_avg_s: 0.0092, ttft_std_s: 0.0000, prefill_avg: 10410, decode_avg: 433.8, decode_std: 1.9, looped_avg_s: 0.0322, speedup: 3.49 },
        { ctx_k: 8, ttft_avg_s: 0.0102, ttft_std_s: 0.0008, prefill_avg: 16790, decode_avg: 785.5, decode_std: 56.4, looped_avg_s: 0.0613, speedup: 5.99 },
        { ctx_k: 16, ttft_avg_s: 0.0188, ttft_std_s: 0.0003, prefill_avg: 18758, decode_avg: 850.2, decode_std: 13.6, looped_avg_s: 0.1250, speedup: 6.64 },
        { ctx_k: 32, ttft_avg_s: 0.0309, ttft_std_s: 0.0009, prefill_avg: 22850, decode_avg: 1035.7, decode_std: 29.5, looped_avg_s: 0.2497, speedup: 8.07 },
        { ctx_k: 64, ttft_avg_s: 0.0491, ttft_std_s: 0.0010, prefill_avg: 28791, decode_avg: 1305.0, decode_std: 25.2, looped_avg_s: 0.4928, speedup: 10.05 },
        { ctx_k: 128, ttft_avg_s: 0.1162, ttft_std_s: 0.0016, prefill_avg: 24314, decode_avg: 1102.1, decode_std: 15.8, looped_avg_s: 0.9911, speedup: 8.53 }
      ]
    },
    {
      id: 'seed-bge-small-hx370',
      model: 'bge-small:en-v1.5',
      task: 'Embedding',
      hardware: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      cpu: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      ram_gb: 32,
      backend: 'open_npue',
      quantization: 'BF16',
      oflm_version: '0.1.0',
      os: 'Linux (Performance power mode)',
      runs: 3,
      status: 'seed',
      source: 'verified_seed',
      date: '2026-09-13',
      headline: { ctx_k: 64, prefill: 12474, decode: 565.4, ttft_ms: 114.0 },
      stages: [
        { ctx_k: 1, ttft_avg_s: 0.0188, ttft_std_s: 0.0001, prefill_avg: 1598, decode_avg: 53.3, decode_std: 0.1, looped_avg_s: 0.0195, speedup: 1.04 },
        { ctx_k: 2, ttft_avg_s: 0.0193, ttft_std_s: 0.0005, prefill_avg: 2589, decode_avg: 103.6, decode_std: 2.9, looped_avg_s: 0.0381, speedup: 1.97 },
        { ctx_k: 4, ttft_avg_s: 0.0188, ttft_std_s: 0.0001, prefill_avg: 5100, decode_avg: 212.5, decode_std: 1.1, looped_avg_s: 0.0727, speedup: 3.86 },
        { ctx_k: 8, ttft_avg_s: 0.0212, ttft_std_s: 0.0013, prefill_avg: 8110, decode_avg: 379.4, decode_std: 22.7, looped_avg_s: 0.1403, speedup: 6.63 },
        { ctx_k: 16, ttft_avg_s: 0.0448, ttft_std_s: 0.0008, prefill_avg: 7876, decode_avg: 357.0, decode_std: 6.3, looped_avg_s: 0.2882, speedup: 6.43 },
        { ctx_k: 32, ttft_avg_s: 0.0809, ttft_std_s: 0.0021, prefill_avg: 8732, decode_avg: 395.8, decode_std: 10.1, looped_avg_s: 0.5660, speedup: 7.00 },
        { ctx_k: 64, ttft_avg_s: 0.1140, ttft_std_s: 0.0101, prefill_avg: 12474, decode_avg: 565.4, decode_std: 47.0, looped_avg_s: 1.1435, speedup: 10.03 },
        { ctx_k: 128, ttft_avg_s: 0.3134, ttft_std_s: 0.0179, prefill_avg: 9039, decode_avg: 409.7, decode_std: 22.5, looped_avg_s: 2.3192, speedup: 7.40 }
      ]
    },
    {
      id: 'seed-bge-base-hx370',
      model: 'bge-base:en-v1.5',
      task: 'Embedding',
      hardware: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      cpu: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      ram_gb: 32,
      backend: 'open_npue',
      quantization: 'BF16',
      oflm_version: '0.1.0',
      os: 'Linux (Performance power mode)',
      runs: 3,
      status: 'seed',
      source: 'verified_seed',
      date: '2026-09-13',
      headline: { ctx_k: 64, prefill: 6868, decode: 311.3, ttft_ms: 205.6 },
      stages: [
        { ctx_k: 1, ttft_avg_s: 0.0253, ttft_std_s: 0.0008, prefill_avg: 1186, decode_avg: 39.5, decode_std: 1.2, looped_avg_s: 0.0250, speedup: 0.99 },
        { ctx_k: 2, ttft_avg_s: 0.0258, ttft_std_s: 0.0010, prefill_avg: 1940, decode_avg: 77.6, decode_std: 3.0, looped_avg_s: 0.0490, speedup: 1.90 },
        { ctx_k: 4, ttft_avg_s: 0.0259, ttft_std_s: 0.0005, prefill_avg: 3712, decode_avg: 154.7, decode_std: 2.7, looped_avg_s: 0.1011, speedup: 3.91 },
        { ctx_k: 8, ttft_avg_s: 0.0314, ttft_std_s: 0.0008, prefill_avg: 5451, decode_avg: 255.0, decode_std: 6.9, looped_avg_s: 0.2018, speedup: 6.43 },
        { ctx_k: 16, ttft_avg_s: 0.0701, ttft_std_s: 0.0006, prefill_avg: 5037, decode_avg: 228.3, decode_std: 1.9, looped_avg_s: 0.3811, speedup: 5.44 },
        { ctx_k: 32, ttft_avg_s: 0.1330, ttft_std_s: 0.0013, prefill_avg: 5308, decode_avg: 240.6, decode_std: 2.3, looped_avg_s: 0.7628, speedup: 5.73 },
        { ctx_k: 64, ttft_avg_s: 0.2056, ttft_std_s: 0.0020, prefill_avg: 6868, decode_avg: 311.3, decode_std: 3.0, looped_avg_s: 1.5513, speedup: 7.55 },
        { ctx_k: 128, ttft_avg_s: 0.5059, ttft_std_s: 0.0018, prefill_avg: 5582, decode_avg: 253.0, decode_std: 0.9, looped_avg_s: 3.1064, speedup: 6.14 }
      ]
    },
    {
      id: 'seed-nomic-hx370',
      model: 'nomic-embed-text:v1.5',
      task: 'Embedding',
      hardware: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      cpu: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      ram_gb: 32,
      backend: 'open_npue',
      quantization: 'BF16',
      oflm_version: '0.1.0',
      os: 'Linux (Performance power mode)',
      runs: 3,
      status: 'seed',
      source: 'verified_seed',
      date: '2026-09-13',
      headline: { ctx_k: 64, prefill: 6228, decode: 239.0, ttft_ms: 268.2 },
      stages: [
        { ctx_k: 1, ttft_avg_s: 0.0338, ttft_std_s: 0.0003, prefill_avg: 1006, decode_avg: 29.6, decode_std: 0.3, looped_avg_s: 0.0344, speedup: 1.02 },
        { ctx_k: 2, ttft_avg_s: 0.0317, ttft_std_s: 0.0017, prefill_avg: 1836, decode_avg: 63.3, decode_std: 3.4, looped_avg_s: 0.0618, speedup: 1.95 },
        { ctx_k: 4, ttft_avg_s: 0.0306, ttft_std_s: 0.0022, prefill_avg: 3683, decode_avg: 131.5, decode_std: 9.8, looped_avg_s: 0.1259, speedup: 4.12 },
        { ctx_k: 8, ttft_avg_s: 0.0430, ttft_std_s: 0.0055, prefill_avg: 4794, decode_avg: 188.9, decode_std: 22.6, looped_avg_s: 0.2403, speedup: 5.59 },
        { ctx_k: 16, ttft_avg_s: 0.0905, ttft_std_s: 0.0095, prefill_avg: 4653, decode_avg: 178.5, decode_std: 17.5, looped_avg_s: 0.4861, speedup: 5.37 },
        { ctx_k: 32, ttft_avg_s: 0.1864, ttft_std_s: 0.0271, prefill_avg: 4563, decode_avg: 175.1, decode_std: 23.6, looped_avg_s: 0.9344, speedup: 5.01 },
        { ctx_k: 64, ttft_avg_s: 0.2682, ttft_std_s: 0.0106, prefill_avg: 6228, decode_avg: 239.0, decode_std: 9.5, looped_avg_s: 1.8608, speedup: 6.94 },
        { ctx_k: 128, ttft_avg_s: 0.6849, ttft_std_s: 0.0354, prefill_avg: 4885, decode_avg: 187.4, decode_std: 10.0, looped_avg_s: 3.8620, speedup: 5.64 }
      ]
    },
    {
      id: 'seed-gte-hx370',
      model: 'gte-multilingual:base',
      task: 'Embedding',
      hardware: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      cpu: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      ram_gb: 32,
      backend: 'open_npue',
      quantization: 'BF16',
      oflm_version: '0.1.0',
      os: 'Linux (Performance power mode)',
      runs: 3,
      status: 'seed',
      source: 'verified_seed',
      date: '2026-09-13',
      headline: { ctx_k: 64, prefill: 6241, decode: 231.1, ttft_ms: 276.9 },
      stages: [
        { ctx_k: 1, ttft_avg_s: 0.0357, ttft_std_s: 0.0002, prefill_avg: 981, decode_avg: 28.0, decode_std: 0.2, looped_avg_s: 0.0361, speedup: 1.01 },
        { ctx_k: 2, ttft_avg_s: 0.0357, ttft_std_s: 0.0001, prefill_avg: 1819, decode_avg: 56.0, decode_std: 0.1, looped_avg_s: 0.0676, speedup: 1.89 },
        { ctx_k: 4, ttft_avg_s: 0.0381, ttft_std_s: 0.0012, prefill_avg: 3364, decode_avg: 105.1, decode_std: 3.2, looped_avg_s: 0.1412, speedup: 3.71 },
        { ctx_k: 8, ttft_avg_s: 0.0492, ttft_std_s: 0.0084, prefill_avg: 4591, decode_avg: 167.0, decode_std: 25.4, looped_avg_s: 0.2686, speedup: 5.46 },
        { ctx_k: 16, ttft_avg_s: 0.1113, ttft_std_s: 0.0012, prefill_avg: 3880, decode_avg: 143.7, decode_std: 1.6, looped_avg_s: 0.5533, speedup: 4.97 },
        { ctx_k: 32, ttft_avg_s: 0.2149, ttft_std_s: 0.0016, prefill_avg: 4022, decode_avg: 148.9, decode_std: 1.1, looped_avg_s: 1.0995, speedup: 5.12 },
        { ctx_k: 64, ttft_avg_s: 0.2769, ttft_std_s: 0.0024, prefill_avg: 6241, decode_avg: 231.1, decode_std: 2.0, looped_avg_s: 2.2119, speedup: 7.99 },
        { ctx_k: 128, ttft_avg_s: 0.8161, ttft_std_s: 0.0078, prefill_avg: 4235, decode_avg: 156.8, decode_std: 1.5, looped_avg_s: 4.3621, speedup: 5.34 }
      ]
    },
    {
      id: 'seed-bge-large-hx370',
      model: 'bge-large:en-v1.5',
      task: 'Embedding',
      hardware: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      cpu: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      ram_gb: 32,
      backend: 'open_npue',
      quantization: 'BF16',
      oflm_version: '0.1.0',
      os: 'Linux (Performance power mode)',
      runs: 3,
      status: 'seed',
      source: 'verified_seed',
      date: '2026-09-13',
      headline: { ctx_k: 64, prefill: 1984, decode: 89.9, ttft_ms: 712.1 },
      stages: [
        { ctx_k: 1, ttft_avg_s: 0.0741, ttft_std_s: 0.0041, prefill_avg: 406, decode_avg: 13.5, decode_std: 0.7, looped_avg_s: 0.0725, speedup: 0.98 },
        { ctx_k: 2, ttft_avg_s: 0.0716, ttft_std_s: 0.0026, prefill_avg: 699, decode_avg: 28.0, decode_std: 1.0, looped_avg_s: 0.1407, speedup: 1.96 },
        { ctx_k: 4, ttft_avg_s: 0.0715, ttft_std_s: 0.0017, prefill_avg: 1343, decode_avg: 56.0, decode_std: 1.4, looped_avg_s: 0.2827, speedup: 3.95 },
        { ctx_k: 8, ttft_avg_s: 0.0978, ttft_std_s: 0.0021, prefill_avg: 1749, decode_avg: 81.8, decode_std: 1.8, looped_avg_s: 0.5660, speedup: 5.79 },
        { ctx_k: 16, ttft_avg_s: 0.2287, ttft_std_s: 0.0025, prefill_avg: 1543, decode_avg: 70.0, decode_std: 0.8, looped_avg_s: 1.1292, speedup: 4.94 },
        { ctx_k: 32, ttft_avg_s: 0.4468, ttft_std_s: 0.0021, prefill_avg: 1580, decode_avg: 71.6, decode_std: 0.3, looped_avg_s: 2.2371, speedup: 5.01 },
        { ctx_k: 64, ttft_avg_s: 0.7121, ttft_std_s: 0.0157, prefill_avg: 1984, decode_avg: 89.9, decode_std: 2.0, looped_avg_s: 4.5209, speedup: 6.35 },
        { ctx_k: 128, ttft_avg_s: 1.6505, ttft_std_s: 0.0096, prefill_avg: 1711, decode_avg: 77.6, decode_std: 0.5, looped_avg_s: 9.0313, speedup: 5.47 }
      ]
    },
    {
      id: 'seed-embed-gemma-hx370',
      model: 'embed-gemma:300m',
      task: 'Embedding',
      hardware: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      cpu: 'AMD Ryzen AI 9 HX 370 w/ Radeon 890M',
      ram_gb: 32,
      backend: 'open_embedding',
      quantization: 'Q8_0',
      oflm_version: '0.1.0',
      os: 'Linux (Performance power mode)',
      runs: 2,
      status: 'seed',
      source: 'verified_seed',
      date: '2026-09-13',
      headline: { ctx_k: 16, prefill: null, decode: 1.6, ttft_ms: 9937.8 },
      stages: [
        { ctx_k: 1, ttft_avg_s: 0.6992, ttft_std_s: 0.0509, prefill_avg: null, decode_avg: 1.4, decode_std: 0.1, looped_avg_s: 0.6882, speedup: 0.98 },
        { ctx_k: 2, ttft_avg_s: 1.3817, ttft_std_s: 0.0241, prefill_avg: null, decode_avg: 1.4, decode_std: 0.0, looped_avg_s: 1.3295, speedup: 0.96 },
        { ctx_k: 4, ttft_avg_s: 2.5648, ttft_std_s: 0.1413, prefill_avg: null, decode_avg: 1.6, decode_std: 0.1, looped_avg_s: 2.6754, speedup: 1.04 },
        { ctx_k: 8, ttft_avg_s: 5.1568, ttft_std_s: 0.0453, prefill_avg: null, decode_avg: 1.6, decode_std: 0.0, looped_avg_s: 4.9844, speedup: 0.97 },
        { ctx_k: 16, ttft_avg_s: 9.9378, ttft_std_s: 0.3594, prefill_avg: null, decode_avg: 1.6, decode_std: 0.1, looped_avg_s: 10.6397, speedup: 1.07 }
      ]
    }
  ];

  const HYBRID_TIPS = {
    open_qwen35: 'Token generation and projections run on the NPU; light recurrent state updates during long prompts run on the CPU.',
    open_qwen36: 'Attention and active expert layers run on the NPU; lightweight expert routing runs on the CPU.',
    open_qwen36_moe: 'Attention and active expert layers run on the NPU; lightweight expert routing runs on the CPU.',
    open_lfm2: 'Dense layers and attention run on the NPU; short-convolution helper steps during prompt reading run on the CPU.',
    open_embedding: 'Linear projections run on the NPU; attention and vector pooling run on the CPU.',
    open_embedgemma: 'Linear projections run on the NPU; attention and vector pooling run on the CPU.',
    open_whisper: 'All 32 audio encoder layers run on the NPU; the lightweight 4-layer text decoder runs on the CPU.'
  };

  function getTokens() {
    if (window.OFLMCharts && typeof window.OFLMCharts.getThemeTokens === 'function') {
      return window.OFLMCharts.getThemeTokens();
    }
    const dark = document.documentElement.hasAttribute('data-dark-mode');
    return {
      dark: dark,
      text: dark ? '#e2e6ef' : '#273043',
      muted: dark ? '#a8b0c2' : '#5c667a',
      grid: dark ? 'rgba(168, 176, 194, 0.14)' : 'rgba(39, 48, 67, 0.10)',
      axis: dark ? '#3c4257' : '#cbd5e1',
      tooltipBg: dark ? '#1e202b' : '#ffffff',
      tooltipBorder: dark ? '#323648' : '#dce3ee',
      primary: dark ? '#6d94ff' : '#2b55d6',
      secondary: dark ? '#34d399' : '#059669',
      amber: dark ? '#fbbf24' : '#d97706',
      palette: dark
        ? ['#6d94ff', '#34d399', '#fbbf24', '#f472b6', '#38bdf8', '#a78bfa', '#fb7185']
        : ['#2b55d6', '#059669', '#d97706', '#db2777', '#0284c7', '#7c3aed', '#e11d48']
    };
  }

  async function fetchJsonSafe(url) {
    try {
      const res = await fetch(url, { mode: 'cors' });
      if (!res || !res.ok) return null;
      const ct = res.headers.get('content-type') || '';
      if (!ct.includes('application/json')) return null;
      return await res.json();
    } catch (_err) {
      return null;
    }
  }

  function formatNum(v, digits) {
    if (v === null || v === undefined || v === '') return '-';
    const n = Number(v);
    if (Number.isNaN(n)) return String(v);
    return digits !== undefined ? n.toFixed(digits) : String(n);
  }

  function makeBadge(cls, icon, label, tip, trailIcon) {
    const iconEl = icon ? '<span class="material-icons" aria-hidden="true">' + esc(icon) + '</span>' : '';
    const trailEl = trailIcon ? '<span class="material-icons oflm-badge-info" aria-hidden="true">' + esc(trailIcon) + '</span>' : '';
    const tipAttrs = tip
      ? (' data-oflm-tip="' + esc(tip) + '"' + (trailIcon ? ' tabindex="0" role="note" aria-label="' + esc(label + ': ' + tip) + '"' : ''))
      : '';
    return '<span class="oflm-badge ' + esc(cls) + '"' + tipAttrs + '>' + iconEl + '<span>' + esc(label) + '</span>' + trailEl + '</span>';
  }

  function renderTaskBadge(task) {
    const t = String(task || '').trim();
    const low = t.toLowerCase();
    if (low === 'llm' || low === 'text') return makeBadge('llm', 'chat', 'Chat');
    if (low === 'embedding' || low === 'embed') return makeBadge('embed', 'hub', 'Embeddings');
    if (low === 'vlm' || low === 'vision') return makeBadge('vlm', 'visibility', 'Vision');
    if (low === 'asr' || low === 'audio') return makeBadge('asr', 'mic', 'Speech');
    if (low === 'image' || low === 'diffusion') return makeBadge('image', 'palette', 'Image');
    return makeBadge('neutral', 'category', t || 'Run');
  }

  function renderBackendBadge(backend) {
    const raw = String(backend || '').trim();
    const key = raw.replace(/\s*\(.*?\)\s*$/i, '').trim().toLowerCase();
    if (/legacy|_npu\b|q4nx/i.test(raw)) {
      return makeBadge('legacy', 'history', 'Legacy NPU');
    }
    if (HYBRID_TIPS[key]) {
      return makeBadge('hybrid', 'splitscreen', 'NPU + CPU', HYBRID_TIPS[key], 'info');
    }
    if (key === 'open_npue' || key === 'open_qwen2' || key === 'open_qwen3vl' || key === 'open_diffusion') {
      return makeBadge('open', 'memory', 'NPU');
    }
    return makeBadge('open', 'memory', raw.replace(/\s*\(Open NPU\)\s*$/i, '').trim() || 'NPU');
  }

  function renderQuantBadge(quant) {
    const raw = String(quant || '').trim();
    const shortQuant = raw.replace(/\s*\(BF16 NPU\)\s*$/i, '').replace(/\s*\/\s*BFP\d+\b/i, '').trim() || '-';
    if (/^(Q4_1|Q8_0)\b/i.test(shortQuant)) {
      return makeBadge('qbest mono', 'star', shortQuant, 'Recommended quantization');
    }
    if (/q4nx/i.test(shortQuant)) {
      return makeBadge('q4nx mono', 'history', shortQuant);
    }
    return makeBadge('quant mono', '', shortQuant, 'Compatible quantization');
  }

  function renderSourceBadge(source, status) {
    if (source === 'verified_seed' || status === 'seed') {
      return makeBadge('ok', 'verified', 'Reference');
    }
    if (status && String(status).toLowerCase() === 'pending') {
      return makeBadge('wip', 'schedule', 'Pending');
    }
    return makeBadge('llm', 'cloud_done', 'Community');
  }

  // ==================== LEADERBOARD ====================
  async function initLeaderboardPage() {
    let rawRows = [];
    let usingSeed = false;

    const apiQuery = new URLSearchParams({ page: '1', size: '100', status: 'validated', sort: 'decode' });
    const liveData = await fetchJsonSafe(API + '/benchmarks?' + apiQuery.toString());
    if (liveData && Array.isArray(liveData.results) && liveData.results.length > 0) {
      // Merge live community submissions with reference runs so all runs are represented
      const liveIds = new Set(liveData.results.map(function (r) { return String(r.id); }));
      rawRows = liveData.results.concat(SEED_RUNS.filter(function (sr) { return !liveIds.has(sr.id); }));
      if (window.OFLMCharts && typeof window.OFLMCharts.syncLiveRuns === 'function') {
        window.OFLMCharts.syncLiveRuns(rawRows);
      }
    } else {
      rawRows = SEED_RUNS.slice();
      usingSeed = true;
      const banner = $('#cb-seed-banner');
      if (banner) banner.classList.remove('d-none');
    }

    let currentRows = rawRows.slice();

    function updateStats(rows) {
      const elRuns = $('#cb-stat-runs');
      const elRunsNote = $('#cb-stat-runs-note');
      const elModels = $('#cb-stat-models');
      const elHw = $('#cb-stat-hw');
      const elTopLlm = $('#cb-stat-top-llm');
      const elTopLlmNote = $('#cb-stat-top-llm-note');

      if (elRuns) elRuns.textContent = String(rows.length);
      if (elRunsNote) elRunsNote.textContent = usingSeed ? 'Reference + community avg' : 'Reference & community runs';
      if (elModels) elModels.textContent = String(new Set(rows.map(function (r) { return r.model; })).size);
      if (elHw) elHw.textContent = String(new Set(rows.map(function (r) { return r.cpu || r.hardware; })).size);

      const llmRows = rows.filter(function (r) { return r.task === 'LLM'; });
      if (elTopLlm) {
        if (llmRows.length) {
          const best = llmRows.slice().sort(function (a, b) { return (b.headline?.decode || 0) - (a.headline?.decode || 0); })[0];
          elTopLlm.textContent = formatNum(best.headline?.decode, 2) + ' tok/s';
          if (elTopLlmNote) elTopLlmNote.textContent = best.model + ' · ' + (best.cpu || best.hardware);
        } else {
          elTopLlm.textContent = '-';
          if (elTopLlmNote) elTopLlmNote.textContent = 'No LLM filter match';
        }
      }
    }

    function applyFiltersAndRender() {
      const qModel = ($('#f-model')?.value || '').trim().toLowerCase();
      const qTask = $('#f-task')?.value || '';
      const qHw = ($('#f-hardware')?.value || '').trim().toLowerCase();
      const qBackend = ($('#f-backend')?.value || '').trim().toLowerCase();
      const qQuant = ($('#f-quant')?.value || '').trim().toLowerCase();
      const sortBy = $('#f-sort')?.value || 'decode';

      currentRows = rawRows.filter(function (r) {
        const okModel = !qModel || String(r.model || '').toLowerCase().includes(qModel);
        const okTask = !qTask || r.task === qTask;
        const okHw = !qHw || String(r.hardware || '').toLowerCase().includes(qHw) || String(r.cpu || '').toLowerCase().includes(qHw);
        const okBack = !qBackend || String(r.backend || '').toLowerCase().includes(qBackend);
        const okQuant = !qQuant || String(r.quantization || '').toLowerCase().includes(qQuant);
        return okModel && okTask && okHw && okBack && okQuant;
      });

      currentRows.sort(function (a, b) {
        if (sortBy === 'prefill') return (Number(b.headline?.prefill) || 0) - (Number(a.headline?.prefill) || 0);
        if (sortBy === 'date') return String(b.date || '').localeCompare(String(a.date || ''));
        return (Number(b.headline?.decode) || 0) - (Number(a.headline?.decode) || 0);
      });

      const tbody = $('#cb-rows');
      if (tbody) {
        if (!currentRows.length) {
          tbody.innerHTML = '<tr><td colspan="9">No benchmark runs match the current filters. Reset filters or <a href="submit/">submit a new run</a>.</td></tr>';
        } else {
          tbody.innerHTML = currentRows.map(function (r) {
            const isEmbed = r.task === 'Embedding';
            const prefillCell = r.headline?.prefill != null
              ? (isEmbed ? formatNum(r.headline.prefill, 0) + ' tok/s' : formatNum(r.headline.prefill, 2) + ' tok/s')
              : 'not reported';
            const decodeCell = r.headline?.decode != null
              ? (isEmbed ? formatNum(r.headline.decode, 1) + ' txt/s' : formatNum(r.headline.decode, 2) + ' tok/s')
              : '-';
            const ttftCell = r.headline?.ttft_ms != null
              ? formatNum(r.headline.ttft_ms, isEmbed ? 1 : 0) + ' ms'
              : '-';
            const stageTag = r.headline?.ctx_k != null
              ? (isEmbed ? 'n=' + r.headline.ctx_k : r.headline.ctx_k + 'k ctx')
              : '';
            const stageBadge = stageTag
              ? '<div class="mt-1">' + makeBadge('ctx mono', 'straighten', stageTag) + '</div>'
              : '';
            const srcBadge = renderSourceBadge(r.source, r.status);
            const ramLine = r.ram_spec || (r.ram_gb ? r.ram_gb + ' GB RAM' : '');

            return '<tr class="oflm-clickable-row" data-run-id="' + esc(r.id) + '">' +
              '<td><div class="fw-semibold"><code>' + esc(r.model) + '</code></div>' + stageBadge + '</td>' +
              '<td>' + renderTaskBadge(r.task) + '</td>' +
              '<td><div>' + esc(r.hardware) + '</div><div class="small" style="color: var(--oflm-muted);">' + esc(ramLine) + '</div></td>' +
              '<td>' + renderBackendBadge(r.backend) + '</td>' +
              '<td>' + renderQuantBadge(r.quantization) + '</td>' +
              '<td>' + esc(prefillCell) + '</td>' +
              '<td><strong>' + esc(decodeCell) + '</strong></td>' +
              '<td>' + esc(ttftCell) + '</td>' +
              '<td><div>' + srcBadge + '</div><div class="small mt-1" style="color: var(--oflm-muted);">' + esc(r.date) + ' · v' + esc(r.oflm_version) + '</div></td>' +
              '</tr>';
          }).join('');

          tbody.querySelectorAll('tr[data-run-id]').forEach(function (tr) {
            tr.addEventListener('click', function () {
              const base = window.location.pathname.endsWith('/') ? window.location.pathname : (window.location.pathname + '/');
              window.location.href = base + 'detail/?id=' + encodeURIComponent(tr.dataset.runId);
            });
          });
        }
      }

      const countEl = $('#cb-count');
      if (countEl) {
        countEl.textContent = 'Showing ' + currentRows.length + ' of ' + rawRows.length + ' runs';
      }

      updateStats(currentRows);
    }

    ['f-model', 'f-task', 'f-hardware', 'f-backend', 'f-quant', 'f-sort'].forEach(function (id) {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('input', applyFiltersAndRender);
        el.addEventListener('change', applyFiltersAndRender);
      }
    });

    const resetBtn = $('#cb-reset');
    if (resetBtn) {
      resetBtn.addEventListener('click', function () {
        ['f-model', 'f-task', 'f-hardware', 'f-backend', 'f-quant'].forEach(function (id) {
          const el = document.getElementById(id);
          if (el) el.value = '';
        });
        const sortEl = $('#f-sort');
        if (sortEl) sortEl.value = 'decode';
        applyFiltersAndRender();
      });
    }

    applyFiltersAndRender();
  }

  // ==================== RUN DETAIL ====================
  async function initDetailPage() {
    const params = new URLSearchParams(window.location.search);
    const requestedId = params.get('id') || window.location.hash.slice(1) || SEED_RUNS[0].id;
    const container = $('#cb-detail');
    if (!container) return;

    let detailChart = null;

    async function renderRunById(runId) {
      let d = await fetchJsonSafe(API + '/benchmarks/' + encodeURIComponent(runId));
      let fromFallback = false;
      if (!d || !d.model) {
        d = SEED_RUNS.find(function (r) { return r.id === runId; }) || null;
        fromFallback = Boolean(d);
      }

      const banner = $('#cb-detail-banner');
      const pickerWrap = $('#cb-detail-picker-wrap');
      const picker = $('#cb-detail-picker');

      if (fromFallback) {
        if (banner) banner.classList.remove('d-none');
        if (pickerWrap && picker) {
          pickerWrap.classList.remove('d-none');
          if (!picker.options.length) {
            SEED_RUNS.forEach(function (sr) {
              const opt = document.createElement('option');
              opt.value = sr.id;
              opt.textContent = sr.model + ' \u00b7 ' + sr.hardware + ' (' + sr.task + ')';
              picker.appendChild(opt);
            });
            picker.addEventListener('change', function () {
              const nextId = picker.value;
              window.history.replaceState(null, '', '?id=' + encodeURIComponent(nextId));
              renderRunById(nextId);
            });
          }
          picker.value = d.id;
        }
      } else if (banner) {
        banner.classList.add('d-none');
      }

      if (!d) {
        container.innerHTML = '<div class="alert alert-warning">Benchmark run <code>' + esc(runId) + '</code> was not found. Return to <a href="../">Benchmarks</a>.</div>';
        return;
      }

      const h = d.headline || {};
      const isEmbed = d.task === 'Embedding';
      const stages = d.stages || [];
      const badgeHtml = renderSourceBadge(d.source, d.status);
      const memDisplay = d.ram_spec || ((d.ram_gb ?? d.survey?.ram_gb) ? (d.ram_gb ?? d.survey?.ram_gb) + ' GB' : '-');

      const statsHtml = isEmbed
        ? '<div class="oflm-stat-grid">' +
            '<div class="oflm-stat-card"><div class="oflm-stat-label">Peak Texts Throughput</div><div class="oflm-stat-value">' + esc(formatNum(h.decode, 1)) + ' txt/s</div><div class="oflm-stat-note">At batch n=' + esc(h.ctx_k) + '</div></div>' +
            '<div class="oflm-stat-card"><div class="oflm-stat-label">Tokens Throughput</div><div class="oflm-stat-value">' + (h.prefill != null ? esc(formatNum(h.prefill, 0)) + ' tok/s' : 'not reported') + '</div><div class="oflm-stat-note">Batched encoder path</div></div>' +
            '<div class="oflm-stat-card"><div class="oflm-stat-label">Batched Call Latency</div><div class="oflm-stat-value">' + esc(formatNum(h.ttft_ms, 1)) + ' ms</div><div class="oflm-stat-note">Batch n=' + esc(h.ctx_k) + ' wall clock</div></div>' +
            '<div class="oflm-stat-card"><div class="oflm-stat-label">Provenance</div><div class="oflm-stat-value">' + badgeHtml + '</div><div class="oflm-stat-note">' + esc(d.date) + ' · OFLM v' + esc(d.oflm_version) + '</div></div>' +
          '</div>'
        : '<div class="oflm-stat-grid">' +
            '<div class="oflm-stat-card"><div class="oflm-stat-label">Headline Decode</div><div class="oflm-stat-value">' + esc(formatNum(h.decode, 2)) + ' tok/s</div><div class="oflm-stat-note">At ' + esc(h.ctx_k) + 'k context</div></div>' +
            '<div class="oflm-stat-card"><div class="oflm-stat-label">Headline Prefill</div><div class="oflm-stat-value">' + esc(formatNum(h.prefill, 2)) + ' tok/s</div><div class="oflm-stat-note">At ' + esc(h.ctx_k) + 'k context</div></div>' +
            '<div class="oflm-stat-card"><div class="oflm-stat-label">Time to First Token</div><div class="oflm-stat-value">' + esc(formatNum(h.ttft_ms, 0)) + ' ms</div><div class="oflm-stat-note">At ' + esc(h.ctx_k) + 'k context</div></div>' +
            '<div class="oflm-stat-card"><div class="oflm-stat-label">Provenance</div><div class="oflm-stat-value">' + badgeHtml + '</div><div class="oflm-stat-note">' + esc(d.date) + ' · OFLM v' + esc(d.oflm_version) + '</div></div>' +
          '</div>';

      const stageHeadHtml = isEmbed
        ? '<tr><th>Batch Stage</th><th>Texts (txt/s)</th><th>Tokens (tok/s)</th><th>Batched (s)</th><th>Looped (s)</th><th>Batch Speedup</th></tr>'
        : '<tr><th>Context Stage</th><th>Prefill (tok/s)</th><th>Decode (tok/s)</th><th>TTFT (s)</th></tr>';

      const stageRowsHtml = stages.map(function (s) {
        if (isEmbed) {
          const txtCell = s.decode_std != null ? (formatNum(s.decode_avg, 1) + ' ± ' + formatNum(s.decode_std, 1)) : formatNum(s.decode_avg, 1);
          const latCell = s.ttft_std_s != null ? (formatNum(s.ttft_avg_s, 4) + ' ± ' + formatNum(s.ttft_std_s, 4)) : formatNum(s.ttft_avg_s, 4);
          return '<tr>' +
            '<td><code>n=' + esc(s.ctx_k) + '</code></td>' +
            '<td><strong>' + esc(txtCell) + '</strong></td>' +
            '<td>' + esc(s.prefill_avg != null ? formatNum(s.prefill_avg, 0) : 'not reported') + '</td>' +
            '<td>' + esc(latCell) + '</td>' +
            '<td>' + esc(s.looped_avg_s != null ? formatNum(s.looped_avg_s, 4) : '-') + '</td>' +
            '<td>' + esc(s.speedup != null ? formatNum(s.speedup, 2) + 'x' : '-') + '</td>' +
          '</tr>';
        }
        const preCell = s.prefill_std != null ? (formatNum(s.prefill_avg, 2) + ' ± ' + formatNum(s.prefill_std, 2)) : formatNum(s.prefill_avg, 2);
        const decCell = s.decode_std != null ? (formatNum(s.decode_avg, 2) + ' ± ' + formatNum(s.decode_std, 2)) : formatNum(s.decode_avg, 2);
        const ttftCell = s.ttft_std_s != null ? (formatNum(s.ttft_avg_s, 3) + ' ± ' + formatNum(s.ttft_std_s, 3)) : formatNum(s.ttft_avg_s, 3);
        return '<tr>' +
          '<td><code>' + esc(s.ctx_k) + 'k</code></td>' +
          '<td>' + esc(preCell) + '</td>' +
          '<td><strong>' + esc(decCell) + '</strong></td>' +
          '<td>' + esc(ttftCell) + '</td>' +
        '</tr>';
      }).join('');

      const surveyRow = d.survey
        ? '<tr><th>Hardware Survey</th><td>Included (anonymous SHA-256 machine hash)</td></tr>'
        : '';

      const leftSubTitle = isEmbed
        ? 'Throughput <span class="oflm-subchart-unit">(texts/s · linear from 0, higher is better)</span>'
        : 'Prefill vs. Decode Throughput <span class="oflm-subchart-unit">(tok/s · log scale, higher is better)</span>';
      const rightSubTitle = isEmbed
        ? 'Batch Speedup vs. Single Text <span class="oflm-subchart-unit">(× looped baseline · linear from 0, higher is better)</span>'
        : 'Time to First Token <span class="oflm-subchart-unit">(seconds · linear from 0, lower is better)</span>';

      container.innerHTML =
        statsHtml +
        '<div class="oflm-chart-card">' +
          '<div class="oflm-chart-header">' +
            '<div>' +
              '<div class="oflm-chart-meta mb-1">' + renderTaskBadge(d.task) + renderBackendBadge(d.backend) + renderQuantBadge(d.quantization) + '</div>' +
              '<h3 class="oflm-chart-title">' + esc(d.model) + ' Stage Scaling</h3>' +
              '<p class="oflm-chart-subtitle">' + esc(d.hardware) + (d.ram_spec ? ' · ' + esc(d.ram_spec) : '') + '</p>' +
            '</div>' +
          '</div>' +
          '<div class="oflm-small-multiples">' +
            '<div class="oflm-subchart">' +
              '<div class="oflm-subchart-title">' + leftSubTitle + '</div>' +
              '<div id="cb-detail-chart-a" class="oflm-echarts-canvas" style="width: 100%; height: 310px;" aria-label="Primary stage scaling chart"></div>' +
            '</div>' +
            '<div class="oflm-subchart">' +
              '<div class="oflm-subchart-title">' + rightSubTitle + '</div>' +
              '<div id="cb-detail-chart-b" class="oflm-echarts-canvas" style="width: 100%; height: 310px;" aria-label="Secondary stage scaling chart"></div>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<h2 class="h4 fw-bold mt-4 mb-3">Per-' + (isEmbed ? 'Batch' : 'Context') + ' Stage Measurements</h2>' +
        '<div class="oflm-table-wrap"><table class="oflm-table"><thead>' + stageHeadHtml + '</thead><tbody>' + stageRowsHtml + '</tbody></table></div>' +
        '<h2 class="h4 fw-bold mt-4 mb-3">System &amp; Runtime Environment</h2>' +
        '<div class="oflm-table-wrap"><table class="oflm-table"><tbody>' +
          '<tr><th style="width: 220px;">Model Tag</th><td><code>' + esc(d.model) + '</code></td></tr>' +
          '<tr><th>Task</th><td>' + renderTaskBadge(d.task) + '</td></tr>' +
          '<tr><th>Hardware Platform</th><td>' + esc(d.hardware) + '</td></tr>' +
          '<tr><th>Processor &amp; Graphics</th><td>' + esc(d.cpu || d.survey?.cpu || '-') + '</td></tr>' +
          '<tr><th>System Memory</th><td>' + esc(memDisplay) + '</td></tr>' +
          '<tr><th>Acceleration</th><td>' + renderBackendBadge(d.backend) + '</td></tr>' +
          '<tr><th>Quantization</th><td>' + renderQuantBadge(d.quantization) + '</td></tr>' +
          '<tr><th>Operating System</th><td>' + esc(d.os || d.survey?.os || '-') + '</td></tr>' +
          '<tr><th>OpenFlowLM Version</th><td>v' + esc(d.oflm_version) + ' (' + esc(d.runs || 3) + ' iterations/stage)</td></tr>' +
          surveyRow +
        '</tbody></table></div>';

      if (window.echarts && stages.length) {
        const elA = document.getElementById('cb-detail-chart-a');
        const elB = document.getElementById('cb-detail-chart-b');
        if (detailChart && Array.isArray(detailChart)) {
          detailChart.forEach(function (c) { try { c.dispose(); } catch (_err) {} });
        } else if (detailChart) {
          try { detailChart.dispose(); } catch (_err) {}
        }
        const chartA = window.echarts.init(elA, null, { renderer: 'svg' });
        const chartB = window.echarts.init(elB, null, { renderer: 'svg' });
        detailChart = [chartA, chartB];

        const labels = stages.map(function (s) { return isEmbed ? ('n=' + s.ctx_k) : (s.ctx_k + 'k'); });

        const drawA = function () {
          const t = getTokens();
          if (isEmbed) {
            chartA.setOption({
              backgroundColor: 'transparent',
              grid: { left: 52, right: 20, top: 32, bottom: 42 },
              tooltip: { trigger: 'axis', backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 12.5 } },
              xAxis: { type: 'category', name: 'Batch Size', nameLocation: 'middle', nameGap: 25, nameTextStyle: { color: t.muted, fontWeight: 600, fontSize: 11.5 }, data: labels, axisLabel: { color: t.text, fontWeight: 600 }, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false } },
              yAxis: { type: 'value', min: 0, axisLabel: { color: t.muted, fontSize: 11 }, axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: t.grid } } },
              series: [{
                name: 'Texts / sec',
                type: 'bar',
                barMaxWidth: 34,
                itemStyle: { color: t.primary, borderRadius: [4, 4, 0, 0] },
                label: { show: true, position: 'top', distance: 5, color: t.text, fontSize: 10.5, fontWeight: 700, formatter: function (p) { return p.value >= 100 ? Math.round(p.value) + '' : Number(p.value).toFixed(1); } },
                data: stages.map(function (s) { return s.decode_avg; })
              }]
            }, true);
          } else {
            chartA.setOption({
              backgroundColor: 'transparent',
              grid: { left: 52, right: 20, top: 40, bottom: 42 },
              legend: { top: 2, left: 0, itemGap: 16, textStyle: { color: t.text, fontSize: 12 } },
              tooltip: { trigger: 'axis', backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 12.5 } },
              xAxis: { type: 'category', name: 'Context Length', nameLocation: 'middle', nameGap: 25, nameTextStyle: { color: t.muted, fontWeight: 600, fontSize: 11.5 }, data: labels, axisLabel: { color: t.text, fontWeight: 600 }, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false } },
              yAxis: { type: 'log', min: 4, axisLabel: { color: t.muted, fontSize: 11 }, axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: t.grid } } },
              series: [
                {
                  name: 'Prefill (tok/s)',
                  type: 'line',
                  smooth: 0.15,
                  symbol: 'circle',
                  symbolSize: 7,
                  itemStyle: { color: t.primary },
                  lineStyle: { width: 2.5, color: t.primary },
                  label: { show: true, position: 'top', distance: 6, color: t.primary, fontSize: 10.5, fontWeight: 700, formatter: function (p) { return Math.round(p.value) + ''; } },
                  data: stages.map(function (s) { return s.prefill_avg; })
                },
                {
                  name: 'Decode (tok/s)',
                  type: 'line',
                  smooth: 0.15,
                  symbol: 'diamond',
                  symbolSize: 8,
                  itemStyle: { color: t.secondary },
                  lineStyle: { width: 2.5, color: t.secondary },
                  label: { show: true, position: 'top', distance: 6, color: t.secondary, fontSize: 10.5, fontWeight: 700, formatter: function (p) { return Number(p.value).toFixed(1); } },
                  data: stages.map(function (s) { return s.decode_avg; })
                }
              ]
            }, true);
          }
        };

        const drawB = function () {
          const t = getTokens();
          if (isEmbed) {
            chartB.setOption({
              backgroundColor: 'transparent',
              grid: { left: 48, right: 20, top: 32, bottom: 42 },
              tooltip: { trigger: 'axis', backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 12.5 } },
              xAxis: { type: 'category', name: 'Batch Size', nameLocation: 'middle', nameGap: 25, nameTextStyle: { color: t.muted, fontWeight: 600, fontSize: 11.5 }, data: labels, axisLabel: { color: t.text, fontWeight: 600 }, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false } },
              yAxis: { type: 'value', min: 0, axisLabel: { color: t.muted, fontSize: 11, formatter: function (v) { return v + '×'; } }, axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: t.grid } } },
              series: [{
                name: 'Batch Speedup',
                type: 'line',
                smooth: 0.15,
                symbol: 'circle',
                symbolSize: 7,
                itemStyle: { color: t.secondary },
                lineStyle: { width: 2.5, color: t.secondary },
                label: { show: true, position: 'top', distance: 6, color: t.secondary, fontSize: 10.5, fontWeight: 700, formatter: function (p) { return Number(p.value).toFixed(1) + '×'; } },
                data: stages.map(function (s) { return s.speedup; })
              }]
            }, true);
          } else {
            chartB.setOption({
              backgroundColor: 'transparent',
              grid: { left: 52, right: 20, top: 32, bottom: 42 },
              tooltip: { trigger: 'axis', backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 12.5 } },
              xAxis: { type: 'category', name: 'Context Length', nameLocation: 'middle', nameGap: 25, nameTextStyle: { color: t.muted, fontWeight: 600, fontSize: 11.5 }, data: labels, axisLabel: { color: t.text, fontWeight: 600 }, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false } },
              yAxis: { type: 'value', min: 0, axisLabel: { color: t.muted, fontSize: 11, formatter: function (v) { return v + 's'; } }, axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: t.grid } } },
              series: [{
                name: 'TTFT (s)',
                type: 'bar',
                barMaxWidth: 34,
                itemStyle: { color: t.amber, borderRadius: [4, 4, 0, 0] },
                label: { show: true, position: 'top', distance: 5, color: t.text, fontSize: 10.5, fontWeight: 700, formatter: function (p) { return Number(p.value).toFixed(1) + 's'; } },
                data: stages.map(function (s) { return s.ttft_avg_s; })
              }]
            }, true);
          }
        };

        drawA();
        drawB();
        if (window.OFLMCharts && typeof window.OFLMCharts.register === 'function') {
          window.OFLMCharts.register(chartA, drawA);
          window.OFLMCharts.register(chartB, drawB);
        }
      }
    }

    await renderRunById(requestedId);
  }

  // ==================== SUBMIT FORM ====================
  async function initSubmitPage() {
    const form = $('#cb-submit');
    if (!form) return;

    const fileInput = $('#s-file');
    if (fileInput) {
      fileInput.addEventListener('change', async function () {
        const f = fileInput.files && fileInput.files[0];
        if (!f) return;
        const text = await f.text();
        const csvArea = $('#s-csv');
        if (csvArea) csvArea.value = text;
        if (f.name.startsWith('bench_embed_') && $('#s-task')) {
          $('#s-task').value = 'Embedding';
        }
      });
    }

    form.addEventListener('submit', async function (e) {
      e.preventDefault();
      const msg = $('#cb-msg');
      if (msg) msg.textContent = 'Requesting upload token...';

      const tokData = await fetchJsonSafe(API + '/token');
      const token = tokData?.token || '';
      if (!token) {
        if (msg) {
          msg.innerHTML = 'Community Benchmarks Worker API is unreachable on this static preview host. On <code>openflowlm.org</code> (or via <code>oflm bench --submit</code>), this form posts to the live Worker endpoint.';
        }
        return;
      }

      const body = {
        submission_id: crypto.randomUUID(),
        model: $('#s-model').value.trim(),
        task: $('#s-task').value,
        hardware: $('#s-hardware').value.trim(),
        cpu: $('#s-cpu').value.trim(),
        ram_gb: Number($('#s-ram').value) || null,
        backend: $('#s-backend').value.trim(),
        quantization: $('#s-quant').value.trim(),
        oflm_version: $('#s-version').value.trim(),
        os: $('#s-os').value.trim(),
        runs: Number($('#s-runs').value) || 0,
        csv: $('#s-csv').value
      };

      const res = await fetch(API + '/benchmarks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-OFLM-Token': token },
        body: JSON.stringify(body)
      }).catch(function () { return null; });

      if (!res) {
        if (msg) msg.textContent = 'Network error while posting benchmark submission.';
        return;
      }

      const out = await res.json().catch(function () { return {}; });
      if (res.ok) {
        const detailHref = '../detail/?id=' + encodeURIComponent(out.id ?? '');
        if (msg) {
          msg.innerHTML = 'Submitted run <code>#' + esc(out.id) + '</code> (' + esc(out.status) + '): <a href="' + esc(detailHref) + '">view submission detail</a>.';
        }
      } else if (msg) {
        msg.textContent = 'Submission rejected (' + res.status + '): ' + (out.error || 'invalid payload') + (out.details ? ' (' + out.details.join(', ') + ')' : '');
      }
    });
  }

  if (kind === 'leaderboard') await initLeaderboardPage();
  if (kind === 'detail') await initDetailPage();
  if (kind === 'submit') await initSubmitPage();
})();
