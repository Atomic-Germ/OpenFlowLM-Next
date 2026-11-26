/* OpenFlowLM Apache ECharts runtime with Light/Dark [data-dark-mode] sync.
 * Architecture:
 *  - Model + Quantization selector -> XDNA2 NPU min-max range (across RAM/power configs) + median dot
 *  - Linked NPU x Memory Config matrix (full view on /benchmarks/, fused compact view on homepage)
 *  - Side-by-side small multiples for Chat Context Scaling & Embedding Batch Scaling
 */
(function () {
    'use strict';

    function isDarkMode() {
        return document.documentElement.hasAttribute('data-dark-mode');
    }

    function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    function getThemeTokens() {
        const dark = isDarkMode();
        return {
            dark: dark,
            text: dark ? '#e2e6ef' : '#1e293b',
            muted: dark ? '#9aa4b8' : '#556075',
            grid: dark ? 'rgba(168, 176, 194, 0.12)' : 'rgba(30, 41, 59, 0.08)',
            axis: dark ? '#3c4257' : '#cbd5e1',
            tooltipBg: dark ? '#1e202b' : '#ffffff',
            tooltipBorder: dark ? '#323648' : '#dce3ee',
            primary: dark ? '#6d94ff' : '#2554d4',
            primaryBand: dark ? 'rgba(109, 148, 255, 0.22)' : 'rgba(37, 84, 212, 0.18)',
            secondary: dark ? '#34d399' : '#059669',
            secondaryBand: dark ? 'rgba(52, 211, 153, 0.22)' : 'rgba(5, 150, 105, 0.18)',
            amber: dark ? '#fbbf24' : '#d97706',
            amberBand: dark ? 'rgba(251, 191, 36, 0.20)' : 'rgba(217, 119, 6, 0.16)',
            capsuleFill: dark ? 'rgba(148, 163, 184, 0.26)' : 'rgba(100, 116, 139, 0.22)',
            capsuleStroke: dark ? 'rgba(148, 163, 184, 0.55)' : 'rgba(71, 85, 105, 0.45)',
            palette: dark
                ? ['#6d94ff', '#34d399', '#fbbf24', '#f472b6', '#38bdf8', '#c084fc', '#fb7185']
                : ['#2554d4', '#059669', '#d97706', '#db2777', '#0284c7', '#7c3aed', '#e11d48']
        };
    }

    function getModalityMeta(task) {
        const low = String(task || '').trim().toLowerCase();
        if (low === 'llm' || low === 'text' || low === 'chat') {
            return { key: 'llm', label: 'Chat', icon: 'chat', badgeClass: 'llm', order: 1 };
        }
        if (low === 'vlm' || low === 'vision') {
            return { key: 'vlm', label: 'Vision', icon: 'visibility', badgeClass: 'vlm', order: 2 };
        }
        if (low === 'embedding' || low === 'embed' || low === 'embeddings') {
            return { key: 'embed', label: 'Embeddings', icon: 'hub', badgeClass: 'embed', order: 3 };
        }
        if (low === 'asr' || low === 'audio' || low === 'speech') {
            return { key: 'asr', label: 'Speech', icon: 'mic', badgeClass: 'asr', order: 4 };
        }
        if (low === 'image' || low === 'diffusion') {
            return { key: 'image', label: 'Image', icon: 'palette', badgeClass: 'image', order: 5 };
        }
        return { key: low || 'other', label: task || 'Workload', icon: 'category', badgeClass: 'neutral', order: 9 };
    }

    function extractHardwarePlatform(run) {
        if (run && run.hardware_platform) return String(run.hardware_platform).trim();
        if (run && run.cpu) return String(run.cpu).trim();
        const raw = String((run && run.hardware) || 'Unknown XDNA2 NPU').trim();
        const ryzenMatch = raw.match(/(AMD\s+Ryzen\s+AI[^,)·]+)/i);
        if (ryzenMatch) return ryzenMatch[1].trim();
        return raw.split('·')[0].trim();
    }

    function shortNpuName(platform) {
        return String(platform || '')
            .replace(/^AMD\s+/i, '')
            .replace(/\s+w\/\s+Radeon\s+\S+/i, '')
            .trim();
    }

    function extractMemoryConfig(run) {
        if (run && run.memory_config) return String(run.memory_config).trim();
        const src = String((run && (run.ram_spec || run.hardware)) || '');
        const lp = src.match(/(LPDDR5X?-\d{4})/i);
        if (lp) return lp[1].toUpperCase() + ' (Unified)';
        const ddr = src.match(/(DDR5-\d{4})/i);
        if (ddr) return ddr[1].toUpperCase() + ' (Non-Unified)';
        return 'System RAM';
    }

    function extractPowerMode(run) {
        if (run && run.power_mode) return String(run.power_mode).trim();
        const osStr = String((run && run.os) || '');
        const m = osStr.match(/(Performance|Balanced|Power[- ]?Saver|Turbo)/i);
        return m ? m[1] : 'Performance';
    }

    const HYBRID_TIPS = {
        open_qwen35: 'Token generation and projections run on the NPU; light recurrent state updates during long prompts run on the CPU.',
        open_qwen36: 'Attention and active expert layers run on the NPU; lightweight expert routing runs on the CPU.',
        open_qwen36_moe: 'Attention and active expert layers run on the NPU; lightweight expert routing runs on the CPU.',
        open_lfm2: 'Dense layers and attention run on the NPU; short-convolution steps during prompt reading run on the CPU.',
        open_embedding: 'Linear projections run on the NPU; attention and vector pooling run on the CPU.',
        open_embedgemma: 'Linear projections run on the NPU; attention and vector pooling run on the CPU.',
        open_whisper: 'All 32 audio encoder layers run on the NPU; the 4-layer text decoder runs on the CPU.'
    };

    function resolveInferenceInfo(run) {
        if (run && run.inference_type) {
            const isHybrid = /cpu/i.test(run.inference_type);
            return {
                label: run.inference_type,
                isHybrid: isHybrid,
                tip: run.inference_tip || (isHybrid ? HYBRID_TIPS[String(run.backend || '').toLowerCase()] || '' : '')
            };
        }
        const key = String((run && run.backend) || '').replace(/\s*\(.*?\)\s*$/i, '').trim().toLowerCase();
        if (HYBRID_TIPS[key]) {
            return { label: 'NPU + CPU', isHybrid: true, tip: HYBRID_TIPS[key] };
        }
        return { label: 'NPU', isHybrid: false, tip: '' };
    }

    function renderInferenceBadge(info, interactiveTip) {
        if (!info) return '';
        if (info.isHybrid) {
            const tipAttrs = (interactiveTip && info.tip)
                ? (' data-oflm-tip="' + esc(info.tip) + '" tabindex="0" role="note" aria-label="' + esc(info.label + ': ' + info.tip) + '"')
                : '';
            const infoIcon = (interactiveTip && info.tip)
                ? '<span class="material-icons oflm-badge-info" aria-hidden="true">info</span>'
                : '';
            return '<span class="oflm-badge hybrid"' + tipAttrs + '><span class="material-icons" aria-hidden="true">splitscreen</span><span>' + esc(info.label) + '</span>' + infoIcon + '</span>';
        }
        return '<span class="oflm-badge open"><span class="material-icons" aria-hidden="true">memory</span><span>' + esc(info.label) + '</span></span>';
    }

    function inferMinRamGb(run) {
        if (run && run.min_ram_gb) return Number(run.min_ram_gb);
        const mod = getModalityMeta(run && run.task);
        if (mod.key === 'embed') return 4;
        const m = String((run && run.model) || '').match(/:(\d+(?:\.\d+)?)b\b/i);
        if (m) {
            const b = parseFloat(m[1]);
            if (b <= 3) return 8;
            if (b <= 10) return 16;
            if (b <= 20) return 32;
            return 48;
        }
        return 16;
    }

    function fmtNum(val, digits) {
        if (val == null || Number.isNaN(Number(val))) return '-';
        const n = Number(val);
        if (digits === 0) return Math.round(n).toLocaleString('en-US');
        return n.toLocaleString('en-US', {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits
        });
    }

    function fmtCompactRate(val) {
        const n = Number(val);
        if (Number.isNaN(n)) return '';
        if (n >= 1000) return Math.round(n).toLocaleString('en-US');
        if (n >= 100) return n.toFixed(1);
        return n.toFixed(1);
    }

    function medianOf(nums) {
        if (!nums.length) return null;
        const s = nums.slice().sort(function (a, b) { return a - b; });
        const mid = Math.floor(s.length / 2);
        return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
    }

    const activeInstances = [];
    const allRunsUpdaters = [];

    function registerInstance(chart, renderFn) {
        activeInstances.push({ chart: chart, render: renderFn });
    }

    function disposeInstances(list) {
        (list || []).forEach(function (entry) {
            const idx = activeInstances.indexOf(entry);
            if (idx !== -1) activeInstances.splice(idx, 1);
            try { entry.chart.dispose(); } catch (_err) { /* noop */ }
        });
    }

    // ==================== UNCERTAINTY HELPERS ====================

    function makeBandPolygonSeries(name, points, colorFill, minClamp) {
        const floor = minClamp !== undefined ? minClamp : 0;
        return {
            name: name + ' ±1σ band',
            type: 'custom',
            silent: true,
            z: 1,
            tooltip: { show: false },
            renderItem: function (params, api) {
                if (params.dataIndex !== 0) return null;
                const upper = [];
                const lower = [];
                for (let i = 0; i < points.length; i++) {
                    upper.push(api.coord([i, Math.max(floor, points[i].mean + points[i].std)]));
                    lower.unshift(api.coord([i, Math.max(floor, points[i].mean - points[i].std)]));
                }
                return {
                    type: 'polygon',
                    shape: { points: upper.concat(lower) },
                    style: { fill: colorFill, stroke: 'none' }
                };
            },
            data: points.map(function (p, idx) { return [idx, Math.max(floor, p.mean)]; })
        };
    }

    function makeWhiskerSeries(name, points, strokeColor, minClamp) {
        const floor = minClamp !== undefined ? minClamp : 0;
        return {
            name: name + ' ±1σ',
            type: 'custom',
            silent: true,
            z: 4,
            tooltip: { show: false },
            renderItem: function (params, api) {
                const idx = api.value(0);
                const low = api.value(1);
                const high = api.value(2);
                const highPt = api.coord([idx, high]);
                const lowPt = api.coord([idx, low]);
                const cap = 4;
                return {
                    type: 'group',
                    children: [
                        {
                            type: 'line',
                            shape: { x1: highPt[0], y1: highPt[1], x2: lowPt[0], y2: lowPt[1] },
                            style: { stroke: strokeColor, lineWidth: 1.4 }
                        },
                        {
                            type: 'line',
                            shape: { x1: highPt[0] - cap, y1: highPt[1], x2: highPt[0] + cap, y2: highPt[1] },
                            style: { stroke: strokeColor, lineWidth: 1.4 }
                        },
                        {
                            type: 'line',
                            shape: { x1: lowPt[0] - cap, y1: lowPt[1], x2: lowPt[0] + cap, y2: lowPt[1] },
                            style: { stroke: strokeColor, lineWidth: 1.4 }
                        }
                    ]
                };
            },
            data: points.map(function (p, idx) {
                return [idx, Math.max(floor, p.mean - p.std), Math.max(floor, p.mean + p.std)];
            })
        };
    }

    // ==================== NPU-CENTRIC MODEL + QUANT -> RANGE & MATRIX ====================

    function indexRunsByModelAndQuant(runs) {
        const modelsMap = new Map();
        (runs || []).forEach(function (r) {
            const tag = String(r.model || '').trim();
            if (!tag) return;
            const mod = getModalityMeta(r.task);
            const quant = String(r.quant || 'Default').trim();
            const npuFull = extractHardwarePlatform(r);
            const npuShort = shortNpuName(npuFull);
            const memCfg = extractMemoryConfig(r);
            const power = extractPowerMode(r);
            const isEmbed = mod.key === 'embed';

            const p1 = r.output_tps != null ? Number(r.output_tps) : null;
            const p2 = isEmbed
                ? (r.single_tps != null ? Number(r.single_tps) : null)
                : (r.token_tps != null ? Number(r.token_tps) : null);

            if (!modelsMap.has(tag)) {
                modelsMap.set(tag, {
                    tag: tag,
                    task: r.task,
                    modality: mod,
                    inferenceInfo: resolveInferenceInfo(r),
                    minRamGb: inferMinRamGb(r),
                    unit: r.output_unit || (isEmbed ? 'txt/s' : 'tok/s'),
                    p1Label: isEmbed ? 'Peak Batched Throughput' : 'Token Decode',
                    p1Sub: isEmbed ? 'txt/s · batched NPU (higher is faster)' : 'tok/s · 1k ctx (memory-bandwidth bound)',
                    p2Label: isEmbed ? 'Single-Text Throughput (n=1)' : 'Prompt Prefill',
                    p2Sub: isEmbed ? 'txt/s · low-latency path (higher is faster)' : 'tok/s · 1k ctx (NPU-compute bound)',
                    quants: new Map()
                });
            }
            const mEntry = modelsMap.get(tag);
            if (!mEntry.quants.has(quant)) {
                mEntry.quants.set(quant, []);
            }
            mEntry.quants.get(quant).push({
                npuFull: npuFull,
                npuShort: npuShort,
                memCfg: memCfg,
                power: power,
                stage: r.stage || '',
                runs: Number(r.runs || 1),
                isCommunity: r.source !== 'verified_seed' && r.status !== 'seed',
                p1: p1,
                p2: p2,
                tokenTps: r.token_tps != null ? Number(r.token_tps) : null,
                latencyMs: r.latency_ms != null ? Number(r.latency_ms) : null
            });
        });

        return Array.from(modelsMap.values()).sort(function (a, b) {
            if (a.modality.order !== b.modality.order) return a.modality.order - b.modality.order;
            return a.tag.localeCompare(b.tag);
        });
    }

    function buildNpuSummaries(entries) {
        const memSet = new Set();
        const byNpu = new Map();

        entries.forEach(function (e) {
            memSet.add(e.memCfg);
            if (!byNpu.has(e.npuShort)) {
                byNpu.set(e.npuShort, {
                    npuShort: e.npuShort,
                    npuFull: e.npuFull,
                    entries: [],
                    byMem: new Map()
                });
            }
            const grp = byNpu.get(e.npuShort);
            grp.entries.push(e);
            if (!grp.byMem.has(e.memCfg)) {
                grp.byMem.set(e.memCfg, []);
            }
            grp.byMem.get(e.memCfg).push(e);
        });

        function summarizeMetric(list, key) {
            const valid = list.filter(function (x) { return x[key] != null && x[key] > 0; });
            if (!valid.length) return null;
            valid.sort(function (a, b) { return a[key] - b[key]; });
            const vals = valid.map(function (x) { return x[key]; });
            const minEntry = valid[0];
            const maxEntry = valid[valid.length - 1];
            return {
                min: minEntry[key],
                max: maxEntry[key],
                median: medianOf(vals),
                minConfig: minEntry.memCfg + ' · ' + minEntry.power,
                maxConfig: maxEntry.memCfg + ' · ' + maxEntry.power,
                count: valid.length
            };
        }

        const npuRows = Array.from(byNpu.values()).map(function (grp) {
            const memSummaries = {};
            grp.byMem.forEach(function (list, memKey) {
                const totalRuns = list.reduce(function (acc, item) { return acc + item.runs; }, 0);
                memSummaries[memKey] = {
                    p1: summarizeMetric(list, 'p1'),
                    p2: summarizeMetric(list, 'p2'),
                    runs: totalRuns,
                    configs: list.length
                };
            });
            const totalRuns = grp.entries.reduce(function (acc, item) { return acc + item.runs; }, 0);
            return {
                npuShort: grp.npuShort,
                npuFull: grp.npuFull,
                totalRuns: totalRuns,
                memCount: grp.byMem.size,
                p1: summarizeMetric(grp.entries, 'p1'),
                p2: summarizeMetric(grp.entries, 'p2'),
                byMem: memSummaries
            };
        });

        // Sort ascending so ECharts yAxis category puts fastest NPU at top
        npuRows.sort(function (a, b) {
            return ((a.p1 && a.p1.median) || 0) - ((b.p1 && b.p1.median) || 0);
        });

        const memCols = Array.from(memSet).sort();
        return { npuRows: npuRows, memCols: memCols };
    }

    function buildNpuRangeOption(model, quant, npuRows, metricKey, hlMem, hlNpu) {
        const t = getThemeTokens();
        const unit = model.unit;
        const accentColor = metricKey === 'p1' ? t.primary : t.secondary;

        const validRows = npuRows.filter(function (row) { return row[metricKey] != null; });
        const yLabels = validRows.map(function (row) { return row.npuShort; });
        const maxVal = validRows.length
            ? Math.max.apply(null, validRows.map(function (r) { return r[metricKey].max; }))
            : 10;
        const xMax = Math.ceil(maxVal * 1.28);

        const rangeData = validRows.map(function (row, idx) {
            const stat = row[metricKey];
            return [idx, stat.min, stat.max, stat.median];
        });

        const medianData = validRows.map(function (row, idx) {
            const stat = row[metricKey];
            return {
                value: [stat.median, idx],
                row: row,
                stat: stat
            };
        });

        const highlightData = [];
        if (hlMem) {
            validRows.forEach(function (row, idx) {
                if (hlNpu && row.npuShort !== hlNpu) return;
                const cell = row.byMem[hlMem];
                if (cell && cell[metricKey]) {
                    highlightData.push({
                        value: [cell[metricKey].median, idx],
                        npuShort: row.npuShort,
                        memCfg: hlMem,
                        val: cell[metricKey].median
                    });
                }
            });
        }

        const series = [
            // Min-Max capsule bar across RAM speeds & power modes
            {
                name: 'Min–Max Range (RAM / Power)',
                type: 'custom',
                z: 2,
                renderItem: function (params, api) {
                    const yIdx = api.value(0);
                    const lo = api.value(1);
                    const hi = api.value(2);
                    const pLo = api.coord([lo, yIdx]);
                    const pHi = api.coord([hi, yIdx]);
                    const pZero = api.coord([0, yIdx]);
                    const barH = 14;

                    const children = [
                        // Subtle baseline rail from 0 to min
                        {
                            type: 'line',
                            shape: { x1: pZero[0], y1: pLo[1], x2: pLo[0], y2: pLo[1] },
                            style: { stroke: t.grid, lineWidth: 2 }
                        }
                    ];

                    if (hi > lo) {
                        const w = Math.max(8, pHi[0] - pLo[0]);
                        children.push({
                            type: 'rect',
                            shape: { x: pLo[0], y: pLo[1] - barH / 2, width: w, height: barH, r: barH / 2 },
                            style: { fill: t.capsuleFill, stroke: t.capsuleStroke, lineWidth: 1 }
                        });
                    }
                    return { type: 'group', children: children };
                },
                data: rangeData
            },
            // Median dot with direct numeric label
            {
                name: 'Median',
                type: 'scatter',
                symbol: 'circle',
                symbolSize: 12,
                z: 5,
                itemStyle: {
                    color: accentColor,
                    borderColor: t.tooltipBg,
                    borderWidth: 2
                },
                label: {
                    show: true,
                    position: 'right',
                    distance: 10,
                    color: t.text,
                    fontSize: 11.5,
                    fontWeight: 700,
                    fontFamily: 'JetBrains Mono, ui-monospace, monospace',
                    formatter: function (p) {
                        const stat = p.data.stat;
                        if (stat.max > stat.min) {
                            return fmtCompactRate(stat.median) + ' (' + fmtCompactRate(stat.min) + '–' + fmtCompactRate(stat.max) + ')';
                        }
                        return fmtCompactRate(stat.median) + ' ' + unit;
                    }
                },
                data: medianData
            }
        ];

        if (highlightData.length) {
            series.push({
                name: hlMem,
                type: 'scatter',
                symbol: 'diamond',
                symbolSize: 14,
                z: 7,
                itemStyle: {
                    color: t.amber,
                    borderColor: t.tooltipBg,
                    borderWidth: 2
                },
                data: highlightData
            });
        }

        return {
            backgroundColor: 'transparent',
            animationDuration: 260,
            grid: { left: 132, right: 120, top: 20, bottom: 38 },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'shadow' },
                backgroundColor: t.tooltipBg,
                borderColor: t.tooltipBorder,
                borderWidth: 1,
                textStyle: { color: t.text, fontSize: 12.5 },
                formatter: function (params) {
                    if (!params || !params.length) return '';
                    const idx = params[0].dataIndex;
                    const row = validRows[idx];
                    if (!row || !row[metricKey]) return '';
                    const stat = row[metricKey];
                    const lines = [
                        '<div style="font-weight:700;margin-bottom:3px;">' + esc(row.npuFull) + '</div>',
                        '<div style="font-size:11.5px;opacity:0.82;margin-bottom:5px;">' + esc(model.tag) + ' · ' + esc(quant) + ' · ' + row.totalRuns + ' run(s) across ' + row.memCount + ' memory config(s)</div>',
                        '<div>Median: <strong>' + fmtCompactRate(stat.median) + ' ' + esc(unit) + '</strong></div>'
                    ];
                    if (stat.max > stat.min) {
                        lines.push('<div>Min: <strong>' + fmtCompactRate(stat.min) + ' ' + esc(unit) + '</strong> <span style="opacity:0.8;">(' + esc(stat.minConfig) + ')</span></div>');
                        lines.push('<div>Max: <strong>' + fmtCompactRate(stat.max) + ' ' + esc(unit) + '</strong> <span style="opacity:0.8;">(' + esc(stat.maxConfig) + ')</span></div>');
                    } else {
                        lines.push('<div>Measured config: <strong>' + esc(stat.minConfig) + '</strong></div>');
                    }
                    if (hlMem && row.byMem[hlMem] && row.byMem[hlMem][metricKey]) {
                        lines.push('<div style="margin-top:4px;color:' + t.amber + ';">Selected (' + esc(hlMem) + '): <strong>' + fmtCompactRate(row.byMem[hlMem][metricKey].median) + ' ' + esc(unit) + '</strong></div>');
                    }
                    return lines.join('');
                }
            },
            xAxis: {
                type: 'value',
                min: 0,
                max: xMax,
                name: unit + ' (higher is faster)',
                nameLocation: 'middle',
                nameGap: 23,
                nameTextStyle: { color: t.muted, fontWeight: 600, fontSize: 11 },
                axisLabel: { color: t.muted, fontSize: 11 },
                axisLine: { show: false },
                axisTick: { show: false },
                splitLine: { lineStyle: { color: t.grid } }
            },
            yAxis: {
                type: 'category',
                data: yLabels,
                axisLabel: { color: t.text, fontWeight: 700, fontSize: 11.5 },
                axisLine: { lineStyle: { color: t.axis } },
                axisTick: { show: false }
            },
            series: series
        };
    }

    function renderCompactSparkbarSvg(row, maxVal, unit, t) {
        const stat = row.p1;
        if (!stat) return '<span style="color:var(--oflm-muted);">-</span>';
        const sw = 220;
        const sh = 24;
        const padL = 6;
        const padR = 76;
        const plotW = sw - padL - padR;
        const xScale = function (v) { return padL + (v / (maxVal * 1.08)) * plotW; };
        const xLo = xScale(stat.min);
        const xHi = xScale(stat.max);
        const xMed = xScale(stat.median);

        const tip = stat.max > stat.min
            ? (row.npuShort + ': Median ' + fmtCompactRate(stat.median) + ' ' + unit + ' | Min ' + fmtCompactRate(stat.min) + ' (' + stat.minConfig + ') | Max ' + fmtCompactRate(stat.max) + ' (' + stat.maxConfig + ')')
            : (row.npuShort + ': ' + fmtCompactRate(stat.median) + ' ' + unit + ' (' + stat.minConfig + ', n=' + row.totalRuns + ')');

        let inner = '<line x1="' + padL + '" y1="12" x2="' + (sw - padR) + '" y2="12" stroke="' + t.grid + '" stroke-width="2" />';
        if (stat.max > stat.min) {
            const w = Math.max(8, xHi - xLo);
            inner += '<rect x="' + xLo.toFixed(1) + '" y="6" width="' + w.toFixed(1) + '" height="12" rx="6" fill="' + t.capsuleFill + '" stroke="' + t.capsuleStroke + '" stroke-width="1" />';
        } else {
            inner += '<line x1="' + padL + '" y1="12" x2="' + xMed.toFixed(1) + '" y2="12" stroke="' + t.primaryBand + '" stroke-width="4" stroke-linecap="round" />';
        }
        inner += '<circle cx="' + xMed.toFixed(1) + '" cy="12" r="5" fill="' + t.primary + '" stroke="' + t.tooltipBg + '" stroke-width="1.5" />';
        const labelStr = stat.max > stat.min
            ? (fmtCompactRate(stat.min) + '–' + fmtCompactRate(stat.max))
            : (fmtCompactRate(stat.median) + ' ' + unit);
        inner += '<text x="' + (sw - padR + 6) + '" y="16" font-size="11" font-weight="700" fill="' + t.text + '" font-family="JetBrains Mono, ui-monospace, monospace">' + esc(labelStr) + '</text>';

        return '<span class="oflm-sparkbar-wrap" data-oflm-tip="' + esc(tip) + '" tabindex="0">' +
            '<svg viewBox="0 0 ' + sw + ' ' + sh + '" class="oflm-sparkbar-svg" role="img" aria-label="' + esc(tip) + '">' + inner + '</svg>' +
        '</span>';
    }

    function initNpuBenchWidget(container) {
        if (!window.echarts) return;
        const bodyEl = container.querySelector('.oflm-npu-bench-body');
        const payloadEl = container.querySelector('.oflm-chart-payload');
        if (!bodyEl || !payloadEl) return;

        let dataset;
        try {
            dataset = JSON.parse(payloadEl.textContent);
        } catch (_err) {
            return;
        }

        const viewMode = container.dataset.view || 'full'; // 'full' (Variation 1) or 'compact' (Variation 2)
        let activeModelTag = null;
        let activeQuant = null;
        let hlMem = null;
        let hlNpu = null;
        let ddOpen = false;
        let childCharts = [];

        function renderWidget() {
            disposeInstances(childCharts);
            childCharts = [];

            const models = indexRunsByModelAndQuant(dataset.runs || []);
            if (!models.length) {
                bodyEl.innerHTML = '<div class="oflm-chart-card"><p class="small mb-0" style="color: var(--oflm-muted);">No benchmark runs available.</p></div>';
                return;
            }

            if (!activeModelTag || !models.some(function (m) { return m.tag === activeModelTag; })) {
                activeModelTag = models[0].tag;
            }
            const model = models.find(function (m) { return m.tag === activeModelTag; });
            const quantKeys = Array.from(model.quants.keys());
            if (!activeQuant || !quantKeys.includes(activeQuant)) {
                activeQuant = quantKeys[0];
            }

            const entries = model.quants.get(activeQuant) || [];
            const summary = buildNpuSummaries(entries);
            const npuRows = summary.npuRows;
            const memCols = summary.memCols;
            const t = getThemeTokens();

            // Hierarchical dropdown: models with their quantizations nested underneath
            const menuGroupsHtml = models.map(function (m) {
                const qList = Array.from(m.quants.keys());
                const optsHtml = qList.map(function (q) {
                    const qEntries = m.quants.get(q) || [];
                    const npuSet = new Set(qEntries.map(function (e) { return e.npuShort; }));
                    const rCount = qEntries.reduce(function (acc, e) { return acc + e.runs; }, 0);
                    const isActive = m.tag === activeModelTag && q === activeQuant;
                    return '<button type="button" class="oflm-dd-option' + (isActive ? ' active' : '') + '" role="option" aria-selected="' + (isActive ? 'true' : 'false') + '" data-pick-model="' + esc(m.tag) + '" data-pick-quant="' + esc(q) + '">' +
                        '<span class="oflm-dd-opt-left">' +
                            '<span class="oflm-badge quant mono"><span>' + esc(q) + '</span></span>' +
                            '<span class="oflm-dd-opt-meta">' + npuSet.size + (npuSet.size === 1 ? ' NPU' : ' NPUs') + ' · ' + rCount + ' runs</span>' +
                        '</span>' +
                        (isActive ? '<span class="material-icons oflm-dd-check" aria-hidden="true">check</span>' : '') +
                    '</button>';
                }).join('');

                return '<div class="oflm-dd-group" role="group" aria-label="' + esc(m.tag) + '">' +
                    '<div class="oflm-dd-group-head">' +
                        '<code class="oflm-dd-model-name">' + esc(m.tag) + '</code>' +
                        '<span class="oflm-dd-group-badges">' +
                            '<span class="oflm-badge ' + esc(m.modality.badgeClass) + '"><span class="material-icons" aria-hidden="true">' + esc(m.modality.icon) + '</span><span>' + esc(m.modality.label) + '</span></span>' +
                            renderInferenceBadge(m.inferenceInfo, false) +
                        '</span>' +
                    '</div>' +
                    '<div class="oflm-dd-options">' + optsHtml + '</div>' +
                '</div>';
            }).join('');

            const totalRuns = npuRows.reduce(function (acc, r) { return acc + r.totalRuns; }, 0);

            const selectorCard =
                '<div class="oflm-bench-selector">' +
                    '<div class="oflm-bench-selector-row">' +
                        '<div class="oflm-dd' + (ddOpen ? ' is-open' : '') + '">' +
                            '<button type="button" class="oflm-dd-trigger" data-dd-toggle aria-haspopup="listbox" aria-expanded="' + (ddOpen ? 'true' : 'false') + '">' +
                                '<span class="oflm-dd-trigger-main">' +
                                    '<code class="oflm-dd-trigger-model">' + esc(model.tag) + '</code>' +
                                    '<span class="oflm-badge quant mono"><span>' + esc(activeQuant) + '</span></span>' +
                                    '<span class="oflm-badge ' + esc(model.modality.badgeClass) + '"><span class="material-icons" aria-hidden="true">' + esc(model.modality.icon) + '</span><span>' + esc(model.modality.label) + '</span></span>' +
                                    renderInferenceBadge(model.inferenceInfo, false) +
                                '</span>' +
                                '<span class="material-icons oflm-dd-chevron" aria-hidden="true">expand_more</span>' +
                            '</button>' +
                            '<div class="oflm-dd-menu" role="listbox" aria-label="Models and quantizations">' + menuGroupsHtml + '</div>' +
                        '</div>' +
                        '<div class="oflm-chart-meta">' +
                            renderInferenceBadge(model.inferenceInfo, true) +
                            '<span class="oflm-badge neutral"><span>Fits ≥ ' + esc(model.minRamGb) + ' GB RAM</span></span>' +
                            '<span class="small" style="color: var(--oflm-muted);">' + npuRows.length + (npuRows.length === 1 ? ' XDNA2 NPU' : ' XDNA2 NPUs') + ' · ' + memCols.length + (memCols.length === 1 ? ' memory config' : ' memory configs') + ' · ' + totalRuns + ' runs</span>' +
                        '</div>' +
                    '</div>' +
                '</div>';

            if (viewMode === 'compact') {
                // VARIATION 2 (Homepage): Ultra-compact fused NPU x Memory table with inline range + median sparkbar
                const maxP1 = npuRows.length
                    ? Math.max.apply(null, npuRows.map(function (r) { return (r.p1 && r.p1.max) || 1; }))
                    : 1;
                const displayRows = npuRows.slice().reverse(); // fastest at top

                const headerCols = memCols.map(function (mc) {
                    return '<th class="text-end">' + esc(mc) + '</th>';
                }).join('');

                const bodyRows = displayRows.map(function (row) {
                    const cells = memCols.map(function (mc) {
                        const cell = row.byMem[mc];
                        if (!cell || !cell.p1) {
                            return '<td class="text-end" style="color: var(--oflm-muted);">—</td>';
                        }
                        const p2Str = cell.p2 ? (' <span class="small" style="color: var(--oflm-muted);">/ ' + fmtCompactRate(cell.p2.median) + '</span>') : '';
                        return '<td class="text-end oflm-mono-cell"><strong>' + fmtCompactRate(cell.p1.median) + '</strong>' + p2Str + '</td>';
                    }).join('');

                    return '<tr>' +
                        '<td><div class="fw-semibold">' + esc(row.npuShort) + '</div></td>' +
                        '<td>' + renderCompactSparkbarSvg(row, maxP1, model.unit, t) + '</td>' +
                        cells +
                    '</tr>';
                }).join('');

                bodyEl.innerHTML =
                    '<div class="oflm-chart-card">' +
                        selectorCard +
                        '<div class="oflm-table-wrap mb-0 mt-3">' +
                            '<table class="oflm-table oflm-npu-matrix">' +
                                '<thead><tr>' +
                                    '<th>XDNA2 NPU</th>' +
                                    '<th>' + esc(model.p1Label) + ' Range &amp; Median (' + esc(model.unit) + ')</th>' +
                                    headerCols +
                                '</tr></thead>' +
                                '<tbody>' + bodyRows + '</tbody>' +
                            '</table>' +
                        '</div>' +
                        '<p class="small mb-0 mt-2" style="color: var(--oflm-muted);">Dot shows median; range bar shows min–max across memory speeds &amp; power modes (hover sparkbar for exact min/max configs). Cells show ' + esc(model.p1Label) + ' / ' + esc(model.p2Label) + ' (' + esc(model.unit) + ').</p>' +
                    '</div>';
            } else {
                // VARIATION 1 (/benchmarks/): Side-by-side NPU range+median charts + linked NPU x Memory matrix
                const hasP2 = npuRows.some(function (r) { return r.p2 != null; });
                const chartHeight = Math.max(155, npuRows.length * 52 + 64);

                const displayRows = npuRows.slice().reverse(); // fastest at top in matrix
                const maxP1 = npuRows.length
                    ? Math.max.apply(null, npuRows.map(function (r) { return (r.p1 && r.p1.median) || 1; }))
                    : 1;

                const matrixHead = memCols.map(function (mc) {
                    const isHl = hlMem === mc && !hlNpu;
                    return '<th class="text-center oflm-matrix-col' + (isHl ? ' is-selected' : '') + '" data-hl-col="' + esc(mc) + '">' +
                        '<div>' + esc(mc) + '</div>' +
                    '</th>';
                }).join('');

                const matrixRows = displayRows.map(function (row) {
                    const cells = memCols.map(function (mc) {
                        const cell = row.byMem[mc];
                        if (!cell || !cell.p1) {
                            return '<td class="text-center" style="color: var(--oflm-muted);">—</td>';
                        }
                        const isCellHl = hlMem === mc && (!hlNpu || hlNpu === row.npuShort);
                        const ratio = Math.min(1, cell.p1.median / maxP1);
                        const alpha = (0.06 + 0.16 * ratio).toFixed(2);
                        const bg = t.dark
                            ? 'rgba(109, 148, 255, ' + alpha + ')'
                            : 'rgba(37, 84, 212, ' + alpha + ')';
                        const p2Line = cell.p2
                            ? '<div class="small" style="color: var(--oflm-muted);">' + fmtCompactRate(cell.p2.median) + ' · n=' + cell.runs + '</div>'
                            : '<div class="small" style="color: var(--oflm-muted);">n=' + cell.runs + '</div>';
                        return '<td class="text-center oflm-matrix-cell' + (isCellHl ? ' is-selected' : '') + '" data-hl-npu="' + esc(row.npuShort) + '" data-hl-mem="' + esc(mc) + '" style="background-color:' + bg + ';">' +
                            '<div class="fw-bold oflm-mono-cell">' + fmtCompactRate(cell.p1.median) + ' <span class="small fw-normal" style="color: var(--oflm-muted);">' + esc(model.unit) + '</span></div>' +
                            p2Line +
                        '</td>';
                    }).join('');

                    return '<tr>' +
                        '<td><div class="fw-semibold">' + esc(row.npuShort) + '</div><div class="small" style="color: var(--oflm-muted);">' + esc(row.npuFull) + '</div></td>' +
                        cells +
                    '</tr>';
                }).join('');

                const clearBtnHtml = (hlMem || hlNpu)
                    ? '<button type="button" class="btn btn-sm btn-link p-0 text-decoration-none" data-clear-hl>Clear highlight</button>'
                    : '';

                bodyEl.innerHTML =
                    '<div class="oflm-chart-card">' +
                        selectorCard +
                        '<div class="' + (hasP2 ? 'oflm-small-multiples' : '') + ' mt-3">' +
                            '<div class="oflm-subchart">' +
                                '<div class="oflm-subchart-title">' + esc(model.p1Label) + ' <span class="oflm-subchart-unit">(' + esc(model.p1Sub) + ')</span></div>' +
                                '<div class="oflm-echarts-canvas" data-npu-panel="p1" style="width: 100%; height: ' + chartHeight + 'px;" aria-label="' + esc(model.tag + ' ' + model.p1Label) + '"></div>' +
                            '</div>' +
                            (hasP2
                                ? '<div class="oflm-subchart">' +
                                      '<div class="oflm-subchart-title">' + esc(model.p2Label) + ' <span class="oflm-subchart-unit">(' + esc(model.p2Sub) + ')</span></div>' +
                                      '<div class="oflm-echarts-canvas" data-npu-panel="p2" style="width: 100%; height: ' + chartHeight + 'px;" aria-label="' + esc(model.tag + ' ' + model.p2Label) + '"></div>' +
                                  '</div>'
                                : '') +
                        '</div>' +
                        '<div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mt-3 mb-2">' +
                            '<div class="small fw-semibold" style="color: var(--oflm-muted);">NPU × Memory Configuration Matrix <span class="fw-normal">(click any column or cell to highlight above; hover dots for min/max configs)</span></div>' +
                            clearBtnHtml +
                        '</div>' +
                        '<div class="oflm-table-wrap mb-0">' +
                            '<table class="oflm-table oflm-npu-matrix">' +
                                '<thead><tr><th>XDNA2 NPU</th>' + matrixHead + '</tr></thead>' +
                                '<tbody>' + matrixRows + '</tbody>' +
                            '</table>' +
                        '</div>' +
                    '</div>';

                const p1El = bodyEl.querySelector('[data-npu-panel="p1"]');
                if (p1El) {
                    const c1 = window.echarts.init(p1El, null, { renderer: 'svg' });
                    const r1 = function () { c1.setOption(buildNpuRangeOption(model, activeQuant, npuRows, 'p1', hlMem, hlNpu), true); };
                    r1();
                    const e1 = { chart: c1, render: r1 };
                    activeInstances.push(e1);
                    childCharts.push(e1);
                }
                const p2El = bodyEl.querySelector('[data-npu-panel="p2"]');
                if (p2El) {
                    const c2 = window.echarts.init(p2El, null, { renderer: 'svg' });
                    const r2 = function () { c2.setOption(buildNpuRangeOption(model, activeQuant, npuRows, 'p2', hlMem, hlNpu), true); };
                    r2();
                    const e2 = { chart: c2, render: r2 };
                    activeInstances.push(e2);
                    childCharts.push(e2);
                }
            }

            // Bind dropdown & matrix interactions
            const ddEl = bodyEl.querySelector('.oflm-dd');
            const ddToggle = bodyEl.querySelector('[data-dd-toggle]');
            if (ddToggle && ddEl) {
                ddToggle.addEventListener('click', function (ev) {
                    ev.stopPropagation();
                    ddOpen = !ddOpen;
                    ddEl.classList.toggle('is-open', ddOpen);
                    ddToggle.setAttribute('aria-expanded', ddOpen ? 'true' : 'false');
                });
            }
            bodyEl.querySelectorAll('[data-pick-model]').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    activeModelTag = btn.dataset.pickModel;
                    activeQuant = btn.dataset.pickQuant;
                    hlMem = null;
                    hlNpu = null;
                    ddOpen = false;
                    renderWidget();
                });
            });
            bodyEl.querySelectorAll('[data-hl-col]').forEach(function (th) {
                th.addEventListener('click', function () {
                    const mc = th.dataset.hlCol;
                    hlNpu = null;
                    hlMem = (hlMem === mc && !hlNpu) ? null : mc;
                    renderWidget();
                });
            });
            bodyEl.querySelectorAll('[data-hl-npu]').forEach(function (td) {
                td.addEventListener('click', function () {
                    const n = td.dataset.hlNpu;
                    const mc = td.dataset.hlMem;
                    const same = hlNpu === n && hlMem === mc;
                    hlNpu = same ? null : n;
                    hlMem = same ? null : mc;
                    renderWidget();
                });
            });
            const clearBtn = bodyEl.querySelector('[data-clear-hl]');
            if (clearBtn) {
                clearBtn.addEventListener('click', function () {
                    hlMem = null;
                    hlNpu = null;
                    renderWidget();
                });
            }
        }

        document.addEventListener('click', function (ev) {
            if (!ddOpen) return;
            const ddEl = bodyEl.querySelector('.oflm-dd');
            if (ddEl && !ddEl.contains(ev.target)) {
                ddOpen = false;
                ddEl.classList.remove('is-open');
                const btn = ddEl.querySelector('[data-dd-toggle]');
                if (btn) btn.setAttribute('aria-expanded', 'false');
            }
        });

        document.addEventListener('keydown', function (ev) {
            if (ev.key === 'Escape' && ddOpen) {
                ddOpen = false;
                const ddEl = bodyEl.querySelector('.oflm-dd');
                if (ddEl) {
                    ddEl.classList.remove('is-open');
                    const btn = ddEl.querySelector('[data-dd-toggle]');
                    if (btn) {
                        btn.setAttribute('aria-expanded', 'false');
                        btn.focus();
                    }
                }
            }
        });

        renderWidget();

        // Re-render compact SVG sparkbars on light/dark theme switch
        document.addEventListener('oflm:themechange', function () {
            if (viewMode === 'compact') renderWidget();
        });

        allRunsUpdaters.push(function (liveRuns) {
            dataset = Object.assign({}, dataset, { runs: liveRuns });
            renderWidget();
        });
    }

    // ==================== LLM CONTEXT SCALING SMALL MULTIPLES ====================

    function buildLlmThroughputOption(dataset) {
        const t = getThemeTokens();
        const rows = dataset.rows || [];
        const labels = rows.map(function (r) { return r.context; });
        const prefillPts = rows.map(function (r) { return { mean: r.prefill_tps, std: r.prefill_std || 0 }; });
        const decodePts = rows.map(function (r) { return { mean: r.decode_tps, std: r.decode_std || 0 }; });

        return {
            backgroundColor: 'transparent',
            animationDuration: 280,
            grid: { left: 52, right: 20, top: 44, bottom: 42 },
            legend: {
                top: 2,
                left: 0,
                itemGap: 16,
                textStyle: { color: t.text, fontSize: 12 }
            },
            toolbox: {
                right: 0,
                top: 0,
                iconStyle: { borderColor: t.muted },
                feature: {
                    saveAsImage: { title: 'Save SVG', name: (dataset.slug || 'oflm-llm') + '-throughput' }
                }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: t.tooltipBg,
                borderColor: t.tooltipBorder,
                borderWidth: 1,
                textStyle: { color: t.text, fontSize: 12.5 },
                formatter: function (params) {
                    if (!params || !params.length) return '';
                    const r = rows[params[0].dataIndex];
                    if (!r) return '';
                    return [
                        '<div style="font-weight:700;margin-bottom:4px;">Context: ' + esc(r.context) + '</div>',
                        '<div>Prefill: <strong>' + r.prefill_tps.toFixed(2) + ' ± ' + r.prefill_std.toFixed(2) + ' tok/s</strong></div>',
                        '<div>Decode: <strong>' + r.decode_tps.toFixed(2) + ' ± ' + r.decode_std.toFixed(2) + ' tok/s</strong></div>',
                        '<div>TTFT: <strong>' + r.ttft_s.toFixed(3) + ' ± ' + r.ttft_std.toFixed(3) + ' s</strong></div>'
                    ].join('');
                }
            },
            xAxis: {
                type: 'category',
                name: 'Context Length',
                nameLocation: 'middle',
                nameGap: 25,
                nameTextStyle: { color: t.muted, fontWeight: 600, fontSize: 11.5 },
                data: labels,
                axisLabel: { color: t.text, fontWeight: 600 },
                axisLine: { lineStyle: { color: t.axis } },
                axisTick: { show: false }
            },
            yAxis: {
                type: 'log',
                min: 4,
                max: 700,
                axisLabel: { color: t.muted, fontSize: 11 },
                axisLine: { show: false },
                axisTick: { show: false },
                splitLine: { lineStyle: { color: t.grid } }
            },
            series: [
                makeBandPolygonSeries('Prefill', prefillPts, t.primaryBand, 0.5),
                makeWhiskerSeries('Prefill', prefillPts, t.primary, 0.5),
                {
                    name: 'Prefill (tok/s)',
                    type: 'line',
                    smooth: 0.15,
                    symbol: 'circle',
                    symbolSize: 7,
                    z: 5,
                    itemStyle: { color: t.primary },
                    lineStyle: { width: 2.5, color: t.primary },
                    label: {
                        show: true,
                        position: 'top',
                        distance: 8,
                        color: t.primary,
                        fontSize: 11,
                        fontWeight: 700,
                        formatter: function (p) { return Math.round(p.value) + ''; }
                    },
                    data: rows.map(function (r) { return r.prefill_tps; })
                },
                makeBandPolygonSeries('Decode', decodePts, t.secondaryBand, 0.5),
                makeWhiskerSeries('Decode', decodePts, t.secondary, 0.5),
                {
                    name: 'Decode (tok/s)',
                    type: 'line',
                    smooth: 0.15,
                    symbol: 'diamond',
                    symbolSize: 8,
                    z: 5,
                    itemStyle: { color: t.secondary },
                    lineStyle: { width: 2.5, color: t.secondary },
                    label: {
                        show: true,
                        position: 'top',
                        distance: 8,
                        color: t.secondary,
                        fontSize: 11,
                        fontWeight: 700,
                        formatter: function (p) { return Number(p.value).toFixed(1); }
                    },
                    data: rows.map(function (r) { return r.decode_tps; })
                }
            ]
        };
    }

    function buildLlmTtftOption(dataset) {
        const t = getThemeTokens();
        const rows = dataset.rows || [];
        const labels = rows.map(function (r) { return r.context; });
        const ttftPts = rows.map(function (r) { return { mean: r.ttft_s, std: r.ttft_std || 0 }; });

        return {
            backgroundColor: 'transparent',
            animationDuration: 280,
            grid: { left: 52, right: 20, top: 44, bottom: 42 },
            toolbox: {
                right: 0,
                top: 0,
                iconStyle: { borderColor: t.muted },
                feature: {
                    saveAsImage: { title: 'Save SVG', name: (dataset.slug || 'oflm-llm') + '-ttft' }
                }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'shadow' },
                backgroundColor: t.tooltipBg,
                borderColor: t.tooltipBorder,
                borderWidth: 1,
                textStyle: { color: t.text, fontSize: 12.5 },
                formatter: function (params) {
                    if (!params || !params.length) return '';
                    const r = rows[params[0].dataIndex];
                    if (!r) return '';
                    return [
                        '<div style="font-weight:700;margin-bottom:4px;">Context: ' + esc(r.context) + '</div>',
                        '<div>TTFT: <strong>' + r.ttft_s.toFixed(3) + ' ± ' + r.ttft_std.toFixed(3) + ' s</strong></div>',
                        '<div>Prefill: <strong>' + r.prefill_tps.toFixed(2) + ' tok/s</strong></div>'
                    ].join('');
                }
            },
            xAxis: {
                type: 'category',
                name: 'Context Length',
                nameLocation: 'middle',
                nameGap: 25,
                nameTextStyle: { color: t.muted, fontWeight: 600, fontSize: 11.5 },
                data: labels,
                axisLabel: { color: t.text, fontWeight: 600 },
                axisLine: { lineStyle: { color: t.axis } },
                axisTick: { show: false }
            },
            yAxis: {
                type: 'value',
                min: 0,
                max: 128,
                axisLabel: {
                    color: t.muted,
                    fontSize: 11,
                    formatter: function (v) { return v + 's'; }
                },
                axisLine: { show: false },
                axisTick: { show: false },
                splitLine: { lineStyle: { color: t.grid } }
            },
            series: [
                {
                    name: 'TTFT (s)',
                    type: 'bar',
                    barMaxWidth: 34,
                    z: 2,
                    itemStyle: { color: t.amber, borderRadius: [4, 4, 0, 0] },
                    label: {
                        show: true,
                        position: 'top',
                        distance: 6,
                        color: t.text,
                        fontSize: 11,
                        fontWeight: 700,
                        formatter: function (p) { return Number(p.value).toFixed(1) + 's'; }
                    },
                    data: rows.map(function (r) { return r.ttft_s; })
                },
                makeWhiskerSeries('TTFT', ttftPts, t.text, 0)
            ]
        };
    }

    // ==================== EMBEDDING BATCH SCALING SMALL MULTIPLES ====================

    function shortEmbedName(tag) {
        return String(tag || '')
            .replace(':en-v1.5', '')
            .replace(':v1.5', '')
            .replace(':l6-v2', '')
            .replace(':base', '')
            .replace(':300m', '');
    }

    function buildEmbedSubOption(dataset, mode) {
        const t = getThemeTokens();
        const batches = dataset.batches || ['n=1', 'n=2', 'n=4', 'n=8', 'n=16', 'n=32', 'n=64', 'n=128'];
        const models = dataset.models || [];
        const isSpeedup = mode === 'speedup';

        const series = models.map(function (m, idx) {
            const color = t.palette[idx % t.palette.length];
            const values = isSpeedup ? m.speedup_x : m.throughput_tps;
            const shortName = shortEmbedName(m.tag);
            const peakVal = isSpeedup
                ? Math.max.apply(null, values.filter(function (v) { return v != null; }))
                : m.peak_tps;

            return {
                name: m.tag,
                type: 'line',
                smooth: 0.15,
                symbol: 'circle',
                symbolSize: 6,
                itemStyle: { color: color },
                lineStyle: {
                    width: m.backend === 'NPU + CPU' ? 2 : 2.4,
                    type: m.backend === 'NPU + CPU' ? 'dashed' : 'solid',
                    color: color
                },
                endLabel: {
                    show: true,
                    distance: 6,
                    color: color,
                    fontSize: 10.5,
                    fontWeight: 700,
                    formatter: function () {
                        if (isSpeedup) {
                            return shortName + ' (' + Number(peakVal).toFixed(1) + '×)';
                        }
                        return shortName + ' (' + fmtCompactRate(peakVal) + ')';
                    }
                },
                labelLayout: { moveOverlap: 'shiftY' },
                data: values
            };
        });

        if (isSpeedup) {
            series[0].markLine = {
                silent: true,
                symbol: 'none',
                lineStyle: { color: t.muted, type: 'dashed', width: 1 },
                label: { show: true, position: 'insideStartTop', formatter: '1.0× (no batching)', color: t.muted, fontSize: 10.5 },
                data: [{ yAxis: 1 }]
            };
        }

        return {
            backgroundColor: 'transparent',
            animationDuration: 280,
            grid: { left: 50, right: 148, top: 24, bottom: 42 },
            toolbox: {
                right: 0,
                top: 0,
                iconStyle: { borderColor: t.muted },
                feature: {
                    saveAsImage: { title: 'Save SVG', name: 'oflm-npu-embeddings-' + mode }
                }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: t.tooltipBg,
                borderColor: t.tooltipBorder,
                borderWidth: 1,
                textStyle: { color: t.text, fontSize: 12 },
                formatter: function (params) {
                    if (!params || !params.length) return '';
                    const batchLabel = params[0].axisValue;
                    const lines = ['<div style="font-weight:700;margin-bottom:4px;">Batch ' + esc(batchLabel) + '</div>'];
                    params.forEach(function (p) {
                        if (p.value == null) return;
                        const valStr = isSpeedup
                            ? (Number(p.value).toFixed(2) + '×')
                            : (fmtNum(p.value, 1) + ' txt/s');
                        lines.push(
                            '<div><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + p.color + ';margin-right:6px;"></span>' +
                            esc(p.seriesName) + ': <strong>' + valStr + '</strong></div>'
                        );
                    });
                    return lines.join('');
                }
            },
            xAxis: {
                type: 'category',
                name: 'Batch Size',
                nameLocation: 'middle',
                nameGap: 25,
                nameTextStyle: { color: t.muted, fontWeight: 600, fontSize: 11.5 },
                data: batches,
                axisLabel: { color: t.text, fontWeight: 600, fontSize: 11 },
                axisLine: { lineStyle: { color: t.axis } },
                axisTick: { show: false }
            },
            yAxis: {
                type: isSpeedup ? 'value' : 'log',
                min: isSpeedup ? 0 : 1,
                axisLabel: {
                    color: t.muted,
                    fontSize: 11,
                    formatter: function (v) { return isSpeedup ? (v + '×') : v; }
                },
                axisLine: { show: false },
                axisTick: { show: false },
                splitLine: { lineStyle: { color: t.grid } }
            },
            series: series
        };
    }

    // ==================== INITIALIZERS ====================

    function initLlmCard(card) {
        if (!window.echarts) return;
        const payloadEl = card.querySelector('.oflm-chart-payload');
        const tpCanvas = card.querySelector('[data-subchart="throughput"]');
        const ttftCanvas = card.querySelector('[data-subchart="ttft"]');
        if (!payloadEl || !tpCanvas || !ttftCanvas) return;

        let dataset;
        try {
            dataset = JSON.parse(payloadEl.textContent);
        } catch (_err) {
            return;
        }

        const tpChart = window.echarts.init(tpCanvas, null, { renderer: 'svg' });
        const renderTp = function () { tpChart.setOption(buildLlmThroughputOption(dataset), true); };
        renderTp();
        registerInstance(tpChart, renderTp);

        const ttftChart = window.echarts.init(ttftCanvas, null, { renderer: 'svg' });
        const renderTtft = function () { ttftChart.setOption(buildLlmTtftOption(dataset), true); };
        renderTtft();
        registerInstance(ttftChart, renderTtft);
    }

    function initEmbedCard(card) {
        if (!window.echarts) return;
        const payloadEl = card.querySelector('.oflm-chart-payload');
        const tpCanvas = card.querySelector('[data-subchart="throughput"]');
        const spCanvas = card.querySelector('[data-subchart="speedup"]');
        if (!payloadEl || !tpCanvas || !spCanvas) return;

        let dataset;
        try {
            dataset = JSON.parse(payloadEl.textContent);
        } catch (_err) {
            return;
        }

        const tpChart = window.echarts.init(tpCanvas, null, { renderer: 'svg' });
        const renderTp = function () { tpChart.setOption(buildEmbedSubOption(dataset, 'throughput'), true); };
        renderTp();
        registerInstance(tpChart, renderTp);

        const spChart = window.echarts.init(spCanvas, null, { renderer: 'svg' });
        const renderSp = function () { spChart.setOption(buildEmbedSubOption(dataset, 'speedup'), true); };
        renderSp();
        registerInstance(spChart, renderSp);
    }

    function bootAllCharts() {
        document.querySelectorAll('[data-oflm-allruns-chart]').forEach(initNpuBenchWidget);
        document.querySelectorAll('[data-oflm-chart]').forEach(initLlmCard);
        document.querySelectorAll('[data-oflm-embed-chart]').forEach(initEmbedCard);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bootAllCharts);
    } else {
        bootAllCharts();
    }

    window.addEventListener('resize', function () {
        activeInstances.forEach(function (entry) { entry.chart.resize(); });
    });

    const observer = new MutationObserver(function () {
        activeInstances.forEach(function (entry) { entry.render(); });
        document.dispatchEvent(new CustomEvent('oflm:themechange', { detail: { dark: isDarkMode() } }));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-dark-mode'] });

    window.OFLMCharts = {
        getThemeTokens: getThemeTokens,
        getModalityMeta: getModalityMeta,
        extractHardwarePlatform: extractHardwarePlatform,
        register: registerInstance,
        syncLiveRuns: function (rawRows) {
            if (!Array.isArray(rawRows) || !rawRows.length) return;
            const mapped = rawRows.map(function (r) {
                const isEmbed = r.task === 'Embedding';
                const mem = r.ram_spec || (r.ram_gb ? r.ram_gb + ' GB RAM' : '');
                const stage1 = Array.isArray(r.stages)
                    ? r.stages.find(function (s) { return Number(s.ctx_k) === 1; })
                    : null;
                return {
                    model: r.model,
                    task: r.task,
                    backend: r.backend || '',
                    source: r.source || (r.status === 'seed' ? 'verified_seed' : 'community'),
                    status: r.status,
                    hardware_platform: r.cpu || extractHardwarePlatform(r),
                    hardware: r.hardware + (mem && !String(r.hardware).includes('GB') ? ' · ' + mem : ''),
                    memory_config: extractMemoryConfig(r),
                    power_mode: extractPowerMode(r),
                    quant: r.quantization || '',
                    runs: r.runs || 1,
                    stage: r.headline?.ctx_k != null ? (isEmbed ? 'n=' + r.headline.ctx_k : r.headline.ctx_k + 'k ctx') : '',
                    single_tps: isEmbed && stage1?.decode_avg != null ? Number(stage1.decode_avg) : null,
                    single_latency_ms: isEmbed && stage1?.ttft_avg_s != null ? Number((stage1.ttft_avg_s * 1000).toFixed(1)) : null,
                    output_tps: r.headline?.decode ?? null,
                    output_unit: isEmbed ? 'txt/s' : 'tok/s',
                    token_tps: r.headline?.prefill ?? null,
                    latency_ms: r.headline?.ttft_ms ?? null
                };
            });
            allRunsUpdaters.forEach(function (fn) { fn(mapped); });
        }
    };
})();
