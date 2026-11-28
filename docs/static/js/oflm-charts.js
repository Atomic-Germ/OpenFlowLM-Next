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

    // ==================== NPU-CENTRIC MODEL + QUANT -> PAIRED BARS, TTFT RANGE & MATRIX ====================

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

            // barA = Prefill (Chat) or Peak Batched (Embed); barB = Decode (Chat) or Single-Text n=1 (Embed)
            const barA = isEmbed
                ? (r.output_tps != null ? Number(r.output_tps) : null)
                : (r.token_tps != null ? Number(r.token_tps) : null);
            const barB = isEmbed
                ? (r.single_tps != null ? Number(r.single_tps) : null)
                : (r.output_tps != null ? Number(r.output_tps) : null);

            // Latency: TTFT (s) for Chat/Vision; Peak Batched (ms) + Single-Text n=1 (ms) for Embeddings
            const latA = r.latency_ms != null
                ? (isEmbed ? Number(r.latency_ms) : Number(r.latency_ms) / 1000.0)
                : null;
            const latB = isEmbed && r.single_latency_ms != null
                ? Number(r.single_latency_ms)
                : null;

            if (!modelsMap.has(tag)) {
                modelsMap.set(tag, {
                    tag: tag,
                    task: r.task,
                    modality: mod,
                    inferenceInfo: resolveInferenceInfo(r),
                    minRamGb: inferMinRamGb(r),
                    unit: r.output_unit || (isEmbed ? 'txt/s' : 'tok/s'),
                    barALabel: isEmbed ? 'Peak Batched' : 'Prefill',
                    barBLabel: isEmbed ? 'Single-Text (n=1)' : 'Decode',
                    latALabel: isEmbed ? 'Peak Batched' : 'TTFT',
                    latBLabel: isEmbed ? 'Single-Text (n=1)' : null,
                    latUnit: isEmbed ? 'ms' : 's',
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
                barA: barA,
                barB: barB,
                latA: latA,
                latB: latB
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
                    barA: summarizeMetric(list, 'barA'),
                    barB: summarizeMetric(list, 'barB'),
                    latA: summarizeMetric(list, 'latA'),
                    latB: summarizeMetric(list, 'latB'),
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
                barA: summarizeMetric(grp.entries, 'barA'),
                barB: summarizeMetric(grp.entries, 'barB'),
                latA: summarizeMetric(grp.entries, 'latA'),
                latB: summarizeMetric(grp.entries, 'latB'),
                byMem: memSummaries
            };
        });

        // Sort ascending so ECharts yAxis puts fastest NPU at top
        npuRows.sort(function (a, b) {
            const av = (a.barB && a.barB.median) || (a.barA && a.barA.median) || 0;
            const bv = (b.barB && b.barB.median) || (b.barA && b.barA.median) || 0;
            return av - bv;
        });

        const memCols = Array.from(memSet).sort();
        return { npuRows: npuRows, memCols: memCols };
    }

    function fmtLat(val, latUnit) {
        if (val == null) return '-';
        if (latUnit === 's') return Number(val).toFixed(2) + ' s';
        if (val >= 1000) return Math.round(val).toLocaleString('en-US') + ' ms';
        return Number(val).toFixed(1) + ' ms';
    }

    function buildDualRangeOption(opts) {
        const t = getThemeTokens();
        const npuRows = opts.npuRows;
        const unit = opts.unit;
        const fmtVal = opts.fmtVal;
        const yLabels = npuRows.map(function (r) { return r.npuShort; });

        const hasA = npuRows.some(function (r) { return r[opts.keyA] != null; });
        const hasB = Boolean(opts.keyB) && npuRows.some(function (r) { return r[opts.keyB] != null; });

        const allVals = [];
        npuRows.forEach(function (r) {
            if (hasA && r[opts.keyA]) { allVals.push(r[opts.keyA].min, r[opts.keyA].max); }
            if (hasB && r[opts.keyB]) { allVals.push(r[opts.keyB].min, r[opts.keyB].max); }
        });
        const maxVal = allVals.length ? Math.max.apply(null, allVals) : 100;
        const minVal = allVals.length ? Math.min.apply(null, allVals) : 1;
        const useLog = allVals.length > 1 && (maxVal / minVal) >= 12;
        const xMin = useLog ? (minVal < 2 ? 1 : Math.pow(10, Math.floor(Math.log10(minVal)))) : 0;
        const xMax = useLog
            ? Math.pow(10, Math.ceil(Math.log10(maxVal * 3.2)))
            : (maxVal * 1.32);

        function makeRangeDotSeries(seriesName, metricKey, color, bandFill, yOff) {
            const rowsData = [];
            npuRows.forEach(function (row, idx) {
                const stat = row[metricKey];
                if (!stat) return;
                rowsData.push([idx, stat.min, stat.max, stat.median]);
            });
            return {
                name: seriesName,
                type: 'custom',
                z: 4,
                encode: { y: 0, x: [1, 2, 3] },
                itemStyle: { color: color },
                renderItem: function (params, api) {
                    const yIdx = api.value(0);
                    const lo = api.value(1);
                    const hi = api.value(2);
                    const med = api.value(3);
                    const pStart = api.coord([xMin, yIdx]);
                    const pLo = api.coord([lo, yIdx]);
                    const pHi = api.coord([hi, yIdx]);
                    const pMed = api.coord([med, yIdx]);
                    const y = pMed[1] + yOff;
                    const capH = 12;

                    const children = [
                        {
                            type: 'line',
                            shape: { x1: pStart[0], y1: y, x2: pLo[0], y2: y },
                            style: { stroke: t.grid, lineWidth: 1.5 }
                        }
                    ];

                    if (hi > lo) {
                        const w = Math.max(8, pHi[0] - pLo[0]);
                        children.push({
                            type: 'rect',
                            shape: { x: pLo[0], y: y - capH / 2, width: w, height: capH, r: capH / 2 },
                            style: { fill: bandFill, stroke: color, lineWidth: 1.2 }
                        });
                    }

                    // Median dot
                    children.push({
                        type: 'circle',
                        shape: { cx: pMed[0], cy: y, r: 5.5 },
                        style: { fill: color, stroke: t.tooltipBg, lineWidth: 1.8 }
                    });

                    // Direct numeric label to the right of max/median
                    const labelX = Math.max(pHi[0], pMed[0]) + 9;
                    const labelText = hi > lo
                        ? (fmtVal(med) + ' (' + fmtVal(lo) + ' - ' + fmtVal(hi) + ')')
                        : fmtVal(med);
                    children.push({
                        type: 'text',
                        style: {
                            x: labelX,
                            y: y + 4,
                            text: labelText,
                            fill: t.text,
                            font: '700 11px JetBrains Mono, ui-monospace, monospace'
                        }
                    });

                    return { type: 'group', children: children };
                },
                data: rowsData
            };
        }

        const series = [];
        const legendNames = [];
        const colorA = opts.colorA;
        const bandA = opts.bandA;
        const colorB = opts.colorB;
        const bandB = opts.bandB;

        if (hasA) {
            const nameA = opts.labelA + ' (' + unit + ')';
            legendNames.push(nameA);
            series.push(makeRangeDotSeries(nameA, opts.keyA, colorA, bandA, hasB ? -11 : 0));
        }
        if (hasB) {
            const nameB = opts.labelB + ' (' + unit + ')';
            legendNames.push(nameB);
            series.push(makeRangeDotSeries(nameB, opts.keyB, colorB, bandB, hasA ? 11 : 0));
        }

        // Memory-tier highlight: crisp ringed halo in the SAME color as each track (excluded from legend)
        if (opts.hlMem) {
            const hlPts = [];
            npuRows.forEach(function (row, idx) {
                if (opts.hlNpu && row.npuShort !== opts.hlNpu) return;
                const cell = row.byMem[opts.hlMem];
                if (!cell) return;
                if (hasA && cell[opts.keyA]) hlPts.push([cell[opts.keyA].median, idx, hasB ? -11 : 0, 0]);
                if (hasB && cell[opts.keyB]) hlPts.push([cell[opts.keyB].median, idx, hasA ? 11 : 0, 1]);
            });
            if (hlPts.length) {
                series.push({
                    type: 'custom',
                    silent: true,
                    z: 8,
                    encode: { x: 0, y: 1 },
                    tooltip: { show: false },
                    renderItem: function (params, api) {
                        const pt = api.coord([api.value(0), api.value(1)]);
                        const y = pt[1] + api.value(2);
                        const trackColor = api.value(3) === 0 ? colorA : colorB;
                        return {
                            type: 'group',
                            children: [
                                {
                                    type: 'circle',
                                    shape: { cx: pt[0], cy: y, r: 9.5 },
                                    style: { fill: 'none', stroke: trackColor, lineWidth: 2 }
                                },
                                {
                                    type: 'circle',
                                    shape: { cx: pt[0], cy: y, r: 4.5 },
                                    style: { fill: t.tooltipBg, stroke: trackColor, lineWidth: 2.2 }
                                }
                            ]
                        };
                    },
                    data: hlPts
                });
            }
        }

        return {
            backgroundColor: 'transparent',
            animationDuration: 260,
            grid: { left: 126, right: 112, top: 36, bottom: 38 },
            legend: {
                show: legendNames.length > 0,
                data: legendNames,
                top: 2,
                left: 0,
                icon: 'circle',
                itemWidth: 10,
                itemHeight: 10,
                itemGap: 16,
                textStyle: { color: t.text, fontSize: 11.5, fontWeight: 600 }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'shadow' },
                backgroundColor: t.tooltipBg,
                borderColor: t.tooltipBorder,
                borderWidth: 1,
                textStyle: { color: t.text, fontSize: 12 },
                formatter: function (params) {
                    if (!params || !params.length) return '';
                    const p0 = params[0];
                    const rowIdx = Array.isArray(p0.value) ? p0.value[0] : p0.dataIndex;
                    const row = npuRows[rowIdx];
                    if (!row) return '';
                    const lines = [
                        '<div style="font-weight:700;margin-bottom:3px;">' + esc(row.npuFull) + '</div>',
                        '<div style="font-size:11px;opacity:0.82;margin-bottom:5px;">' + esc(opts.modelTag) + ' · ' + esc(opts.quant) + ' · ' + row.totalRuns + ' runs across ' + row.memCount + ' memory config(s)</div>'
                    ];
                    function pushStat(label, stat) {
                        if (!stat) return;
                        lines.push('<div>' + esc(label) + ' median: <strong>' + esc(fmtVal(stat.median)) + '</strong></div>');
                        if (stat.max > stat.min) {
                            lines.push('<div style="font-size:11px;opacity:0.82;padding-left:8px;">Min ' + esc(fmtVal(stat.min)) + ' (' + esc(stat.minConfig) + ') | Max ' + esc(fmtVal(stat.max)) + ' (' + esc(stat.maxConfig) + ')</div>');
                        } else {
                            lines.push('<div style="font-size:11px;opacity:0.82;padding-left:8px;">Config: ' + esc(stat.minConfig) + '</div>');
                        }
                    }
                    if (hasA) pushStat(opts.labelA, row[opts.keyA]);
                    if (hasB) pushStat(opts.labelB, row[opts.keyB]);
                    return lines.join('');
                }
            },
            xAxis: {
                type: useLog ? 'log' : 'value',
                min: xMin,
                max: xMax,
                name: unit + (useLog ? ' (log scale)' : ''),
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

    function buildCombinedThroughputOption(model, quant, npuRows, hlMem, hlNpu) {
        const t = getThemeTokens();
        return buildDualRangeOption({
            modelTag: model.tag,
            quant: quant,
            npuRows: npuRows,
            keyA: 'barA',
            keyB: 'barB',
            labelA: model.barALabel,
            labelB: model.barBLabel,
            colorA: t.primary,
            bandA: t.primaryBand,
            colorB: t.secondary,
            bandB: t.secondaryBand,
            unit: model.unit,
            fmtVal: function (v) { return fmtCompactRate(v) + ' ' + model.unit; },
            hlMem: hlMem,
            hlNpu: hlNpu
        });
    }

    function buildLatencyRangeOption(model, quant, npuRows, hlMem, hlNpu) {
        const t = getThemeTokens();
        const hasDual = npuRows.some(function (r) { return r.latB != null; });
        return buildDualRangeOption({
            modelTag: model.tag,
            quant: quant,
            npuRows: npuRows,
            keyA: 'latA',
            keyB: hasDual ? 'latB' : null,
            labelA: model.latALabel,
            labelB: model.latBLabel,
            colorA: hasDual ? t.primary : t.amber,
            bandA: hasDual ? t.primaryBand : t.amberBand,
            colorB: t.secondary,
            bandB: t.secondaryBand,
            unit: model.latUnit,
            fmtVal: function (v) { return fmtLat(v, model.latUnit); },
            hlMem: hlMem,
            hlNpu: hlNpu
        });
    }

    function renderCompactSparkbarSvg(row, maxVal, unit, t) {
        const stat = row.barB || row.barA;
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
        let filterQuery = '';
        let filterModality = 'all';
        let childCharts = [];

        function buildDropdownListHtml(models) {
            const q = filterQuery.trim().toLowerCase();
            const filtered = models.filter(function (m) {
                const modOk = filterModality === 'all' || m.modality.key === filterModality;
                if (!modOk) return false;
                if (!q) return true;
                if (m.tag.toLowerCase().includes(q)) return true;
                if (m.modality.label.toLowerCase().includes(q)) return true;
                return Array.from(m.quants.keys()).some(function (qk) { return qk.toLowerCase().includes(q); });
            });

            if (!filtered.length) {
                return '<div class="oflm-dd-empty">No models match "' + esc(filterQuery) + '".</div>';
            }

            return filtered.map(function (m) {
                const qList = Array.from(m.quants.keys());
                const optsHtml = qList.map(function (qk) {
                    const qEntries = m.quants.get(qk) || [];
                    const npuSet = new Set(qEntries.map(function (e) { return e.npuShort; }));
                    const rCount = qEntries.reduce(function (acc, e) { return acc + e.runs; }, 0);
                    const isActive = m.tag === activeModelTag && qk === activeQuant;
                    return '<button type="button" class="oflm-dd-option' + (isActive ? ' active' : '') + '" role="option" aria-selected="' + (isActive ? 'true' : 'false') + '" data-pick-model="' + esc(m.tag) + '" data-pick-quant="' + esc(qk) + '">' +
                        '<span class="oflm-dd-opt-left">' +
                            '<span class="oflm-badge quant mono"><span>' + esc(qk) + '</span></span>' +
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
        }

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

            // Badge-style modality filter pills present in current dataset
            const modMap = new Map();
            models.forEach(function (m) {
                if (!modMap.has(m.modality.key)) modMap.set(m.modality.key, m.modality);
            });
            const modButtonsHtml =
                '<button type="button" class="oflm-badge neutral oflm-dd-mod-btn' + (filterModality === 'all' ? ' active' : '') + '" data-dd-mod="all"><span>All</span></button>' +
                Array.from(modMap.values()).map(function (mm) {
                    const active = filterModality === mm.key ? ' active' : '';
                    return '<button type="button" class="oflm-badge ' + esc(mm.badgeClass) + ' oflm-dd-mod-btn' + active + '" data-dd-mod="' + esc(mm.key) + '">' +
                        '<span class="material-icons" aria-hidden="true">' + esc(mm.icon) + '</span>' +
                        '<span>' + esc(mm.label) + '</span>' +
                    '</button>';
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
                            '<div class="oflm-dd-menu" role="listbox" aria-label="Models and quantizations">' +
                                '<div class="oflm-dd-controls">' +
                                    '<input type="search" class="oflm-dd-search" data-dd-search value="' + esc(filterQuery) + '" placeholder="Filter models..." autocomplete="off" aria-label="Filter models by name" />' +
                                    '<div class="oflm-dd-mod-bar" role="group" aria-label="Filter by modality">' + modButtonsHtml + '</div>' +
                                '</div>' +
                                '<div class="oflm-dd-list" data-dd-list>' + buildDropdownListHtml(models) + '</div>' +
                            '</div>' +
                        '</div>' +
                        '<div class="oflm-chart-meta">' +
                            renderInferenceBadge(model.inferenceInfo, true) +
                            '<span class="oflm-badge neutral"><span>Fits ≥ ' + esc(model.minRamGb) + ' GB RAM</span></span>' +
                            '<span class="small" style="color: var(--oflm-muted);">' + npuRows.length + (npuRows.length === 1 ? ' XDNA2 NPU' : ' XDNA2 NPUs') + ' · ' + memCols.length + (memCols.length === 1 ? ' memory config' : ' memory configs') + ' · ' + totalRuns + ' runs</span>' +
                        '</div>' +
                    '</div>' +
                '</div>';

            if (viewMode === 'compact') {
                // VARIATION 2 (Homepage): Fused NPU x Memory table with inline range + median sparkbar
                const maxPrimary = npuRows.length
                    ? Math.max.apply(null, npuRows.map(function (r) {
                        const s = r.barB || r.barA;
                        return (s && s.max) || 1;
                    }))
                    : 1;
                const displayRows = npuRows.slice().reverse();
                const sparkTitle = model.modality.key === 'embed' ? model.barALabel : model.barBLabel;

                const headerCols = memCols.map(function (mc) {
                    return '<th class="text-end">' + esc(mc) + '</th>';
                }).join('');

                const bodyRows = displayRows.map(function (row) {
                    const cells = memCols.map(function (mc) {
                        const cell = row.byMem[mc];
                        if (!cell || (!cell.barA && !cell.barB)) {
                            return '<td class="text-end" style="color: var(--oflm-muted);">-</td>';
                        }
                        const mainStat = model.modality.key === 'embed' ? cell.barA : (cell.barB || cell.barA);
                        const subStat = model.modality.key === 'embed' ? cell.barB : cell.barA;
                        const subStr = subStat ? (' <span class="small" style="color: var(--oflm-muted);">/ ' + fmtCompactRate(subStat.median) + '</span>') : '';
                        return '<td class="text-end oflm-mono-cell"><strong>' + fmtCompactRate(mainStat.median) + '</strong>' + subStr + '</td>';
                    }).join('');

                    return '<tr>' +
                        '<td><div class="fw-semibold">' + esc(row.npuShort) + '</div></td>' +
                        '<td>' + renderCompactSparkbarSvg(row, maxPrimary, model.unit, t) + '</td>' +
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
                                    '<th>' + esc(sparkTitle) + ' Range &amp; Median (' + esc(model.unit) + ', higher is better)</th>' +
                                    headerCols +
                                '</tr></thead>' +
                                '<tbody>' + bodyRows + '</tbody>' +
                            '</table>' +
                        '</div>' +
                        '<p class="small mb-0 mt-2" style="color: var(--oflm-muted);">Dot shows median; range bar shows min-max across memory speeds and power modes (hover sparkbar for min and max configs). Cells show ' + esc(model.modality.key === 'embed' ? (model.barALabel + ' / ' + model.barBLabel) : (model.barBLabel + ' / ' + model.barALabel)) + ' (' + esc(model.unit) + ').</p>' +
                    '</div>';
            } else {
                // VARIATION 1 (/benchmarks/): Throughput + Latency dual range charts + linked NPU x Memory matrix
                const hasLat = npuRows.some(function (r) { return r.latA != null || r.latB != null; });
                const chartHeight = Math.max(175, npuRows.length * 64 + 72);

                const displayRows = npuRows.slice().reverse();
                const maxMain = npuRows.length
                    ? Math.max.apply(null, npuRows.map(function (r) {
                        const s = model.modality.key === 'embed' ? r.barA : (r.barB || r.barA);
                        return (s && s.median) || 1;
                    }))
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
                        if (!cell || (!cell.barA && !cell.barB)) {
                            return '<td class="text-center" style="color: var(--oflm-muted);">-</td>';
                        }
                        const mainStat = model.modality.key === 'embed' ? cell.barA : (cell.barB || cell.barA);
                        const subStat = model.modality.key === 'embed' ? cell.barB : cell.barA;
                        const isCellHl = hlMem === mc && (!hlNpu || hlNpu === row.npuShort);
                        const ratio = Math.min(1, mainStat.median / maxMain);
                        const alpha = (0.06 + 0.16 * ratio).toFixed(2);
                        const bg = t.dark
                            ? 'rgba(109, 148, 255, ' + alpha + ')'
                            : 'rgba(37, 84, 212, ' + alpha + ')';
                        const subParts = [];
                        if (subStat) subParts.push(fmtCompactRate(subStat.median) + ' ' + model.unit);
                        if (cell.latA) subParts.push(fmtLat(cell.latA.median, model.latUnit));
                        subParts.push('n=' + cell.runs);
                        return '<td class="text-center oflm-matrix-cell' + (isCellHl ? ' is-selected' : '') + '" data-hl-npu="' + esc(row.npuShort) + '" data-hl-mem="' + esc(mc) + '" style="background-color:' + bg + ';">' +
                            '<div class="fw-bold oflm-mono-cell">' + fmtCompactRate(mainStat.median) + ' <span class="small fw-normal" style="color: var(--oflm-muted);">' + esc(model.unit) + '</span></div>' +
                            '<div class="small" style="color: var(--oflm-muted);">' + esc(subParts.join(' · ')) + '</div>' +
                        '</td>';
                    }).join('');

                    return '<tr>' +
                        '<td><div class="fw-semibold">' + esc(row.npuShort) + '</div><div class="small" style="color: var(--oflm-muted);">' + esc(row.npuFull) + '</div></td>' +
                        cells +
                    '</tr>';
                }).join('');

                const clearBtnHtml = (hlMem || hlNpu)
                    ? '<span class="d-inline-flex align-items-center gap-2 small">' +
                          '<span class="oflm-badge neutral"><span>Highlighted: ' + esc(hlNpu ? (hlNpu + ' · ' + hlMem) : hlMem) + '</span></span>' +
                          '<button type="button" class="btn btn-sm btn-link p-0 text-decoration-none" data-clear-hl>Clear</button>' +
                      '</span>'
                    : '';

                const leftSub = model.unit + ' · higher is better';
                const rightSub = model.latUnit + ' · lower is better';

                bodyEl.innerHTML =
                    '<div class="oflm-chart-card">' +
                        selectorCard +
                        '<div class="' + (hasLat ? 'oflm-small-multiples' : '') + ' mt-3">' +
                            '<div class="oflm-subchart">' +
                                '<div class="oflm-subchart-title">Throughput <span class="oflm-subchart-unit">(' + esc(leftSub) + ')</span></div>' +
                                '<div class="oflm-echarts-canvas" data-npu-panel="tp" style="width: 100%; height: ' + chartHeight + 'px;" aria-label="' + esc(model.tag + ' throughput chart') + '"></div>' +
                            '</div>' +
                            (hasLat
                                ? '<div class="oflm-subchart">' +
                                      '<div class="oflm-subchart-title">Latency <span class="oflm-subchart-unit">(' + esc(rightSub) + ')</span></div>' +
                                      '<div class="oflm-echarts-canvas" data-npu-panel="lat" style="width: 100%; height: ' + chartHeight + 'px;" aria-label="' + esc(model.tag + ' latency chart') + '"></div>' +
                                  '</div>'
                                : '') +
                        '</div>' +
                        '<div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mt-3 mb-2">' +
                            '<div class="small fw-semibold" style="color: var(--oflm-muted);">NPU × Memory Configuration Matrix <span class="fw-normal">(click a column or cell to highlight above)</span></div>' +
                            clearBtnHtml +
                        '</div>' +
                        '<div class="oflm-table-wrap mb-0">' +
                            '<table class="oflm-table oflm-npu-matrix">' +
                                '<thead><tr><th>XDNA2 NPU</th>' + matrixHead + '</tr></thead>' +
                                '<tbody>' + matrixRows + '</tbody>' +
                            '</table>' +
                        '</div>' +
                    '</div>';

                const tpEl = bodyEl.querySelector('[data-npu-panel="tp"]');
                if (tpEl) {
                    const c1 = window.echarts.init(tpEl, null, { renderer: 'svg' });
                    const r1 = function () { c1.setOption(buildCombinedThroughputOption(model, activeQuant, npuRows, hlMem, hlNpu), true); };
                    r1();
                    const e1 = { chart: c1, render: r1 };
                    activeInstances.push(e1);
                    childCharts.push(e1);
                }
                const latEl = bodyEl.querySelector('[data-npu-panel="lat"]');
                if (latEl) {
                    const c2 = window.echarts.init(latEl, null, { renderer: 'svg' });
                    const r2 = function () { c2.setOption(buildLatencyRangeOption(model, activeQuant, npuRows, hlMem, hlNpu), true); };
                    r2();
                    const e2 = { chart: c2, render: r2 };
                    activeInstances.push(e2);
                    childCharts.push(e2);
                }
            }

            // Bind dropdown interactions
            const ddEl = bodyEl.querySelector('.oflm-dd');
            const ddToggle = bodyEl.querySelector('[data-dd-toggle]');
            const ddSearch = bodyEl.querySelector('[data-dd-search]');
            const ddList = bodyEl.querySelector('[data-dd-list]');

            function bindOptionClicks() {
                if (!ddList) return;
                ddList.querySelectorAll('[data-pick-model]').forEach(function (btn) {
                    btn.addEventListener('click', function () {
                        activeModelTag = btn.dataset.pickModel;
                        activeQuant = btn.dataset.pickQuant;
                        hlMem = null;
                        hlNpu = null;
                        ddOpen = false;
                        renderWidget();
                    });
                });
            }

            if (ddToggle && ddEl) {
                ddToggle.addEventListener('click', function (ev) {
                    ev.stopPropagation();
                    ddOpen = !ddOpen;
                    ddEl.classList.toggle('is-open', ddOpen);
                    ddToggle.setAttribute('aria-expanded', ddOpen ? 'true' : 'false');
                    if (ddOpen && ddSearch) {
                        setTimeout(function () { ddSearch.focus(); }, 10);
                    }
                });
            }

            if (ddSearch && ddList) {
                ddSearch.addEventListener('click', function (ev) { ev.stopPropagation(); });
                ddSearch.addEventListener('input', function () {
                    filterQuery = ddSearch.value;
                    ddList.innerHTML = buildDropdownListHtml(models);
                    bindOptionClicks();
                });
            }

            bodyEl.querySelectorAll('[data-dd-mod]').forEach(function (btn) {
                btn.addEventListener('click', function (ev) {
                    ev.stopPropagation();
                    filterModality = btn.dataset.ddMod;
                    bodyEl.querySelectorAll('[data-dd-mod]').forEach(function (b) {
                        b.classList.toggle('active', b.dataset.ddMod === filterModality);
                    });
                    if (ddList) {
                        ddList.innerHTML = buildDropdownListHtml(models);
                        bindOptionClicks();
                    }
                });
            });

            bindOptionClicks();

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
