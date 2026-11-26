/* OpenFlowLM Apache ECharts runtime with Light/Dark [data-dark-mode] sync.
 * Follows Cleveland & McGill perceptual hierarchy, Tufte/Few direct labeling,
 * small multiples, and strict hardware/modality/unit faceting.
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
            primaryBand: dark ? 'rgba(109, 148, 255, 0.16)' : 'rgba(37, 84, 212, 0.12)',
            secondary: dark ? '#34d399' : '#059669',
            secondaryBand: dark ? 'rgba(52, 211, 153, 0.16)' : 'rgba(5, 150, 105, 0.12)',
            amber: dark ? '#fbbf24' : '#d97706',
            amberBand: dark ? 'rgba(251, 191, 36, 0.16)' : 'rgba(217, 119, 6, 0.12)',
            // High-contrast, colorblind-conscious palette for multi-series scaling curves
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
        const raw = String((run && run.hardware) || 'Unknown NPU Platform').trim();
        const ryzenMatch = raw.match(/(AMD\s+Ryzen\s+AI[^,)·]+)/i);
        if (ryzenMatch) return ryzenMatch[1].trim();
        return raw.split('·')[0].trim();
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

    // ==================== FACETED PAIRED-BAR PANELS ====================

    function aggregateFacetRuns(runs) {
        const byFacet = new Map();
        (runs || []).forEach(function (r) {
            const platform = extractHardwarePlatform(r);
            const mod = getModalityMeta(r.task);
            const facetKey = platform + '||' + mod.key;
            if (!byFacet.has(facetKey)) {
                byFacet.set(facetKey, {
                    platform: platform,
                    modality: mod,
                    hardwareDetails: new Set(),
                    modelsMap: new Map()
                });
            }
            const facet = byFacet.get(facetKey);
            if (r.hardware) facet.hardwareDetails.add(r.hardware);

            const isComm = r.source !== 'verified_seed' && r.status !== 'seed';
            const modelKey = r.model + '||' + (isComm ? 'community' : 'seed');
            if (!facet.modelsMap.has(modelKey)) {
                facet.modelsMap.set(modelKey, {
                    model: r.model,
                    task: r.task,
                    modality: mod,
                    isCommunity: isComm,
                    quantSet: new Set(),
                    stage: r.stage || '',
                    outputUnit: r.output_unit || (mod.key === 'embed' ? 'txt/s' : 'tok/s'),
                    submissions: 0,
                    iterations: 0,
                    outSum: 0, outN: 0,
                    tokSum: 0, tokN: 0,
                    singleSum: 0, singleN: 0,
                    latSum: 0, latN: 0,
                    singleLatSum: 0, singleLatN: 0
                });
            }
            const m = facet.modelsMap.get(modelKey);
            m.submissions += 1;
            m.iterations += Number(r.runs || 1);
            if (r.quant) m.quantSet.add(r.quant);
            if (r.output_tps != null && Number(r.output_tps) > 0) {
                m.outSum += Number(r.output_tps);
                m.outN += 1;
            }
            if (r.token_tps != null && Number(r.token_tps) > 0) {
                m.tokSum += Number(r.token_tps);
                m.tokN += 1;
            }
            if (r.single_tps != null && Number(r.single_tps) > 0) {
                m.singleSum += Number(r.single_tps);
                m.singleN += 1;
            }
            if (r.latency_ms != null && Number(r.latency_ms) > 0) {
                m.latSum += Number(r.latency_ms);
                m.latN += 1;
            }
            if (r.single_latency_ms != null && Number(r.single_latency_ms) > 0) {
                m.singleLatSum += Number(r.single_latency_ms);
                m.singleLatN += 1;
            }
        });

        return Array.from(byFacet.values()).map(function (f) {
            const items = Array.from(f.modelsMap.values()).map(function (m) {
                return {
                    model: m.model,
                    task: m.task,
                    modality: m.modality,
                    isCommunity: m.isCommunity,
                    quant: Array.from(m.quantSet).join(', '),
                    stage: m.stage,
                    outputUnit: m.outputUnit,
                    submissions: m.submissions,
                    iterations: m.iterations,
                    outputAvg: m.outN ? (m.outSum / m.outN) : null,
                    tokenAvg: m.tokN ? (m.tokSum / m.tokN) : null,
                    singleAvg: m.singleN ? (m.singleSum / m.singleN) : null,
                    latencyAvg: m.latN ? (m.latSum / m.latN) : null,
                    singleLatencyAvg: m.singleLatN ? (m.singleLatSum / m.singleLatN) : null
                };
            });
            // Sort ascending for ECharts yAxis category (bottom-to-top => fastest at top)
            items.sort(function (a, b) {
                return (a.outputAvg || 0) - (b.outputAvg || 0);
            });
            return {
                platform: f.platform,
                modality: f.modality,
                hardwareDetails: Array.from(f.hardwareDetails),
                items: items
            };
        }).sort(function (a, b) {
            if (a.modality.order !== b.modality.order) return a.modality.order - b.modality.order;
            return a.platform.localeCompare(b.platform);
        });
    }

    function buildFacetGroupedBarOption(facet) {
        const t = getThemeTokens();
        const mod = facet.modality;
        const items = facet.items;
        const isEmbed = mod.key === 'embed';
        const isLlmLike = mod.key === 'llm' || mod.key === 'vlm';
        const unit = isEmbed ? 'txt/s' : (items[0]?.outputUnit || 'tok/s');

        const yLabels = items.map(function (item) {
            const q = item.quant ? ('  [' + item.quant + ']') : '';
            return item.model + q;
        });

        let seriesAName, seriesBName, dataA, dataB;
        if (isEmbed) {
            seriesAName = 'Peak Batched (txt/s)';
            seriesBName = 'Single-Text n=1 (txt/s)';
            dataA = items.map(function (it) { return it.outputAvg != null ? Number(it.outputAvg.toFixed(1)) : null; });
            dataB = items.map(function (it) { return it.singleAvg != null ? Number(it.singleAvg.toFixed(1)) : null; });
        } else if (isLlmLike) {
            seriesAName = 'Prompt Prefill (tok/s)';
            seriesBName = 'Token Decode (tok/s)';
            dataA = items.map(function (it) { return it.tokenAvg != null ? Number(it.tokenAvg.toFixed(2)) : null; });
            dataB = items.map(function (it) { return it.outputAvg != null ? Number(it.outputAvg.toFixed(2)) : null; });
        } else {
            seriesAName = 'Throughput (' + unit + ')';
            seriesBName = null;
            dataA = items.map(function (it) { return it.outputAvg != null ? Number(it.outputAvg.toFixed(2)) : null; });
            dataB = [];
        }

        const allVals = dataA.concat(dataB).filter(function (v) { return v != null && v > 0; });
        const maxVal = allVals.length ? Math.max.apply(null, allVals) : 100;
        const minVal = allVals.length ? Math.min.apply(null, allVals) : 1;
        const useLog = allVals.length > 1 && (maxVal / minVal) >= 12;

        const hasSeriesB = Boolean(seriesBName) && dataB.some(function (v) { return v != null; });
        const series = [];

        const labelStyle = {
            show: true,
            position: 'right',
            distance: 6,
            color: t.text,
            fontSize: 11.5,
            fontWeight: 600,
            fontFamily: 'JetBrains Mono, ui-monospace, monospace'
        };

        series.push({
            name: seriesAName,
            type: 'bar',
            barMaxWidth: 20,
            barGap: '18%',
            itemStyle: { color: t.primary, borderRadius: [0, 4, 4, 0] },
            label: Object.assign({}, labelStyle, {
                formatter: function (p) {
                    if (p.value == null) return '';
                    return fmtCompactRate(p.value) + ' ' + unit;
                }
            }),
            data: dataA
        });

        if (hasSeriesB) {
            series.push({
                name: seriesBName,
                type: 'bar',
                barMaxWidth: 20,
                barGap: '18%',
                itemStyle: { color: t.secondary, borderRadius: [0, 4, 4, 0] },
                label: Object.assign({}, labelStyle, {
                    formatter: function (p) {
                        if (p.value == null) return '';
                        return fmtCompactRate(p.value) + ' ' + unit;
                    }
                }),
                data: dataB
            });
        }

        const xMax = useLog
            ? Math.pow(10, Math.ceil(Math.log10(maxVal * 2.8)))
            : Math.ceil(maxVal * 1.28);

        return {
            backgroundColor: 'transparent',
            animationDuration: 280,
            grid: { left: 205, right: 98, top: 38, bottom: 40 },
            legend: {
                show: hasSeriesB,
                top: 2,
                left: 0,
                itemWidth: 12,
                itemHeight: 10,
                itemGap: 18,
                textStyle: { color: t.text, fontSize: 12, fontWeight: 500 }
            },
            toolbox: {
                right: 0,
                top: 0,
                iconStyle: { borderColor: t.muted },
                feature: {
                    saveAsImage: {
                        title: 'Save SVG',
                        name: 'oflm-' + mod.key + '-' + facet.platform.toLowerCase().replace(/[^a-z0-9]+/g, '-')
                    }
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
                    const idx = params[0].dataIndex;
                    const item = items[idx];
                    if (!item) return '';
                    const prov = item.isCommunity
                        ? ('Community avg (' + item.iterations + ' runs)')
                        : ('Repo sweep (' + item.iterations + ' runs)');
                    const lines = [
                        '<div style="font-weight:700;margin-bottom:3px;">' + esc(item.model) + ' · ' + esc(mod.label) + '</div>',
                        '<div style="font-size:11.5px;opacity:0.82;margin-bottom:4px;">' + esc(facet.platform) + ' · ' + esc(prov) + '</div>'
                    ];
                    if (isEmbed) {
                        if (item.outputAvg != null) {
                            lines.push('<div>Peak Batched (' + esc(item.stage) + '): <strong>' + fmtNum(item.outputAvg, 1) + ' txt/s</strong></div>');
                        }
                        if (item.singleAvg != null) {
                            const gain = item.outputAvg ? (item.outputAvg / item.singleAvg).toFixed(1) + '×' : '';
                            lines.push('<div>Single-Text (n=1): <strong>' + fmtNum(item.singleAvg, 1) + ' txt/s</strong>' + (gain ? ' (' + gain + ' batch gain)' : '') + '</div>');
                        }
                        if (item.tokenAvg != null) {
                            lines.push('<div>Encoder Tokens: <strong>' + fmtNum(item.tokenAvg, 0) + ' tok/s</strong></div>');
                        }
                        if (item.singleLatencyAvg != null) {
                            lines.push('<div>Single-Text Latency: <strong>' + fmtNum(item.singleLatencyAvg, 1) + ' ms</strong></div>');
                        }
                    } else {
                        if (item.tokenAvg != null) {
                            lines.push('<div>Prompt Prefill (' + esc(item.stage) + '): <strong>' + fmtNum(item.tokenAvg, 2) + ' tok/s</strong></div>');
                        }
                        if (item.outputAvg != null) {
                            lines.push('<div>Token Decode (' + esc(item.stage) + '): <strong>' + fmtNum(item.outputAvg, 2) + ' tok/s</strong></div>');
                        }
                        if (item.latencyAvg != null) {
                            lines.push('<div>Time to First Token: <strong>' + (item.latencyAvg / 1000).toFixed(2) + ' s</strong></div>');
                        }
                    }
                    return lines.join('');
                }
            },
            xAxis: {
                type: useLog ? 'log' : 'value',
                min: useLog ? (minVal < 2 ? 1 : undefined) : 0,
                max: xMax,
                name: unit + (useLog ? ' (log scale · higher is faster)' : ' (higher is faster)'),
                nameLocation: 'middle',
                nameGap: 24,
                nameTextStyle: { color: t.muted, fontWeight: 600, fontSize: 11.5 },
                axisLabel: { color: t.muted, fontSize: 11 },
                axisLine: { show: false },
                axisTick: { show: false },
                splitLine: { lineStyle: { color: t.grid } }
            },
            yAxis: {
                type: 'category',
                data: yLabels,
                axisLabel: { color: t.text, fontWeight: 600, fontSize: 11.5 },
                axisLine: { lineStyle: { color: t.axis } },
                axisTick: { show: false }
            },
            series: series
        };
    }

    function renderFacetedRunPanels(gridEl, runs, existingEntries) {
        disposeInstances(existingEntries);
        const created = [];
        const facets = aggregateFacetRuns(runs);
        if (!facets.length) {
            gridEl.innerHTML = '<div class="oflm-chart-card"><p class="small mb-0" style="color: var(--oflm-muted);">No benchmark runs match the current selection.</p></div>';
            return created;
        }

        gridEl.innerHTML = '';
        facets.forEach(function (facet) {
            const mod = facet.modality;
            const isEmbed = mod.key === 'embed';
            const isLlmLike = mod.key === 'llm' || mod.key === 'vlm';
            const hasCommunity = facet.items.some(function (it) { return it.isCommunity; });
            const hasSeed = facet.items.some(function (it) { return !it.isCommunity; });
            const provText = hasCommunity && hasSeed
                ? 'Repo sweeps & community avg'
                : (hasCommunity ? 'Community average' : 'Repository sweep');
            const metricDesc = isEmbed
                ? 'Peak Batched vs. Single-Text (n=1) · txt/s'
                : (isLlmLike ? 'Prompt Prefill vs. Token Decode · tok/s (1k ctx)' : 'Measured throughput');
            const detailLine = facet.hardwareDetails[0] || facet.platform;
            const chartHeight = Math.max(158, facet.items.length * 56 + 74);

            const card = document.createElement('div');
            card.className = 'oflm-chart-card oflm-facet-card';
            card.innerHTML =
                '<div class="oflm-chart-header">' +
                    '<div>' +
                        '<div class="oflm-chart-meta mb-1">' +
                            '<span class="oflm-badge ' + esc(mod.badgeClass) + '"><span class="material-icons" aria-hidden="true">' + esc(mod.icon) + '</span><span>' + esc(mod.label) + '</span></span>' +
                            '<span class="oflm-badge neutral"><span class="material-icons" aria-hidden="true">memory</span><span>' + esc(facet.platform) + '</span></span>' +
                            '<span class="oflm-badge ' + (hasCommunity ? 'llm' : 'ok') + '"><span>' + esc(provText) + '</span></span>' +
                        '</div>' +
                        '<h3 class="oflm-chart-title">' + esc(mod.label) + ' on ' + esc(facet.platform) + '</h3>' +
                        '<p class="oflm-chart-subtitle">' + esc(metricDesc) + ' · ' + esc(detailLine) + '</p>' +
                    '</div>' +
                '</div>' +
                '<div class="oflm-echarts-canvas" style="width: 100%; height: ' + chartHeight + 'px;" aria-label="' + esc(mod.label + ' benchmark chart on ' + facet.platform) + '"></div>';

            gridEl.appendChild(card);
            const canvas = card.querySelector('.oflm-echarts-canvas');
            const chart = window.echarts.init(canvas, null, { renderer: 'svg' });
            const renderFn = function () {
                chart.setOption(buildFacetGroupedBarOption(facet), true);
            };
            renderFn();
            const entry = { chart: chart, render: renderFn };
            activeInstances.push(entry);
            created.push(entry);
        });

        return created;
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

    function initAllRunsCard(container) {
        if (!window.echarts) return;
        const gridEl = container.querySelector('.oflm-facet-grid');
        const payloadEl = container.querySelector('.oflm-chart-payload');
        if (!gridEl || !payloadEl) return;

        let dataset;
        try {
            dataset = JSON.parse(payloadEl.textContent);
        } catch (_err) {
            return;
        }

        let panelEntries = renderFacetedRunPanels(gridEl, dataset.runs || [], []);

        allRunsUpdaters.push(function (liveRuns) {
            dataset = Object.assign({}, dataset, { runs: liveRuns });
            panelEntries = renderFacetedRunPanels(gridEl, dataset.runs || [], panelEntries);
        });
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
        document.querySelectorAll('[data-oflm-allruns-chart]').forEach(initAllRunsCard);
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
        renderFacetedRunPanels: renderFacetedRunPanels,
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
                    source: r.source || (r.status === 'seed' ? 'verified_seed' : 'community'),
                    status: r.status,
                    hardware_platform: r.cpu || extractHardwarePlatform(r),
                    hardware: r.hardware + (mem && !String(r.hardware).includes('GB') ? ' · ' + mem : ''),
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
