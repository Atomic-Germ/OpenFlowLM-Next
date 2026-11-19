/* OpenFlowLM interactive Apache ECharts runtime with Light/Dark [data-dark-mode] sync */
(function () {
    'use strict';

    function isDarkMode() {
        return document.documentElement.hasAttribute('data-dark-mode');
    }

    function getThemeTokens() {
        const dark = isDarkMode();
        return {
            dark: dark,
            text: dark ? '#e2e6ef' : '#273043',
            muted: dark ? '#a8b0c2' : '#5c667a',
            grid: dark ? 'rgba(168, 176, 194, 0.14)' : 'rgba(39, 48, 67, 0.10)',
            axis: dark ? '#3c4257' : '#cbd5e1',
            tooltipBg: dark ? '#1e202b' : '#ffffff',
            tooltipBorder: dark ? '#323648' : '#dce3ee',
            primary: dark ? '#6d94ff' : '#2b55d6',
            primaryBand: dark ? 'rgba(109, 148, 255, 0.18)' : 'rgba(43, 85, 214, 0.14)',
            secondary: dark ? '#34d399' : '#059669',
            secondaryBand: dark ? 'rgba(52, 211, 153, 0.18)' : 'rgba(5, 150, 105, 0.14)',
            amber: dark ? '#fbbf24' : '#d97706',
            amberBand: dark ? 'rgba(251, 191, 36, 0.18)' : 'rgba(217, 119, 6, 0.14)',
            palette: dark
                ? ['#6d94ff', '#34d399', '#fbbf24', '#f472b6', '#38bdf8', '#a78bfa', '#fb7185']
                : ['#2b55d6', '#059669', '#d97706', '#db2777', '#0284c7', '#7c3aed', '#e11d48']
        };
    }

    const activeInstances = [];
    const allRunsUpdaters = [];

    function makeBandPolygonSeries(name, points, colorFill, yAxisIndex, minClamp) {
        const floor = minClamp !== undefined ? minClamp : 0;
        return {
            name: name + ' ±1σ band',
            type: 'custom',
            yAxisIndex: yAxisIndex || 0,
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

    function makeWhiskerSeries(name, points, strokeColor, yAxisIndex, minClamp) {
        const floor = minClamp !== undefined ? minClamp : 0;
        return {
            name: name + ' ±1σ',
            type: 'custom',
            yAxisIndex: yAxisIndex || 0,
            silent: true,
            z: 4,
            tooltip: { show: false },
            renderItem: function (params, api) {
                const idx = api.value(0);
                const low = api.value(1);
                const high = api.value(2);
                const highPt = api.coord([idx, high]);
                const lowPt = api.coord([idx, low]);
                const cap = 5;
                return {
                    type: 'group',
                    children: [
                        {
                            type: 'line',
                            shape: { x1: highPt[0], y1: highPt[1], x2: lowPt[0], y2: lowPt[1] },
                            style: { stroke: strokeColor, lineWidth: 1.5 }
                        },
                        {
                            type: 'line',
                            shape: { x1: highPt[0] - cap, y1: highPt[1], x2: highPt[0] + cap, y2: highPt[1] },
                            style: { stroke: strokeColor, lineWidth: 1.5 }
                        },
                        {
                            type: 'line',
                            shape: { x1: lowPt[0] - cap, y1: lowPt[1], x2: lowPt[0] + cap, y2: lowPt[1] },
                            style: { stroke: strokeColor, lineWidth: 1.5 }
                        }
                    ]
                };
            },
            data: points.map(function (p, idx) {
                return [idx, Math.max(floor, p.mean - p.std), Math.max(floor, p.mean + p.std)];
            })
        };
    }

    function buildLlmBenchOption(dataset, metric) {
        const t = getThemeTokens();
        const rows = dataset.rows || [];
        const labels = rows.map(function (r) { return r.context; });

        const prefillPts = rows.map(function (r) { return { mean: r.prefill_tps, std: r.prefill_std || 0 }; });
        const decodePts = rows.map(function (r) { return { mean: r.decode_tps, std: r.decode_std || 0 }; });
        const ttftPts = rows.map(function (r) { return { mean: r.ttft_s, std: r.ttft_std || 0 }; });

        const baseGrid = { left: 62, right: 28, top: 76, bottom: 46 };
        let yAxis;
        let series = [];
        let legendData = [];

        if (metric === 'dual') {
            // Single vertical axis with logarithmic scale (avoids dual vertical axes with the same tok/s unit)
            legendData = ['Prefill (tok/s)', 'Decode (tok/s)'];
            yAxis = [
                {
                    type: 'log',
                    min: 4,
                    max: 600,
                    name: 'Throughput (tok/s, log scale)',
                    nameGap: 14,
                    nameTextStyle: { color: t.muted, fontWeight: 600, align: 'left', padding: [0, 0, 0, -28] },
                    axisLabel: { color: t.muted },
                    axisLine: { show: true, lineStyle: { color: t.axis } },
                    splitLine: { lineStyle: { color: t.grid } }
                }
            ];
            series = [
                makeBandPolygonSeries('Prefill', prefillPts, t.primaryBand, 0, 0.5),
                makeWhiskerSeries('Prefill', prefillPts, t.primary, 0, 0.5),
                {
                    name: 'Prefill (tok/s)',
                    type: 'line',
                    yAxisIndex: 0,
                    smooth: 0.2,
                    symbol: 'circle',
                    symbolSize: 7,
                    z: 5,
                    itemStyle: { color: t.primary },
                    lineStyle: { width: 2.5, color: t.primary },
                    data: rows.map(function (r) { return r.prefill_tps; })
                },
                makeBandPolygonSeries('Decode', decodePts, t.secondaryBand, 0, 0.5),
                makeWhiskerSeries('Decode', decodePts, t.secondary, 0, 0.5),
                {
                    name: 'Decode (tok/s)',
                    type: 'line',
                    yAxisIndex: 0,
                    smooth: 0.2,
                    symbol: 'diamond',
                    symbolSize: 8,
                    z: 5,
                    itemStyle: { color: t.secondary },
                    lineStyle: { width: 2.5, color: t.secondary },
                    data: rows.map(function (r) { return r.decode_tps; })
                }
            ];
        } else {
            let pts = decodePts;
            let seriesName = 'Decode Speed (tok/s)';
            let yLabel = 'tok/s';
            let color = t.secondary;
            let band = t.secondaryBand;
            if (metric === 'prefill') {
                pts = prefillPts;
                seriesName = 'Prefill Speed (tok/s)';
                yLabel = 'tok/s';
                color = t.primary;
                band = t.primaryBand;
            } else if (metric === 'ttft') {
                pts = ttftPts;
                seriesName = 'Time to First Token (s)';
                yLabel = 'seconds (lower is faster)';
                color = t.amber;
                band = t.amberBand;
            }
            legendData = [seriesName];
            yAxis = [{
                type: 'value',
                name: yLabel,
                nameGap: 14,
                nameTextStyle: { color: t.muted, fontWeight: 600, align: 'left', padding: [0, 0, 0, -28] },
                axisLabel: { color: t.muted },
                axisLine: { show: true, lineStyle: { color: t.axis } },
                splitLine: { lineStyle: { color: t.grid } }
            }];
            series = [
                makeBandPolygonSeries(seriesName, pts, band, 0, 0),
                makeWhiskerSeries(seriesName, pts, color, 0, 0),
                {
                    name: seriesName,
                    type: 'line',
                    smooth: 0.2,
                    symbol: 'circle',
                    symbolSize: 8,
                    z: 5,
                    itemStyle: { color: color },
                    lineStyle: { width: 2.75, color: color },
                    data: pts.map(function (p) { return p.mean; })
                }
            ];
        }

        return {
            backgroundColor: 'transparent',
            animationDuration: 320,
            grid: baseGrid,
            legend: {
                top: 2,
                left: 'center',
                data: legendData,
                textStyle: { color: t.text, fontSize: 12 }
            },
            toolbox: {
                right: 4,
                top: 0,
                iconStyle: { borderColor: t.muted },
                feature: {
                    saveAsImage: { title: 'Save SVG', name: (dataset.slug || 'oflm-benchmark') + '-' + metric }
                }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: t.tooltipBg,
                borderColor: t.tooltipBorder,
                borderWidth: 1,
                textStyle: { color: t.text, fontSize: 12.5 },
                axisPointer: { type: 'cross', crossStyle: { color: t.muted } },
                formatter: function (params) {
                    if (!params || !params.length) return '';
                    const idx = params[0].dataIndex;
                    const r = rows[idx];
                    if (!r) return '';
                    return [
                        '<div style="font-weight:700;margin-bottom:4px;">Context Length: ' + r.context + '</div>',
                        '<div>Prefill: <strong>' + r.prefill_tps.toFixed(2) + ' ± ' + r.prefill_std.toFixed(2) + ' tok/s</strong></div>',
                        '<div>Decode: <strong>' + r.decode_tps.toFixed(2) + ' ± ' + r.decode_std.toFixed(2) + ' tok/s</strong></div>',
                        '<div>TTFT: <strong>' + r.ttft_s.toFixed(3) + ' ± ' + r.ttft_std.toFixed(3) + ' s</strong></div>'
                    ].join('');
                }
            },
            xAxis: {
                type: 'category',
                name: 'Context',
                nameLocation: 'middle',
                nameGap: 28,
                nameTextStyle: { color: t.muted, fontWeight: 600 },
                boundaryGap: true,
                data: labels,
                axisLabel: { color: t.text, fontWeight: 600 },
                axisLine: { lineStyle: { color: t.axis } }
            },
            yAxis: yAxis,
            series: series
        };
    }

    function aggregateAllRuns(runs) {
        const byKey = new Map();
        (runs || []).forEach(function (r) {
            const isComm = r.source !== 'verified_seed' && r.status !== 'seed';
            const key = r.model + '||' + r.task + '||' + (isComm ? 'community' : 'seed');
            if (!byKey.has(key)) {
                byKey.set(key, {
                    model: r.model,
                    task: r.task,
                    isCommunity: isComm,
                    hardwareSet: new Set(),
                    quantSet: new Set(),
                    stage: r.stage || '',
                    outputUnit: r.output_unit || (r.task === 'Embedding' ? 'txt/s' : 'tok/s'),
                    submissions: 0,
                    iterations: 0,
                    outSum: 0,
                    outN: 0,
                    tokSum: 0,
                    tokN: 0,
                    latSum: 0,
                    latN: 0
                });
            }
            const g = byKey.get(key);
            g.submissions += 1;
            g.iterations += Number(r.runs || 1);
            if (r.hardware) g.hardwareSet.add(r.hardware);
            if (r.quant) g.quantSet.add(r.quant);
            if (r.output_tps != null && Number(r.output_tps) > 0) {
                g.outSum += Number(r.output_tps);
                g.outN += 1;
            }
            if (r.token_tps != null && Number(r.token_tps) > 0) {
                g.tokSum += Number(r.token_tps);
                g.tokN += 1;
            }
            if (r.latency_ms != null && Number(r.latency_ms) > 0) {
                g.latSum += Number(r.latency_ms);
                g.latN += 1;
            }
        });

        return Array.from(byKey.values()).map(function (g) {
            return {
                model: g.model,
                task: g.task,
                isCommunity: g.isCommunity,
                hardware: Array.from(g.hardwareSet).join(' · '),
                quant: Array.from(g.quantSet).join(', '),
                stage: g.stage,
                outputUnit: g.outputUnit,
                submissions: g.submissions,
                iterations: g.iterations,
                outputAvg: g.outN ? (g.outSum / g.outN) : null,
                tokenAvg: g.tokN ? (g.tokSum / g.tokN) : null,
                latencyAvg: g.latN ? (g.latSum / g.latN) : null
            };
        });
    }

    function buildAllRunsOption(dataset, metric) {
        const t = getThemeTokens();
        const aggregated = aggregateAllRuns(dataset.runs || []).filter(function (item) {
            if (metric === 'tokens') return item.tokenAvg != null && item.tokenAvg > 0;
            if (metric === 'latency') return item.latencyAvg != null && item.latencyAvg > 0;
            return item.outputAvg != null && item.outputAvg > 0;
        });

        aggregated.sort(function (a, b) {
            if (metric === 'tokens') return a.tokenAvg - b.tokenAvg;
            if (metric === 'latency') return b.latencyAvg - a.latencyAvg;
            return a.outputAvg - b.outputAvg;
        });

        const labels = aggregated.map(function (item) {
            const suffix = item.isCommunity ? ' (Community avg)' : '';
            return item.model + suffix;
        });

        let xName = 'Headline Throughput (tok/s or texts/s · log scale)';
        if (metric === 'tokens') xName = 'Prompt / Encoder Throughput (tok/s · log scale)';
        if (metric === 'latency') xName = 'First Token / Batched Call Latency (ms · log scale, lower is faster)';

        const commData = aggregated.map(function (item) {
            if (!item.isCommunity) return null;
            if (metric === 'tokens') return Number(item.tokenAvg.toFixed(2));
            if (metric === 'latency') return Number(item.latencyAvg.toFixed(1));
            return Number(item.outputAvg.toFixed(2));
        });

        const repoData = aggregated.map(function (item) {
            if (item.isCommunity) return null;
            if (metric === 'tokens') return Number(item.tokenAvg.toFixed(1));
            if (metric === 'latency') return Number(item.latencyAvg.toFixed(1));
            return Number(item.outputAvg.toFixed(1));
        });

        return {
            backgroundColor: 'transparent',
            animationDuration: 320,
            grid: { left: 215, right: 34, top: 44, bottom: 48 },
            legend: {
                top: 2,
                left: 'center',
                itemGap: 18,
                data: ['Community Run Average', 'Repository Sweep'],
                textStyle: { color: t.text, fontSize: 12 }
            },
            toolbox: {
                right: 4,
                top: 0,
                iconStyle: { borderColor: t.muted },
                feature: {
                    saveAsImage: { title: 'Save SVG', name: 'oflm-all-runs-' + metric }
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
                    const item = aggregated[idx];
                    if (!item) return '';
                    const prov = item.isCommunity
                        ? ('Community Average (' + item.submissions + (item.submissions === 1 ? ' submission, ' : ' submissions, ') + item.iterations + ' iterations)')
                        : ('Repository Sweep (' + item.iterations + ' iterations)');
                    const lines = [
                        '<div style="font-weight:700;margin-bottom:4px;">' + item.model + ' · ' + (item.task === 'LLM' ? 'Chat' : item.task) + '</div>',
                        '<div style="font-size:11.5px;opacity:0.85;margin-bottom:4px;">' + prov + '</div>',
                        '<div>Hardware: <strong>' + item.hardware + '</strong></div>'
                    ];
                    if (item.outputAvg != null) {
                        lines.push('<div>Output Speed: <strong>' + item.outputAvg.toFixed(item.task === 'LLM' ? 2 : 1) + ' ' + item.outputUnit + '</strong> (' + item.stage + ')</div>');
                    }
                    if (item.tokenAvg != null) {
                        lines.push('<div>Token Speed: <strong>' + item.tokenAvg.toFixed(item.task === 'LLM' ? 2 : 0) + ' tok/s</strong></div>');
                    }
                    if (item.latencyAvg != null) {
                        lines.push('<div>Latency / TTFT: <strong>' + item.latencyAvg.toFixed(item.task === 'LLM' ? 0 : 1) + ' ms</strong></div>');
                    }
                    return lines.join('');
                }
            },
            xAxis: {
                type: 'log',
                name: xName,
                nameLocation: 'middle',
                nameGap: 30,
                nameTextStyle: { color: t.muted, fontWeight: 600 },
                axisLabel: { color: t.muted },
                axisLine: { lineStyle: { color: t.axis } },
                splitLine: { lineStyle: { color: t.grid } }
            },
            yAxis: {
                type: 'category',
                data: labels,
                axisLabel: { color: t.text, fontWeight: 600, fontSize: 11.5 },
                axisLine: { lineStyle: { color: t.axis } }
            },
            series: [
                {
                    name: 'Community Run Average',
                    type: 'bar',
                    barGap: '-100%',
                    barMaxWidth: 24,
                    itemStyle: { color: t.primary, borderRadius: [0, 5, 5, 0] },
                    data: commData
                },
                {
                    name: 'Repository Sweep',
                    type: 'bar',
                    barGap: '-100%',
                    barMaxWidth: 24,
                    itemStyle: { color: t.secondary, borderRadius: [0, 5, 5, 0] },
                    data: repoData
                }
            ]
        };
    }

    function buildEmbedSweepOption(dataset, metric) {
        const t = getThemeTokens();
        const batches = dataset.batches || ['n=1', 'n=8', 'n=16', 'n=32', 'n=64'];
        const models = dataset.models || [];

        let yName = 'Throughput (texts/s · log scale)';
        if (metric === 'speedup') yName = 'Batch Speedup vs n=1 (×)';
        if (metric === 'latency') yName = 'Total Batch Latency (ms · log scale)';

        const series = models.map(function (m, idx) {
            const color = t.palette[idx % t.palette.length];
            let values = m.throughput_tps;
            if (metric === 'speedup') values = m.speedup_x;
            if (metric === 'latency') values = m.latency_ms;
            return {
                name: m.tag,
                type: 'line',
                smooth: 0.18,
                symbol: 'circle',
                symbolSize: 7,
                itemStyle: { color: color },
                lineStyle: { width: 2.4, color: color },
                data: values
            };
        });

        return {
            backgroundColor: 'transparent',
            animationDuration: 320,
            grid: { left: 62, right: 26, top: 76, bottom: 46 },
            legend: {
                top: 2,
                left: 0,
                right: 36,
                type: 'scroll',
                itemGap: 14,
                textStyle: { color: t.text, fontSize: 11.5 }
            },
            toolbox: {
                right: 0,
                top: 0,
                iconStyle: { borderColor: t.muted },
                feature: {
                    saveAsImage: { title: 'Save SVG', name: 'oflm-npu-embeddings-' + metric }
                }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: t.tooltipBg,
                borderColor: t.tooltipBorder,
                borderWidth: 1,
                textStyle: { color: t.text, fontSize: 12 }
            },
            xAxis: {
                type: 'category',
                name: 'Batch Size (128 tokens/text)',
                nameLocation: 'middle',
                nameGap: 28,
                nameTextStyle: { color: t.muted, fontWeight: 600 },
                data: batches,
                axisLabel: { color: t.text, fontWeight: 600 },
                axisLine: { lineStyle: { color: t.axis } }
            },
            yAxis: {
                type: metric === 'speedup' ? 'value' : 'log',
                name: yName,
                nameGap: 14,
                nameTextStyle: { color: t.muted, fontWeight: 600, align: 'left', padding: [0, 0, 0, -32] },
                axisLabel: { color: t.muted },
                axisLine: { show: true, lineStyle: { color: t.axis } },
                splitLine: { lineStyle: { color: t.grid } }
            },
            series: series
        };
    }

    function initLlmCard(card) {
        if (!window.echarts) return;
        const canvas = card.querySelector('.oflm-echarts-canvas');
        const payloadEl = card.querySelector('.oflm-chart-payload');
        if (!canvas || !payloadEl) return;
        let dataset;
        try {
            dataset = JSON.parse(payloadEl.textContent);
        } catch (err) {
            return;
        }
        let metric = card.dataset.defaultMetric || 'dual';
        const chart = window.echarts.init(canvas, null, { renderer: 'svg' });
        const render = function () {
            chart.setOption(buildLlmBenchOption(dataset, metric), true);
        };
        render();

        card.querySelectorAll('.oflm-pill-tabs button').forEach(function (btn) {
            btn.addEventListener('click', function () {
                card.querySelectorAll('.oflm-pill-tabs button').forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                metric = btn.dataset.metric;
                render();
            });
        });

        activeInstances.push({ chart: chart, render: render });
    }

    function initAllRunsCard(card) {
        if (!window.echarts) return;
        const canvas = card.querySelector('.oflm-echarts-canvas');
        const payloadEl = card.querySelector('.oflm-chart-payload');
        if (!canvas || !payloadEl) return;
        let dataset;
        try {
            dataset = JSON.parse(payloadEl.textContent);
        } catch (err) {
            return;
        }
        let metric = card.dataset.defaultMetric || 'output';
        const chart = window.echarts.init(canvas, null, { renderer: 'svg' });
        const render = function () {
            chart.setOption(buildAllRunsOption(dataset, metric), true);
        };
        render();

        card.querySelectorAll('.oflm-pill-tabs button').forEach(function (btn) {
            btn.addEventListener('click', function () {
                card.querySelectorAll('.oflm-pill-tabs button').forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                metric = btn.dataset.metric;
                render();
            });
        });

        allRunsUpdaters.push(function (liveRuns) {
            dataset = Object.assign({}, dataset, { runs: liveRuns });
            render();
        });
        activeInstances.push({ chart: chart, render: render });
    }

    function initEmbedCard(card) {
        if (!window.echarts) return;
        const canvas = card.querySelector('.oflm-echarts-canvas');
        const payloadEl = card.querySelector('.oflm-chart-payload');
        if (!canvas || !payloadEl) return;
        let dataset;
        try {
            dataset = JSON.parse(payloadEl.textContent);
        } catch (err) {
            return;
        }
        let metric = card.dataset.defaultMetric || 'throughput';
        const chart = window.echarts.init(canvas, null, { renderer: 'svg' });
        const render = function () {
            chart.setOption(buildEmbedSweepOption(dataset, metric), true);
        };
        render();

        card.querySelectorAll('.oflm-pill-tabs button').forEach(function (btn) {
            btn.addEventListener('click', function () {
                card.querySelectorAll('.oflm-pill-tabs button').forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                metric = btn.dataset.metric;
                render();
            });
        });

        activeInstances.push({ chart: chart, render: render });
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
        register: function (chart, renderFn) {
            activeInstances.push({ chart: chart, render: renderFn });
        },
        syncLiveRuns: function (rawRows) {
            if (!Array.isArray(rawRows) || !rawRows.length) return;
            const mapped = rawRows.map(function (r) {
                const isEmbed = r.task === 'Embedding';
                const mem = r.ram_spec || (r.ram_gb ? r.ram_gb + ' GB RAM' : '');
                return {
                    model: r.model,
                    task: r.task,
                    source: r.source || (r.status === 'seed' ? 'verified_seed' : 'community'),
                    status: r.status,
                    hardware: r.hardware + (mem && !String(r.hardware).includes('GB') ? ' · ' + mem : ''),
                    quant: r.quantization || '',
                    runs: r.runs || 1,
                    stage: r.headline?.ctx_k != null ? (isEmbed ? 'n=' + r.headline.ctx_k : r.headline.ctx_k + 'k ctx') : '',
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
