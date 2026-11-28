---
title: "Submit Benchmark Results"
description: "Share AMD Ryzen AI NPU benchmark runs with one CLI flag or upload a saved CSV file."
url: "/benchmarks/submit/"
aliases:
  - /benchmarks/community/submit/
kicker: "Share Your Speeds"
icon: "upload_file"
community_bench: true
ctas:
  - text: "Back to Benchmarks"
    url: "/benchmarks/"
    icon: "speed"
  - text: "Browse Models"
    url: "/models/"
    icon: "view_module"
    outline: true
---

<div data-cb-page="submit">

## One-Command CLI Upload

Pass `--submit` to `oflm bench` or `oflm bench-embed` while your PC is plugged in and idle:

```bash
oflm bench qwen3.5:4b --submit
oflm bench-embed bge-base:en-v1.5 --max-batch 128 --bench-iterations 3 --submit
```

On first upload, OpenFlowLM prompts once before attaching processor model, RAM size, operating system, and a random install ID.

## Manual CSV Upload

Load or paste a saved `bench_*.csv` or `bench_embed_*.csv` file from an offline machine:

<div class="oflm-card mt-3">
<form id="cb-submit">
<div class="oflm-form-grid">
<label>
Model Tag *
<input type="text" id="s-model" required placeholder="e.g. qwen3.5:4b or bge-base:en-v1.5" />
</label>
<label>
Capability *
<select id="s-task" required>
<option value="LLM">Chat &amp; Reasoning</option>
<option value="Embedding">Embeddings</option>
<option value="VLM">Vision</option>
<option value="ASR">Speech-to-Text</option>
<option value="Image">Image Generation</option>
</select>
</label>
<label>
Laptop or Mini-PC Model *
<input type="text" id="s-hardware" required placeholder="e.g. ASUS Zenbook S 16 (Ryzen AI 9 HX 370)" />
</label>
<label>
Processor *
<input type="text" id="s-cpu" required placeholder="e.g. AMD Ryzen AI 9 HX 370" />
</label>
<label>
System Memory (GB) *
<input type="number" id="s-ram" min="8" max="512" step="1" required placeholder="32" />
</label>
<label>
Runtime *
<input type="text" id="s-backend" required placeholder="e.g. NPU Runtime (auto-filled from CLI)" value="OpenFlowLM NPU" />
</label>
<label>
Quantization *
<input type="text" id="s-quant" required placeholder="e.g. Q4_1, Q4_K_M, Q8_0, BF16" />
</label>
<label>
OpenFlowLM Version *
<input type="text" id="s-version" required placeholder="0.1.0" value="0.1.0" />
</label>
<label>
Operating System *
<input type="text" id="s-os" required placeholder="e.g. Windows 11 or Ubuntu 24.04" />
</label>
<label>
Iterations per Stage
<input type="number" id="s-runs" min="1" max="50" step="1" value="3" />
</label>
<label class="oflm-form-full">
Load Local CSV File (optional)
<input type="file" id="s-file" accept=".csv,text/csv,text/plain" />
</label>
<label class="oflm-form-full">
Benchmark CSV Output *
<textarea id="s-csv" rows="8" required placeholder="# Paste the full contents of bench_<model>_<date>.csv here..."></textarea>
</label>
</div>

<div class="d-flex flex-wrap align-items-center gap-3 mt-3">
<button type="submit" class="btn btn-primary">Submit Benchmark Run</button>
<div id="cb-msg" class="small" style="color: var(--oflm-muted);" aria-live="polite"></div>
</div>
</form>
</div>

</div>
