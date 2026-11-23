# Qwen3.8 PR4A profiling status

**No accepted current performance report.** This iteration repairs instrumentation
and CPU-testable capture validation only. It does not run a new NPU performance
campaign or the long synthetic benchmark, implement PR4B, or establish a speedup.

The current contracts, review-resolution matrix, reproduction commands and
limitations are in [the instrumentation gate](qwen38-pipeline-instrumentation.md).

[The historical numerical JSON](qwen38-deltanet-pipeline-performance.json) remains
unchanged for traceability. Its resolution matrix is historical, not the current
gate result. Its raw captures lack the current build/snapshot validation and must
not be presented as fresh evidence from the repaired implementation. Raw files
are never relabeled, repaired or overwritten to make them pass validation.

Phase-specific CPU/wait attribution is **NOT MEASURED**. Command-residence
intervals can overlap under submit-ahead; summing them does not identify exclusive
device time, CPU use, NPU utilization, or a wait-bound bottleneck. Unattributed
residual time is not assigned to an operation.

No end-to-end gain, instrumentation-overhead estimate, updated bottleneck ranking,
or PR4B recommendation is derived from those historical measurements. Further
NPU profiling and PR4B require a subsequent review.
