import { test } from "node:test";
import assert from "node:assert/strict";
import { validateSubmission, UUID_V4_RE, stagesFromCsv } from "../src/parse.js";

const LLM_CSV = [
  "context_length_k,ttft_avg_s,ttft_std_s,ttft_min_s,ttft_max_s,prefill_avg_toks_per_s,prefill_std_toks_per_s,prefill_min_toks_per_s,prefill_max_toks_per_s,decoding_avg_toks_per_s,decoding_std_toks_per_s,decoding_min_toks_per_s,decoding_max_toks_per_s",
  "1,0.210000,0.010000,0.200000,0.220000,820.00,20.00,800.00,840.00,34.70,4.10,32.40,36.50",
].join("\n");
const EMBED_CSV = [
  "# oflm bench-embed", "# model=bge-x",
  "batch,batched_avg_s,batched_std_s,batched_min_s,batched_max_s,texts_avg_per_s,texts_std_per_s,texts_min_per_s,texts_max_per_s,tokens_avg_per_s,tokens_std_per_s,tokens_min_per_s,tokens_max_per_s,looped_avg_s,looped_std_s,batch_speedup",
  "16,0.07,0.01,0.06,0.08,228.57,30.00,200.00,250.00,12500.00,300.00,12000.00,13000.00,0.40,0.02,5.71",
].join("\n");
const UUID = "550e8400-e29b-41d4-a716-446655440000";

test("accepts a uuid v4 install_id with survey opt-in", () => {
  assert.match(UUID, UUID_V4_RE);
  const { errs } = validateSubmission({ model: "m", submission_id: "x", csv: LLM_CSV, install_id: UUID, survey_opt_in: true });
  assert.deepEqual(errs, []);
});

test("rejects a non-uuid install_id", () => {
  const { errs } = validateSubmission({ model: "m", submission_id: "x", csv: LLM_CSV, install_id: "not-a-uuid" });
  assert.ok(errs.includes("install_id_invalid"));
});

test("embed CSV keeps looped + speedup", () => {
  const { isEmbed, stages } = stagesFromCsv(EMBED_CSV);
  assert.equal(isEmbed, true);
  assert.equal(stages[0].ctx_k, 16);
  assert.ok(Math.abs(stages[0].decode_avg - 228.57) < 1e-9);
  assert.ok(Math.abs(stages[0].prefill_avg - 12500) < 1e-9);
  assert.ok(Math.abs(stages[0].speedup - 5.71) < 1e-9);
});
