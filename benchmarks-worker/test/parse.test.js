import { test } from "node:test";
import assert from "node:assert/strict";
import { stagesFromCsv, validateSubmission } from "../src/parse.js";

const LLM_CSV = [
  "context_length_k,ttft_avg_s,ttft_std_s,ttft_min_s,ttft_max_s,prefill_avg_toks_per_s,prefill_std_toks_per_s,prefill_min_toks_per_s,prefill_max_toks_per_s,decoding_avg_toks_per_s,decoding_std_toks_per_s,decoding_min_toks_per_s,decoding_max_toks_per_s",
  "1,0.210000,0.010000,0.200000,0.220000,820.00,20.00,800.00,840.00,34.70,4.10,32.40,36.50",
  "2,0.410000,0.020000,0.390000,0.430000,790.00,22.00,770.00,810.00,33.10,3.80,31.00,35.00",
].join("\n");

test("parses oflm bench CSV", () => {
  const { isEmbed, stages } = stagesFromCsv(LLM_CSV);
  assert.equal(isEmbed, false);
  assert.equal(stages.length, 2);
  assert.equal(stages[0].ctx_k, 1);
  assert.ok(Math.abs(stages[0].decode_avg - 34.7) < 1e-9);
});

test("rejects avg outside min/max", () => {
  const bad = LLM_CSV.replace("34.70", "34.70").replace("32.40", "40.00");
  assert.throws(() => stagesFromCsv(bad), /avg_below_min/);
});

test("rejects non power-of-two context", () => {
  const bad = LLM_CSV.replace("\n1,", "\n3,");
  assert.throws(() => stagesFromCsv(bad), /power_of_two/);
});

test("parses bench-embed CSV with # provenance", () => {
  const csv = ["# oflm bench-embed", "# model=bge-x", "batch,batched_avg_s,batched_std_s,batched_min_s,batched_max_s,texts_avg_per_s,texts_std_per_s,texts_min_per_s,texts_max_per_s,tokens_avg_per_s,tokens_std_per_s,tokens_min_per_s,tokens_max_per_s,looped_avg_s,looped_std_s,batch_speedup",
    "16,0.07,0.01,0.06,0.08,228.57,30.00,200.00,250.00,12500.00,300.00,12000.00,13000.00,0.40,0.02,5.71"].join("\n");
  const { isEmbed, stages } = stagesFromCsv(csv);
  assert.equal(isEmbed, true);
  assert.equal(stages[0].ctx_k, 16);
});

test("validateSubmission requires model/csv/id", () => {
  assert.ok(validateSubmission({}).errs.length >= 3);
  assert.deepEqual(validateSubmission({ model: "m", submission_id: "x", csv: LLM_CSV }).errs, []);
});
