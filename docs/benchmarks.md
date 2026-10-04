---
layout: page
title: "Benchmarks"
permalink: /benchmarks/
description: "Transparent Ryzen AI telemetry for chat, multimodal, and agent workloads."
sections:
  - type: hero
    kicker: "Benchmarks"
    title: "Measured on real laptops."
    body: |
      Every OpenFlowLM release is validated on Ryzen™ AI NPUs.
      We publish the results in `docs/benchmarks` so teams can compare apples-to-apples.

      The runtime extends AMD’s native 2K context limit for long-context LLMs and
      VLMs. All published numbers are throughput and latency (TTFT, prefill
      tok/s, decode tok/s) measured with `oflm bench`; we do not publish energy
      or power-draw figures, because `oflm bench` does not instrument them.
    ctas:
      - label: "View benchmark docs"
        href: "/docs/benchmarks/"
        style: primary
    right:
      metrics:
        - label: "GPT-OSS 20B"
          value: "18.2 tok/s"
          desc: "decode @ 1k · AMD Ryzen™ AI 9 HX 370"
        - label: "Qwen 3 0.6B"
          value: "2,003 tok/s"
          desc: "prefill @ 2k · AMD Ryzen™ AI 9 HX 370"
        - label: "Gemma 3 1B"
          value: "1,785 tok/s"
          desc: "prefill @ 16k · AMD Ryzen™ AI 7 350"

  # - type: media
  #   variant: alt
  #   kicker: "Llama 3.2 on Ryzen™ AI"
  #   title: "Prefill + decoding throughput across 256K tokens"
  #   gallery:
  #     - src: "/assets/bench/llama3-2-3b.png"
  #       alt: "Llama 3.2 3B prefill throughput on the Ryzen AI NPU"
  #     - src: "/assets/bench/llama3-2-3b-decoding.png"
  #       alt: "Llama 3.2 3B decoding throughput chart"
  #   items:
  #     - heading: "Long-context ready"
  #       body: "Screenshots are taken straight from the long-context regression run in `docs/benchmarks`."
  #     - heading: "Laptop-verified"
  #       body: "Instrumentation overlays remain visible from the Ryzen™ AI 9 HX 370 Halo reference design capture."

  - type: media
    variant: alt
    kicker: "Gemma 3 4B"
    title: "Prefill and decode throughput on the NPU"
    media:
      src: "/assets/bench/gemma3-4b.png"
      alt: "Gemma 3 4B benchmark overview of prefill and decoding throughput on the NPU"
      title: "Throughput on Ryzen™ AI silicon"

---

