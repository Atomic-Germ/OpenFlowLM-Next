---
layout: docs
title: SmolVLA
nav_order: 13
parent: Models
---

> 🚧 **Not yet supported in OpenFlowLM 0.1.0.**
>
> SmolVLA has **no tag, no engine, no kernels and no build support** in this
> repository. There is nothing to `oflm pull` or `oflm run` yet, and the figures
> that appeared on this page in earlier revisions were not produced by anything
> in the tree. This page is a placeholder for a future release.

## 🧩 Planned Model Card: [SmolVLA](https://huggingface.co/lerobot/smolvla_base)

- **Type:** Vision-Language-Action
- **Think:** No
- **Tool Calling Support:** No  
- **Base Model:** [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)
- **Max Camera Input:** 3 images

📝 **Note:**

- SmolVLA is a robotics policy model that maps camera images and language
  instructions directly to robot actions. It is not a chat or embedding model,
  so it would not run in OFLM's standard CLI or Server modes even once
  supported -- it needs a separate inference path.
- No benchmark data is published for it, because there is no code in this
  repository that can produce one.

---

See the [support-status matrix](/docs/models/#open-vs-closed-support-status) for
what *is* supported today.
