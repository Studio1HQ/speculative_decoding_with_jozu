<div align="center">

<img src="https://raw.githubusercontent.com/Studio1HQ/Studio1HQ/main/assets/banners/speculative_decoding_with_jozu.webp" alt="Speculative Decoding on Jozu RIC: Speed up local inference with a draft model on a Jozu Rapid Inference Container" width="100%" />

<br />
<br />

# Speculative Decoding on Jozu RIC

### Speed up local inference with a draft model on a Jozu Rapid Inference Container

A notebook that packages a Qwen3-30B target model and a Qwen3-0.6B draft model in one ModelKit, serves them with a Jozu RIC and compares speed with and without speculative decoding.

<br />

[![Built by Studio1](https://img.shields.io/badge/Built_by_Studio1-f97316?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAgMTYwIiBmaWxsPSJub25lIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTAgMzVDMCAxNi42ODA3IDE0LjA3NDQgMS42NDg0NCAzMiAwLjEyNjk1M1YxNDFINTEuMDUyM0M1Mi40MzA4IDEzNy4zNDQgNTQuNzQ0NiAxMzMuNjg5IDU3Ljk5MzggMTMwLjAzM0M2MS4xNDQ2IDEyNi41MDggNjQuNjg5MiAxMjMuNDQgNjguNjI3NyAxMjAuODI5QzcyLjU2NjIgMTE4LjM0OSA3Ni4zNTY5IDExNi43ODIgODAgMTE2LjEyOVY4Ni43NTQ0Qzc1LjQ3MDggODcuOTI5MiA3MS4zODQ2IDg5LjY5MTkgNjcuNzQxNSA5Mi4wNDE1QzY0IDk0LjUyMjUgNjAuNzUwOCA5Ny40NiA1Ny45OTM4IDEwMC44NTRDNTYuNTM0NSAxMDIuNzIgNTUuMjU0NSAxMDQuNjU5IDU0LjE1MzkgMTA2LjY3MVYwSDEyNUMxNDQuMzMgMCAxNjAgMTUuNjY5OSAxNjAgMzVWMTI1QzE2MCAxNDQuMzMgMTQ0LjMzIDE2MCAxMjUgMTYwSDEyNFYxOEgxMDQuOTQ4QzEwMy41NjkgMjEuNjgxNiAxMDEuMjU1IDI1LjM2MjggOTguMDA2MiAyOS4wNDQ0Qzk0Ljg1NTQgMzIuNTk0MiA5MS4zMTA4IDM1LjY4NDEgODcuMzcyMyAzOC4zMTRDODMuNDMzOCA0MC44MTIgNzkuNjQzMSA0Mi4zODk2IDc2IDQzLjA0NzRWNzIuNjMwNEM4MC41MjkyIDcxLjQ0NzMgODQuNjE1NCA2OS42NzI0IDg4LjI1ODUgNjcuMzA1N0M5MiA2NC44MDc2IDk1LjI0OTIgNjEuODQ5MSA5OC4wMDYyIDU4LjQzMDdDOTkuNDY1NSA1Ni41NTEzIDEwMC43NDUgNTQuNTk4NiAxMDEuODQ2IDUyLjU3MjNWMTYwSDM1QzE1LjY3IDE2MCAwIDE0NC4zMyAwIDEyNVYzNVoiIGZpbGw9IiNmZmZmZmYiLz48L3N2Zz4K)](https://studio1hq.com)
[![Built with Jozu](https://img.shields.io/badge/Built_with_Jozu-141414?style=for-the-badge)](https://jozu.com)

[Jozu docs](https://jozu.com/docs) &nbsp;·&nbsp; [Jozu](https://jozu.com) &nbsp;·&nbsp; [KitOps](https://kitops.org) &nbsp;·&nbsp; [More examples](https://github.com/Studio1HQ) &nbsp;·&nbsp; [studio1hq.com](https://studio1hq.com)

</div>

<br />

## Overview

This repo accompanies the blog post "Accelerate Local AI Inference with speculative decoding on Jozu RIC". The notebook (`main.ipynb`) contains the commands from the article.

You download a quantized Qwen3-30B-A3B target model and a small Qwen3-0.6B draft model, package both in one [KitOps](https://kitops.org) ModelKit (the draft is a `parts` entry on the model), push it to [Jozu Hub](https://jozu.com) and run it with a Jozu Rapid Inference Container (RIC). You run `llama-server` with and without a draft model and time the generation.

## Features

- **Two models, one ModelKit**: The target GGUF is the Kitfile `model` and the draft GGUF is listed under `parts`.
- **Speculative decoding**: `llama-server` runs with `-md` (draft model) and `--draft-n 4`.
- **Baseline run**: The same container command without the draft flags.
- **Speed test**: A `curl` loop against `/completion` generates 2048 tokens three times and prints the `timings` field.
- **Custom entrypoint**: The RIC is started with `--entrypoint /app/llama-server` so you control the flags.

## Tech stack

| Layer | Tool |
| --- | --- |
| Target model | [Qwen3-30B-A3B-Instruct-2507 GGUF (UD-TQ1_0)](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) |
| Draft model | [Qwen3-0.6B GGUF (Q8_0)](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) |
| Packaging | [KitOps](https://kitops.org) CLI and `kitops` Python library |
| Registry and runtime | [Jozu Hub](https://jozu.ml) and Jozu Rapid Inference Container (llama.cpp) |
| Runtime | Python 3.11+, Docker, Jupyter |

## Getting started

### Prerequisites

- Python 3.11 or later and Jupyter
- Docker
- A Jozu Hub account at [jozu.ml](https://jozu.ml)
- Homebrew on Linux or macOS (the notebook installs it if missing)
- A machine with an NVIDIA GPU and the NVIDIA container toolkit for the `--gpus all` Docker command

### Installation

```bash
git clone https://github.com/Studio1HQ/speculative_decoding_with_jozu.git
cd speculative_decoding_with_jozu
brew tap kitops-ml/kitops
brew install kitops
pip install kitops huggingface_hub
```

### Environment variables

No environment variables. The notebook authenticates with the email and password of your Jozu Hub account and uses placeholders you replace.

| Placeholder | Description |
| --- | --- |
| `email-prefix-here` | The part of your Jozu Hub email before the `@` (for `jack123@gmail.com` use `jack123`). Keep tags lowercase. |
| `full-email-here`, `password-here` | Jozu Hub credentials passed to `UserCredentials` in the notebook. |

### Run

Log in to Jozu Hub in a terminal (you are prompted for email and password), then open the notebook:

```bash
kit login jozu.ml
jupyter notebook main.ipynb
```


## How it works

1. Download both models into `./Qwen3-family/` with `hf_hub_download`.
2. Build the Kitfile (`package.name` `speculative-decoding-with-jozu`) and push to `jozu.ml/email-prefix-here/speculative-decoding-with-jozu:latest`.
3. Run with speculative decoding:
   ```bash
   sudo docker run -it --rm -p 8000:8000 --gpus all \
     --entrypoint /app/llama-server \
     "jozu.ml/email-prefix-here/speculative-decoding-with-jozu/llama-cpp-cuda:latest" \
     -m /Qwen3-family/Qwen3-30B-A3B-Instruct-2507-UD-TQ1_0.gguf \
     -md /Qwen3-family/Qwen3-0.6B-Q8_0.gguf \
     --jinja --chat-template-kwargs '{"enable_thinking": false}' \
     --host 0.0.0.0 --port 8000 -c 40965 --draft-n 4
   ```
4. Run the baseline by dropping `-md` and `--draft-n 4`.
5. Time each setup:
   ```bash
   for i in 1 2 3; do
     curl -s http://localhost:8000/completion -d '{"prompt": "Write an elaborate step by step technical guide on how to pre-train, finetune, and inference large language models.", "n_predict": 2048, "temperature": 0}' \
       | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['timings'], indent=2))"
   done
   ```

## Resources

- [Jozu documentation](https://jozu.com/docs): Jozu Hub and Rapid Inference Container guides.
- [Jozu](https://jozu.com): Product site.
- [KitOps documentation](https://kitops.org/docs/overview/): ModelKit and Kitfile reference.

- [llama.cpp](https://github.com/ggml-org/llama.cpp): The server used inside the RIC.

<br />

---

<div align="center">

<a href="https://studio1hq.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Studio1HQ/website/main/brand/studio1-logo-on-dark.svg" />
    <img src="https://raw.githubusercontent.com/Studio1HQ/website/main/brand/studio1-logo-on-light.svg" alt="Studio1" width="160" />
  </picture>
</a>

Built by **[Studio1](https://studio1hq.com)**, the technical content and DevRel partner for devtool teams. Made with [Jozu](https://jozu.com).<br />
Browse more open-source examples at [github.com/Studio1HQ](https://github.com/Studio1HQ).

[![Website](https://img.shields.io/badge/studio1hq.com-0b0807?style=for-the-badge&logo=googlechrome&logoColor=white)](https://studio1hq.com)
[![Book a call](https://img.shields.io/badge/Book_a_strategy_call-f97316?style=for-the-badge&logo=googlecalendar&logoColor=white)](https://cal.com/studio1/collab)
[![X](https://img.shields.io/badge/%40Studio1HQ-141414?style=for-the-badge&logo=x&logoColor=white)](https://x.com/Studio1HQ)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0a66c2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/studio1hq/)

<sub><a href="https://studio1hq.com">studio1hq.com</a> · Technical content, docs, video and DevRel for devtools</sub>

</div>
