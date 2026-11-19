---
title: "Open WebUI"
description: "Connect Open WebUI Desktop or Docker to OpenFlowLM for chat, multi-model queuing, search, and RAG."
icon: "dns"
weight: 40
---

## Run Open WebUI with OpenFlowLM

### Option 1: Open WebUI Desktop

1. Install [Open WebUI Desktop](https://github.com/open-webui/desktop/releases) and create your local admin account.
2. Open **User icon -> Admin Panel -> Settings -> Connections**, disable **Ollama API**, and add an **OpenAI API** connection:
    - **Name:** `OpenFlowLM`
    - **Base URL:** `http://127.0.0.1:52625/v1`
    - **API Key:** `oflm`
3. Start `oflm serve llama3.2:1b` (or any tag from [`oflm add`](/docs/instructions/cli/#add-a-model-oflm-add)) and begin chatting.

> [!TIP]
> Under **Settings -> Interface**, disable `Title`, `Follow Up`, `Tags`, `Retrieval Query`, and `Web Search Query` generation so Open WebUI does not dispatch extra NPU calls after every turn.

### Option 2: Open WebUI in Docker

1. Save `docker-compose.yaml` in a new `open-webui/` directory ([OpenFlowLM](/docs/install/) runs on the host):

    ```yaml
    services:
      open-webui:
        image: ghcr.io/open-webui/open-webui:main
        container_name: open-webui
        ports:
          - "3000:8080"
        extra_hosts:
          - "host.docker.internal:host-gateway"
        volumes:
          - open-webui-data:/app/backend/data
        environment:
          - OPENAI_API_BASE_URL=http://host.docker.internal:52625/v1
          - OPENAI_API_KEY=dummy-key
          - OLLAMA_BASE_URL=
          - WEBUI_AUTH=false
          - WEBUI_SECRET_KEY=dummysecretkey
          - ENABLE_TITLE_GENERATION=false
          - ENABLE_FOLLOW_UP_GENERATION=false
          - ENABLE_TAGS_GENERATION=false
          - ENABLE_RETRIEVAL_QUERY_GENERATION=false
          - ENABLE_IMAGE_PROMPT_GENERATION=false
          - ENABLE_WEB_SEARCH=false
          - ENABLE_SEARCH_QUERY_GENERATION=false
        restart: unless-stopped

    volumes:
      open-webui-data:
    ```

    > [!NOTE]
    > On Linux, `127.0.0.1` on the host is unreachable via `host.docker.internal` on a bridge network. Either set `network_mode: host` (with `OPENAI_API_BASE_URL=http://127.0.0.1:52625/v1`) or start `oflm serve --host 0.0.0.0 --cors 0` behind firewall rules.

2. Launch the container and server, then open `http://127.0.0.1:3000`:

    ```bash
    docker compose up -d
    oflm serve llama3.2:1b
    # Manage container lifecycle:
    docker compose stop       # pause
    docker compose down       # remove container (keeps volume)
    docker compose down -v    # remove container + volume
    docker compose pull       # update image
    ```

---

## Workflows in Open WebUI

### Multi-Model Comparison

1. Select a primary model in the top-left picker and click **+** to add comparison models (such as Qwen3.5 4B and Gemma 3 4B).
2. Submit a prompt. OpenFlowLM queues the requests and runs each model sequentially.
3. Pick a preferred reply or click **Merge Responses** to synthesize them.

<a id="example-agentic-ai-web-search-with-openflowlm"></a>

### Agentic Web Search (Google PSE)

1. In the [Google PSE Control Panel](https://programmablesearchengine.google.com/controlpanel/all), create an engine with **Search the entire web**, copy its **Search Engine ID**, and generate a **Programmatic Access API Key**.
2. In Open WebUI (**Admin Settings -> Web Search**), enable **Web Search**, select `google_pse`, paste your key and ID, and click **Save**.
3. Open a new chat with Qwen3 4B or Qwen3.5 4B, toggle **Integrations -> Web Search**, and ask about current events.

### Local Document RAG

1. Open **Workspace -> Knowledge -> +**, create a collection, and drag in Markdown or text files.
2. Open **Workspace -> Models -> +**, name a preset (such as `OFLM_RAG`), pick a base model, attach your Knowledge collection, enable **Citation**, and save.
3. Chat with `OFLM_RAG` for grounded answers with source citations.

### Vision & Custom Request Parameters

Attach images directly in Open WebUI chats when serving open-runtime VLMs such as `qwen3vl-it:4b` or `qwen3.5:4b`. To pass OpenFlowLM request extensions per chat:

1. Select your model (such as `qwen3.5:4b`), open **Controls** (top-right), and click **Add Custom Parameter**.
2. Set `custom_param_name` to `think` (`false`) or `top_k` (`40`). On non-NixOS installations using legacy Gemma 4 vision (`gemma4-it:12b` or `gemma4-it:e4b`; {{< b k="wip" issue="73" t="Issue #73" >}}, {{< b k="wip" issue="244" t="Issue #244" >}}), set `image-max-tokens` (for example, `560`) to cap visual tokens.