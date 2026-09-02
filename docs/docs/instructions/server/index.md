---
layout: docs
title: Local Server (Server Mode)
parent: Instructions
nav_order: 7
has_children: true
---

# Local Server (Server Mode)

## Activate "Server Mode" 

Open PowerShell and enter:

```bash
flm serve llama3.2:1b
```

You can choose to change the server port (default is 52625) by going to **System Properties** → **Environment Variables**, then modifying the value of `FLM_SERVE_PORT`.

> ⚠️ **Be cautious**: If you update this value, be sure to change any higher-level port settings in your application as well to ensure everything works correctly.

## NPU Model Loading Behavior

FLM can keep one NPU model loaded per type at a time:

- `asr`
- `llm`
- `embedding`

Different model types can run together (for example, one LLM and one embedding model).

### Load all three model types in one server process (default port)

```shell
flm serve lfm2:1.2b -e 1 -a 1
```

### Run model types in separate server processes (different ports)

```shell
flm serve -e 1 --port 52625
flm serve -a 1 --port 52627
flm serve lfm2:1.2b --port 52628
```




## Set Context Length at Launch

The default context length for each model can be found [here](https://fastflowlm.com/docs/models/). 

To change it at launch, in PowerShell, run:

```bash
flm serve llama3.2:1b --ctx-len 8192
```

> - Internally, FLM enforces a minimum context length of 512. If you specify a smaller value, it will automatically be adjusted up to 512.  
> - If you enter a context length that is not a power of 2, FLM automatically rounds it up to the nearest power of 2. For example: input `8000` → adjusted to `8192`.  

## Show Server Port 

Show current FLM port (default) in PowerShell:  
  
  ```shell
  flm port
  ```

## Set Server Port at Launch

Set a custom port at launch:

  ```shell
  flm serve llama3.2:1b --port 8000
  flm serve llama3.2:1b -p 8000
  ```

> ⚠️ `--port` (`-p`) only affects the **current run**; it won’t change the default port.


## Set Request Queue in Server Mode

Since v0.9.10, FLM adds a request queue in server mode to prevent overload under high traffic.  
This keeps processing stable and orderly when multiple requests arrive.

- **Default:** 10  
- **Change with:** `--q-len` (or `-q`)  

To change it at launch, in PowerShell, run:
  
```shell
flm serve llama3.2:1b --q-len 20
```

## Customizable Socket Connections in Server Mode

Set the maximum number of concurrent socket connections to control network resource usage.  
👉 *Recommended:* set sockets **equal to or greater than** the queue length.  

- **Default:** 10  
- **Change with:** `--socket` (or `-s`)  

To change it at launch, in PowerShell, run:

```shell
flm serve llama3.2:1b --socket 20
```

### Cross-Origin Resource Sharing (CORS)

CORS lets browser apps hosted on a different origin call your FLM server safely.

- Enable CORS

```shell
flm serve --cors 1
```
- Disable CORS

```shell
flm serve --cors 0
```

> ⚠️ **Default:** CORS is **enabled**.  
> 🔒 **Security tip:** Disable CORS (or restrict at your proxy) if your server is exposed beyond localhost (127.0.0.1).


## Suppress Logs for Higher-Level Applications

When FLM is run as a subprocess inside another application, use quiet mode to reduce FLM log output:

```shell
flm serve --quiet
```

This keeps the parent application's logs cleaner and easier to read.