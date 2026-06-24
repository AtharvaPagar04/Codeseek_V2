# Local Models Configuration

CodeSeek supports running queries and embeddings completely locally using providers like Ollama. When running on consumer hardware (such as laptops or desktop workstations), configuring model settings properly is essential to prevent thermal throttling, out-of-memory crashes, or interface lag.

---

## Environment Variables

The following environment variables configure the behavior of local query generation and embedding execution:

### `CODESEEK_QUERY_NUM_CTX`
* **Purpose:** Sets the size of the context window (in tokens) for the local LLM query.
* **Default:** `8192`
* **Description:** A larger context window allows CodeSeek to send more retrieved code chunks as context, improving answer accuracy. However, larger context windows consume significantly more VRAM.

### `CODESEEK_QUERY_MAX_TOKENS`
* **Purpose:** Sets the maximum number of tokens generated in the response.
* **Default:** `2048`
* **Description:** Limits the length of the model's answer, preventing run-away generation loops and keeping generation time reasonable.

### `CODESEEK_QUERY_OLLAMA_KEEP_ALIVE`
* **Purpose:** Dictates how long the model remains loaded in system/GPU memory after a request.
* **Default:** `5m` (5 minutes)
* **Description:** Keeps the model cached in memory so subsequent questions receive instant answers without needing to reload the model weights (which can take 10-30 seconds on local disks). Set to `0` to unload the model immediately after generation.

### `CODESEEK_EMBEDDING_COOLDOWN_EVERY`
* **Purpose:** Sets the batch frequency for embedding execution cooldowns during ingestion.
* **Default:** `50`
* **Description:** Specifies the number of files or chunks processed before pausing. This prevents the GPU/CPU from running at 100% capacity continuously during massive repository indexing runs.

### `CODESEEK_EMBEDDING_COOLDOWN_SECONDS`
* **Purpose:** Sets the duration of the cooldown pause.
* **Default:** `10`
* **Description:** The number of seconds the pipeline will sleep when the cooldown threshold is hit. This allows local hardware to cool down and prevents system-wide freezes.

---

## Recommended Laptop-Safe Settings

For consumer laptops (e.g., MacBooks or laptops with mid-range Nvidia/AMD GPUs with 8GB-16GB VRAM), we recommend the following configurations in your `.env` file:

### Configuration Profile: Balanced (Recommended)
This profile provides a good balance between context window length and generation performance:
```bash
CODESEEK_QUERY_NUM_CTX=8192
CODESEEK_QUERY_MAX_TOKENS=1024
CODESEEK_QUERY_OLLAMA_KEEP_ALIVE=10m
CODESEEK_EMBEDDING_COOLDOWN_EVERY=40
CODESEEK_EMBEDDING_COOLDOWN_SECONDS=15
```

### Configuration Profile: Low VRAM / Budget Laptops (Integrated Graphics)
If you are running Ollama on CPU or integrated graphics with less than 8GB of shared memory:
```bash
CODESEEK_QUERY_NUM_CTX=4096
CODESEEK_QUERY_MAX_TOKENS=512
CODESEEK_QUERY_OLLAMA_KEEP_ALIVE=1m
CODESEEK_EMBEDDING_COOLDOWN_EVERY=20
CODESEEK_EMBEDDING_COOLDOWN_SECONDS=20
```

> [!TIP]
> **Monitor System Load:** If you experience keyboard input delay or system freezes during repository indexing, increase the `CODESEEK_EMBEDDING_COOLDOWN_SECONDS` value or reduce `CODESEEK_EMBEDDING_COOLDOWN_EVERY` to allow your machine more frequent cooling periods.
