# CodeSeek

CodeSeek is a repository-grounded code assistant. It clones a GitHub repository, parses and chunks supported files, stores embeddings and metadata in Qdrant, and answers questions using retrieved source evidence.

## Components

- `backend/rag_ingestion/`: discovery, filtering, parsing, chunking, metadata, summaries, embeddings, Qdrant storage, and graph construction.
- `backend/retrieval/`: FastAPI API, sessions, retrieval, generation, memory, persistence, authentication, and diagnostics.
- `frontend/`: React and Vite interface for repositories, indexing, chat, source evidence, providers, and repository graphs.
- `tests/e2e/`: Playwright deployment workflow tests.

## End-to-End System Architecture & Workflow Diagram

```mermaid
flowchart TD
    subgraph Client["Frontend Interface (React / Vite)"]
        UI["User Interface Dashboard"]
        GraphUI["Interactive Graph Canvas (D3 / SVG)"]
        LogUI["Live Indexing Terminal (SSE Stream)"]
        CardUI["Source Evidence Cards"]
    end

    subgraph Ingestion["1. RAG Ingestion Engine"]
        Repo["Git Clone / Local Repository"] --> Discovery["Path Discovery & Enumeration"]
        Discovery --> Filter["Ignore Rules & Binary Filtering"]
        Filter --> AST["Tree-sitter AST Parsing (Python, JS, TS, JSX, TSX)"]
        AST --> Chunking["Structural Semantic Chunking"]
        Chunking --> Overflow{"Exceeds 2048 Tokens?"}
        Overflow -->|"Yes"| SlidingWindow["Sliding Window Overflow Handler (100 Lines, 20 Line Overlap)"]
        Overflow -->|"No"| Summarization["Multi-Level Summarization (Chunk, File, Architecture)"]
        SlidingWindow --> Summarization
        Summarization --> Embeddings["Dense Embedding Generation (SentenceTransformers / BGE)"]
        Embeddings --> QdrantDB[("Qdrant Vector Database")]
        AST --> GraphBuild["Relational Knowledge Graph Assembly (Nodes, Edges, Confidence Tiers)"]
        GraphBuild --> RelationalDB[("Relational DB (SQLite / PostgreSQL)")]
    end

    subgraph QueryPipeline["2. Query Processing & Hybrid Search Engine"]
        UserQuery["User Prompt"] --> IntentClass["Query Intent Classifier"]
        UserQuery --> HintExtract["Structural Hint & Symbol Extractor"]
        
        IntentClass --> HybridSearch["12-Channel Hybrid Search Engine"]
        HintExtract --> HybridSearch
        
        subgraph Modes["12 Candidate Search Modes"]
            M1["Dense Vector Search"]
            M2["Lexical BM25 Sparse Search"]
            M3["Exact Structural Entity Search"]
            M4["Metadata & Label Filter"]
            M5["Code Topic & Feature Route"]
            M6["Domain Boost & Repo Profile"]
            M7["Feature Recall Discovery"]
            M8["Framework-Aware Routing"]
            M9["Graph & Import Tree Expansion"]
            M10["Conversation History Injection"]
            M11["Local Source Truth Match"]
            M12["Component Semantic Targeting"]
        end
        
        HybridSearch --> Modes
        Modes --> Fusion["Reciprocal Rank Fusion (RRF) & Multi-Hit Weighting"]
        Fusion --> Rerank["Intent-Aware Dynamic Reranking"]
        Rerank --> GraphExpand["Symbol & Import Tree Graph Expansion (1-3 Hops)"]
        GraphExpand --> SourceTruth["Source Truth Disk Verification & Grounding Filter"]
    end

    subgraph Generation["3. Dual-Path Answer Generation & Safeguards"]
        SourceTruth --> IntentRouter{"Query Intent Destination"}
        
        IntentRouter -->|"Structural Inquiry"| Builders["Deterministic Builder Suite (File Summary, Snippet, Architecture, Flow)"]
        Builders --> TruthCheck{"Truthiness Validation Check"}
        TruthCheck -->|"Valid Code Evidence"| ExactGrounding["Exact Value Grounding & Post-Validation"]
        TruthCheck -->|"Empty / Whitespace"| DynamicFallback["Dynamic Fallback Circuit Breaker"]
        
        IntentRouter -->|"Conceptual Reasoning"| DynamicFallback
        DynamicFallback --> LLMPath["LLM Generation Engine (OpenAI, Anthropic, Ollama, Local)"]
        LLMPath --> ExactGrounding
        
        ExactGrounding --> FinalResponse["Final Grounded Response & Diagnostic Trace"]
    end

    FinalResponse --> UI
    FinalResponse --> CardUI
    Ingestion -->|"Real-time SSE Status"| LogUI
    RelationalDB -->|"Node / Edge Topology API"| GraphUI
```

---


## Local Start

Requirements: Docker with Compose.

```bash
cp backend/.env.example backend/.env
docker compose -f docker-compose.dev.yml up --build
```

The development stack exposes:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- Backend health: `http://localhost:8000/api/v1/health`
- Qdrant: `http://localhost:6333`
- PostgreSQL: `localhost:5432`

Configure GitHub OAuth, provider credentials, and embedding settings before indexing private repositories or generating answers.

## Verification

```bash
PYTHONPATH=backend backend/.venv/bin/python -m pytest backend/tests
npm --prefix frontend test
npm --prefix tests/e2e test
```

## Documentation

Start with [the documentation index](docs/README.md). It links the product, architecture, pipeline, API, operations, testing, and reference pages.

## Current Boundaries

- Tree-sitter parsers are configured for Python, JavaScript, TypeScript, JSX, and TSX.
- Qdrant stores repository chunks; SQLite or PostgreSQL stores application state.
- Retrieval combines deterministic lookup, dense search, optional lexical search, metadata search, expansion, reranking, and source filtering.
- Graph retrieval is configurable as shadow diagnostics, per-query Graph Assist, or process-level active mode.
- Provider keys and GitHub credentials are encrypted before database storage when the application encryption key is configured.
