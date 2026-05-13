# Financial Lab — Engineering Walkthrough

A peer-to-peer explanation of what this project is, how it is wired, and why the moving pieces were chosen the way they were. Written assuming the reader is comfortable with Python, vector databases, RAG patterns, and modern LLM-orchestration.

---

## 1. What the project is

`financial-lab` is a **multi-source financial research stack** that ingests public company information (SEC filings + financial news), indexes it into a **hybrid vector store**, and serves three increasingly opinionated read paths over it:

1. `/search` — raw hybrid retrieval over the corpus.
2. `/rag` — single-turn retrieval-augmented Q&A grounded in the corpus.
3. `/agent` — an **agentic ensemble** that fans out into three parallel analyst personas (fundamental / momentum / sentiment) and then aggregates their outputs into a final BUY / HOLD / SELL recommendation.

The whole thing is a FastAPI service ([app/main.py](app/main.py), [app/application.py](app/application.py)) backed by **Qdrant** for retrieval and **Groq (Llama 3.1 8B Instant)** for generation. Offline ingestion lives under [ingestion/](ingestion/).

---

## 2. High-level architecture

```
                ┌─────────────────────────────────────────────┐
                │              Offline Ingestion              │
                │                                             │
                │   EDGAR (10-K / 10-Q)  ─►  SemanticChunker  │
                │   Yahoo Finance news  ─►  SimpleChunker     │
                │                            │                │
                │                            ▼                │
                │           dense + sparse + colbert encode   │
                │                            │                │
                │                            ▼                │
                │                      Qdrant collection      │
                │                       "financial"           │
                └─────────────────────────────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────────┐
                │                 FastAPI app                 │
                │                                             │
                │   /search ─►  SearchService                 │
                │                 ├─ EmbeddingService         │
                │                 └─ Qdrant hybrid query      │
                │                                             │
                │   /rag    ─►  RAGService    (Groq)          │
                │                                             │
                │   /agent  ─►  AgentService                  │
                │                 ├─ 3 parallel asyncio tasks │
                │                 │   (fundamental / momentum │
                │                 │    / sentiment)           │
                │                 └─ aggregation prompt       │
                └─────────────────────────────────────────────┘
```

---

## 3. Code map

| Path | Role |
|------|------|
| [app/main.py](app/main.py) | Uvicorn entrypoint. |
| [app/application.py](app/application.py) | `ApplicationBuilder` — wires the routers. |
| [app/config/settings.py](app/config/settings.py) | `pydantic-settings`–based config. Holds Qdrant creds, model identifiers, Groq model. |
| [app/config/prompts.py](app/config/prompts.py) | All system prompts and seed-queries for the agent. |
| [app/routers/](app/routers/) | Thin HTTP layer. Each router owns one service instance. |
| [app/services/embeddings.py](app/services/embeddings.py) | Wraps the three FastEmbed models. |
| [app/services/search.py](app/services/search.py) | The hybrid 3-stage Qdrant query. |
| [app/services/rag.py](app/services/rag.py) | Retrieval + Groq completion. |
| [app/services/agent.py](app/services/agent.py) | Parallel multi-persona analysis + aggregation. |
| [ingestion/ingestion.py](ingestion/ingestion.py) | 10-K / 10-Q ingestion pipeline. |
| [ingestion/news_ingestion.py](ingestion/news_ingestion.py) | Yahoo Finance news ingestion. |
| [ingestion/create_collection.py](ingestion/create_collection.py) | One-shot Qdrant collection bootstrap. |
| [ingestion/utils/edgar_client.py](ingestion/utils/edgar_client.py) | EDGAR fetcher; extracts specific Items (1, 1A, 7, 8, 9A for 10-K; 1–4 for 10-Q). |
| [ingestion/utils/news_client.py](ingestion/utils/news_client.py) | Yahoo Finance news + `trafilatura` text extraction. |
| [ingestion/utils/semantic_chunker.py](ingestion/utils/semantic_chunker.py) | HDBSCAN-based semantic clustering chunker. |
| [ingestion/utils/simple_chunker.py](ingestion/utils/simple_chunker.py) | Naive token-budget paragraph chunker. |

---

## 4. The retrieval pipeline — and **why three models**

This is the most interesting part of the system, and the part that has the most engineering rationale behind it. The pipeline in [app/services/search.py:18-33](app/services/search.py#L18-L33) does this:

```
                       ┌────────────────────┐
                       │   user query text   │
                       └─────────┬──────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      dense (MiniLM)     sparse (BM25)       colbert (v2.0)
              │                  │                  │
              └────────┬─────────┘                  │
                       ▼                            │
                Reciprocal Rank Fusion              │
                  (top-15 fused)                    │
                       │                            │
                       ▼                            │
                first-stage candidates ─────────────┘
                       │
                       ▼
              ColBERT MaxSim rerank
                  (top-N final)
```

That is, **two parallel first-stage retrievers** (semantic + lexical) are fused with RRF, then a **late-interaction reranker** takes the top fused candidates and produces the final ordering. Each of the three embedding models exists for a specific reason.

### 4.1 `sentence-transformers/all-MiniLM-L6-v2` — the dense retriever

- **What it is.** A 6-layer distilled BERT producing **384-dim** sentence embeddings, trained on >1B sentence pairs with a contrastive objective. Single-vector, cosine similarity.
- **Why we picked it.**
  - **Semantic recall.** It catches paraphrases and conceptual matches that BM25 misses entirely — e.g. a user asking *"what could derail near-term growth?"* still surfaces a 10-Q passage that talks about *"emerging supply-chain headwinds"*.
  - **Cheap and fast.** 384 dims keeps the Qdrant footprint small; inference is CPU-friendly. The whole pipeline runs three encodes per query, so latency matters.
  - **General-purpose.** It is not finance-tuned, but for a project at this stage that is a *feature*: zero-shot domain coverage and stable behavior without us owning a fine-tune.
  - **Reuse.** The same model backs the **SemanticChunker** ([ingestion/utils/semantic_chunker.py:14-24](ingestion/utils/semantic_chunker.py#L14-L24)) and its tokenizer is reused by the **SimpleChunker** ([ingestion/utils/simple_chunker.py:6-10](ingestion/utils/simple_chunker.py#L6-L10)). One model, three jobs (chunking, query encoding, passage encoding) keeps the dependency surface minimal.
- **Limits we accept.** 256-token effective context, no finance-specific vocabulary (e.g. SEC item numbering), and the usual single-vector ceiling on fine-grained matching. Both gaps are covered by the other two models.

### 4.2 `Qdrant/bm25` — the sparse retriever

- **What it is.** A pure **lexical** scorer in BM25, served as a Qdrant sparse vector (token-id → tf-idf-style weight).
- **Why we picked it.**
  - **Lexical anchors matter in finance.** Tickers (`AAPL`), boilerplate phrases (`Item 1A. Risk Factors`), regulatory references (`10-Q`, `MD&A`), product names, and proper nouns are **rare tokens whose exact presence is meaningful**. Dense embeddings smear these into nearby clusters; BM25 nails them.
  - **Out-of-distribution robustness.** New tickers / new product names that MiniLM never saw still retrieve correctly with BM25.
  - **Cheap complement.** Computing a sparse vector is ~free compared to a transformer pass, so the retrieval cost is essentially "dense + epsilon."
- **Why RRF over score-weighted fusion?** Reciprocal Rank Fusion is **scale-free** — it only consumes the *rank lists* from each retriever, so we never have to calibrate cosine similarities against BM25 scores. This is exactly the pattern used in [app/services/search.py:21-28](app/services/search.py#L21-L28) via `models.FusionQuery(fusion=models.Fusion.RRF)`.

### 4.3 `colbert-ir/colbertv2.0` — the late-interaction reranker

- **What it is.** ColBERTv2 is a **multi-vector / late-interaction** model: it embeds *every token* of a passage as its own 128-dim vector and scores a passage against a query by computing **MaxSim** — for each query token, take the max cosine to any passage token, then sum.
- **Why we picked it.**
  - **Precision at the top.** Single-vector dense models compress a whole passage into one point, so they confuse passages that "feel similar" but answer different questions. ColBERT's token-level interaction recovers the precision that the compression destroyed — exactly what you want as a final-stage reranker.
  - **Stage-appropriate cost.** ColBERT is heavier per scoring operation (it's a multivector comparison) and would be expensive over the full corpus. We **only ever run it over the ~15 fused candidates from stage one** ([app/services/search.py:27](app/services/search.py#L27)), keeping the cost bounded.
  - **Native Qdrant support.** Qdrant exposes it as a regular vector index with `multivector_config=MultiVectorConfig(comparator=MAX_SIM)` ([ingestion/_ingestion.py:35-41](ingestion/_ingestion.py#L35-L41)), so we don't need a separate reranking service.
- **Why this design pattern.** This is the **"hybrid recall + late-interaction rerank"** pattern that has become the default for serious RAG systems: cheap retrievers cast a wide net, expensive late-interaction tightens the top.

### 4.4 How they compose

The composition in [app/services/search.py](app/services/search.py) is the key insight to internalize:

1. Dense (top-20) + Sparse (top-20) → RRF → top-15 *fused* candidates.
2. ColBERT MaxSim reranks those 15 → top-`limit` (default 3) final results.
3. Scores are min-max normalized to `[0,1]` against the top hit so downstream callers can reason about confidence on a stable scale.

In other words: **MiniLM gives us semantic recall, BM25 gives us lexical recall, ColBERT gives us precision.** Each one is the cheapest model that does its job, and each runs on a stage of the pipeline sized to its cost.

---

## 5. Ingestion

Two pipelines feed the same `financial` collection.

### 5.1 SEC filings — [ingestion/ingestion.py](ingestion/ingestion.py)

- `EdgarClient.fetch_filing_data(ticker, form_type)` pulls the latest 10-K or 10-Q and surfaces only the **information-dense Items** (Risk Factors, MD&A, Financial Statements, etc.) — defined in `FORM_ITEMS` at [ingestion/utils/edgar_client.py:7-10](ingestion/utils/edgar_client.py#L7-L10). This is deliberate: 10-Ks contain ~150 pages of which only specific items have alpha-relevant content.
- Text goes through `SemanticChunker`, which:
  1. Splits to paragraphs and filters out lines under 10 tokens (kills tables-of-contents, page numbers, footers).
  2. Embeds each paragraph with MiniLM and runs **HDBSCAN** to cluster *semantically related* paragraphs ([ingestion/utils/semantic_chunker.py:32-44](ingestion/utils/semantic_chunker.py#L32-L44)).
  3. Within each cluster, greedily packs paragraphs into chunks up to `max_tokens=300`.
  4. Orphans (HDBSCAN's `-1` label) get a second clustering pass with a smaller `min_cluster_size`, and anything still solo is emitted as its own chunk.

  This is meaningfully better than a fixed-window chunker for filings because related risk factors / accounting notes tend to be spread across non-adjacent paragraphs, and HDBSCAN keeps them in the same chunk regardless of physical layout.

### 5.2 News — [ingestion/news_ingestion.py](ingestion/news_ingestion.py)

- `NewsClient.fetch_news(ticker)` uses `yfinance` for the headlines, filters to `STORY` content on `finance.yahoo.com`, and pulls the body with `trafilatura` (which strips boilerplate / ads).
- Chunking uses `SimpleChunker` instead of the semantic one. **The choice is deliberate:** news articles are short and topically coherent already, so paying for HDBSCAN gives you nothing. A token-budget paragraph packer is the right call.

### 5.3 Encoding & upload

Both pipelines encode each chunk with all three models and upload to Qdrant as a `PointStruct` whose `vector` is a dict — `dense`, `sparse`, `colbert` — matching the `vectors_config` defined in [ingestion/create_collection.py:18-33](ingestion/create_collection.py#L18-L33) (dense=384-dim cosine, ColBERT=128-dim MAX_SIM multivector, sparse=`SparseVectorParams`).

---

## 6. The `/agent` endpoint — multi-persona ensemble

This is the most product-y piece. The flow lives in [app/services/agent.py](app/services/agent.py):

```
                          ticker (e.g. "AAPL")
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
 _analyze_fundamental    _analyze_momentum       _analyze_sentiment
   (4 seed queries          (3 seed queries        (1 ticker-specific
    over 10-K)               over 10-Q)              news query)
        │                        │                        │
        ▼                        ▼                        ▼
   FUNDAMENTAL_PROMPT      MOMENTUM_PROMPT          SENTIMENT_PROMPT
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 ▼
                       AGGREGATION_PROMPT
                                 │
                                 ▼
                       final_recommendation
```

Key design choices worth calling out:

- **Seed queries instead of dynamic query generation.** Each persona has a fixed list of search strings in [app/config/prompts.py](app/config/prompts.py) (`FUNDAMENTAL_QUERIES`, `MOMENTUM_QUERIES`, `SENTIMENT_QUERY_TEMPLATE`). Cheap, deterministic, and easy to evaluate — no extra LLM call to plan retrieval. The cost is that the queries don't adapt to the company; the win is reproducibility and that the queries are tuned to the *content axis* each persona cares about (risk factors / financials / MD&A for fundamental; quarter-over-quarter changes for momentum; recent news for sentiment).
- **Parallel via `asyncio.gather`.** The three persona analyses are independent, so [app/services/agent.py:63-67](app/services/agent.py#L63-L67) fans them out concurrently. The LLM calls dominate latency, so this is roughly a 3× speedup.
- **Structured prompts with explicit output schemas.** Each persona prompt enforces a specific output shape (investment grade A–D, momentum score 0–10, sentiment score 1–10, etc.). This makes the aggregator's job much easier because it is comparing apples to apples.
- **Aggregation is itself an LLM call.** `AGGREGATION_PROMPT` instructs the model to weight the three streams, flag divergence, and produce a single BUY / HOLD / SELL with rationale. The decision framework ("2+ streams align ⇒ high confidence; streams conflict ⇒ lower confidence + explain divergence") is explicit in the prompt, which is more robust than burying that logic in Python.
- **`temperature=0` everywhere.** Across both `RAGService` and `AgentService` the Groq client is called with `temperature=0` for determinism — the right call for a research/analysis tool where reproducibility matters more than creativity.

---

## 7. Configuration & infra

- **Settings.** [app/config/settings.py](app/config/settings.py) uses `pydantic-settings` to load from `.env`. The three model identifiers and the Groq model live here too, so we can swap them without code changes.
- **LLM.** Groq's `llama-3.1-8b-instant` — cheap, fast, good enough for structured analysis. Easy to swap to a larger model by changing `groq_model`.
- **Vector DB.** Qdrant Cloud (URL + API key in settings).
- **Python.** ≥3.12, managed with `uv` (see `uv.lock`, `pyproject.toml`).

---

## 8. What I would call out for review

If I were reviewing this for a teammate, here is the punch list I'd raise — not blockers, but worth knowing:

1. **`SearchService` and `EmbeddingService` are instantiated per-router** in [app/routers/search.py:14](app/routers/search.py#L14) and [app/routers/rag.py:14](app/routers/rag.py#L14) — so the FastEmbed models are loaded **twice** at boot. A single shared instance (DI via FastAPI `Depends`, or a module-level singleton) would halve memory and startup time.
2. **No `try/except` around the LLM or Qdrant calls.** Acceptable for a lab project; an obvious target if this ever goes to production.
3. **`max_score` normalization** in [app/services/search.py:35](app/services/search.py#L35) will blow up on empty result sets. Same for the agent paths if the corpus is empty.
4. **`EMAIL` env var requirement** for EDGAR is documented only via a runtime `raise ValueError` in [ingestion/ingestion.py:30](ingestion/ingestion.py#L30) — would be nicer to surface it in settings.
5. **No CLAUDE.md or README content** to onboard a new contributor; this document tries to fill that gap.

---

## 9. TL;DR — the mental model

> Retrieve broadly with two cheap retrievers, rerank precisely with one expensive one, then run three specialist LLM "analysts" in parallel over the resulting context and have a fourth LLM call adjudicate.
>
> MiniLM is the recall workhorse and chunking model. BM25 is the lexical safety net that ensures tickers and SEC-specific phrases never get lost. ColBERT is the precision layer that decides which of the recalled candidates actually answers the query. Together they cover the three failure modes a single retriever can't: semantic miss, lexical miss, and false-positive at the top.
