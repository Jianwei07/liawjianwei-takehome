# Part C — Opinion: Investigating Stale or Irrelevant Answers

**Author:** Liaw Jian Wei (Jayden)
**Date:** 01 May 2026

---

## Scenario

Six months post-deployment, users report stale or irrelevant answers even though source documents on disk are current. The system follows Part A's three-stage retrieval funnel: BM25 + FAISS → RRF fusion → cross-encoder. Each investigation below targets a distinct failure point in that funnel.

---

## The First Three Things I Would Investigate

### 1. Silent ingestion failures (Stage 0: ingestion pipeline)

**Why it's an issue:** The ingestion pipeline is the only path for documents to enter the system. If individual files are silently skipped — a corrupted PDF caught by a bare `except Exception: pass`, a permission error swallowed, a network timeout ignored — the index is incomplete but appears healthy. The batch job exits code 0, both indexes swap atomically, no alert fires. The system confidently answers from stale data with no visible failure signal. This is the most dangerous failure mode: silent, complete, and undetectable without explicit instrumentation.

**Fix 1:** At ingest, `md5sum` every source file and store the checksum alongside chunk metadata. Run a scheduled reconciliation script that diffs stored checksums against current source files — any mismatch flags a document for re-ingestion.

**Fix 2:** Replace all bare `except` blocks in the ingestion pipeline with explicit error surfacing. Every skipped file must be logged with the reason and counted. The batch job should exit non-zero if any file fails, making the failure alertable.

---

### 2. BM25 vocabulary drift (Stage 1: BM25 sparse retrieval)

**Why it's an issue:** BM25's vocabulary is frozen at index build time. Six months of document updates introduce new terminology — revised policy names, new acronyms, updated code references — that BM25 has never seen. Queries using new terms return near-zero BM25 scores for every chunk. RRF then propagates those degraded ranks into the fused top-20 silently, since its equal-weight assumption provides no mechanism to detect that one retriever has degraded. Updated documents are indexed, but effectively unreachable for new-terminology queries.

**Fix 1:** After each monthly rebuild, compare vocabulary size and token coverage before and after. A drop in coverage signals drift.

**Fix 2:** Run recall@5 on a sample of recent queries split by BM25-only, FAISS-only, and fused. A BM25-specific regression with FAISS recall intact points directly to vocabulary drift and narrows the fix to an index rebuild.

---

### 3. Answer fragmentation across chunk boundaries (Stage 0: chunker)

**Why it's an issue:** Fixed 400-token chunking does not respect semantic boundaries. A policy answer — the header in chunk N, the numeric value in chunk N+1 — can span two adjacent chunks. Both chunks may be retrieved, but the cross-encoder scores them independently. Neither alone contains the complete fact, so the LLM receives partial context and hedges or hallucinates the missing value. This failure is detectable via the Part B eval harness: low `keyword_f1` and `sequence_similarity` scores on queries with known numeric answers are a reliable early signal.

**Fix 1:** Sample 20 user-flagged queries and manually verify whether the complete answer fits within one chunk or straddles a boundary.

**Fix 2:** If fragmented in more than 30% of cases, increase `chunk_overlap` from 10% (40 tokens) to 25% (100 tokens) in chunker config and rebuild both indexes. Re-run the Part B eval harness to confirm score improvement before promoting to production.
