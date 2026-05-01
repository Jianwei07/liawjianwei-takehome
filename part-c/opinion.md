# Part C — Opinion: Investigating Stale or Irrelevant Answers

**Author:** Liaw Jian Wei (Jayden)
**Date:** 01 May 2026

---

## Scenario

Six months post-deployment, users report stale or irrelevant answers even though source documents are correct. The bug lives in the ingestion or retrieval pipeline. The system follows Part A's three-stage funnel: BM25 + FAISS → RRF fusion → cross-encoder. Each investigation targets a different failure point in that funnel.

---

## The First Three Things I Would Investigate

### 1. Silent per-file failures (Stage 0: ingestion)

**Why it's an issue:** The batch job exits with code 0 and both indexes swap atomically — everything looks healthy. But individual files can be silently skipped: a corrupted PDF caught by a bare `except Exception: pass`, a permission error swallowed, a network timeout on a mounted drive ignored. Both indexes are internally consistent but missing the same documents. No alert fires.

**Suggestion Fix:** `md5sum` every source file at ingest; store the checksum alongside chunk metadata. Run a reconciliation script on schedule — any hash mismatch flags a stale document for re-ingestion.

> Distinct from Part A's atomic rename check, which catches FAISS/BM25 going out of sync. This catches both indexes being consistently wrong.

---

### 2. Retrieval recall regression + RRF vocabulary drift (Stage 1–2: BM25, FAISS, RRF)

**Why it's an issue:** BM25's vocabulary is frozen at index build time. Six months of document updates introduce new terminology — policy revisions, new acronyms, code numbers — that BM25 has never seen. It returns poor ranks for queries on those terms. RRF propagates the bad ranks silently into the fused top-20; its equal-weight assumption means it cannot detect that one retriever has degraded.

**Suggestion Fix:** Sample 50 queries from user logs; measure recall@5 for BM25-only, FAISS-only, and fused. BM25 regressed → rebuild index, verify new terms are covered. FAISS regressed → re-embed. Both fine but answers still wrong → spot-check cross-encoder scores on 10 failure cases manually.

---

### 3. Chunk boundary fragmentation (Stage 0: 400-token chunker)

**Why it's an issue:** A policy answer often spans a section header and its value — split across two adjacent 400-token chunks. Each chunk is scored in isolation by the cross-encoder; neither chunk alone contains the complete fact. The LLM receives partial context and either hedges or hallucinates the missing half.

**Suggestion Fix:** Sample 20 user-flagged queries; manually verify whether the complete answer fits within one chunk or straddles a boundary. If fragmented in > 30% of cases: increase `chunk_overlap` from 10% (40 tokens) to 25% (100 tokens) in chunker config and rebuild both indexes.
