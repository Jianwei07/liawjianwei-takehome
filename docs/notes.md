# Part A notes

### Why FAISS Wins in This Specific Context

The assignment forces three major constraints: On-premise/air-gapped, no dedicated DevOps, and a very small dataset (2,000 documents).

Zero Infrastructure Overhead: Because FAISS is a C++ library with Python bindings, it runs directly inside your application's memory space. You just pip install faiss-cpu. There is no separate server to stand up, no networking protocols to configure between microservices, and no external state to manage. For a small team with no DevOps, this is a massive operational win.

Perfect Fit for RAM: 2,000 documents, even chunked into 10,000 segments and embedded as 384-dimensional vectors (like bge-small-en), will consume only a few megabytes of RAM. You can use an exact search index (IndexFlatIP for inner product/cosine similarity) which guarantees 100% recall without relying on the approximations (ANN) required for billion-scale datasets.

Compute Isolation: By running your dense retrieval entirely on the CPU in RAM via FAISS, you reserve the GPU cluster entirely for the LLM text generation.

Is it only done ONCE? In the context of this specific assignment, it is done once per month because the brief states you have roughly 2,000 documents updated monthly. This is a "batch ingestion" architecture. You do not need a streaming ingestion pipeline.

Does FAISS do the organization? Yes. You vectorize the text, hand those vectors to FAISS, and FAISS builds the mathematical index (the spatial organization of those vectors) which you then save to disk.

If you chose Elasticsearch (Option 3), you would have to maintain a heavy Java-based search cluster. If you chose ChromaDB (Option 2), you'd be adding another background service dependency that can fail independently of your main application.

### Why FAISS still wins?

The Challengers (Advanced Ingestion Strategies)

1. Hierarchical Chunking (Parent-Child Retrieval)

How it works: During ingestion, you chunk the document twice. First into large "Parent" chunks (e.g., a whole section), and then subdivide those into small "Child" chunks (e.g., individual sentences). You only vectorize and pass the Child chunks to FAISS. However, you map each Child to its Parent's ID.

Why it challenges the baseline: FAISS is incredibly accurate at finding the small, hyper-specific Child chunk. But instead of sending that tiny, context-less chunk to the LLM, your system retrieves its Parent chunk and sends that to the LLM. You get the precision of small vectors with the rich context of large text blocks.

2. Semantic Chunking

How it works: Instead of cutting blindly at a token limit, you use a lightweight NLP model (like NLTK or SpaCy) during ingestion to split the text at natural boundaries—like the end of sentences or paragraphs. More advanced versions use an embedding model to detect shifts in topic and slice the document there.

Why it challenges the baseline: It ensures that every vector in your FAISS index represents a complete, cohesive thought rather than a fragmented string of words.

3. Metadata Enrichment (Crucial for Hybrid RAG)

How it works: Before vectorizing, you run a script to extract metadata from the document (e.g., Date_Updated, Department, Doc_Type). You store this metadata alongside the chunk in a simple key-value store (like SQLite or an in-memory dictionary).

Why it challenges the baseline: FAISS only understands mathematical similarity, not objective facts. If a user asks "What is the 2026 HR policy?", dense retrieval might return the 2024 policy because the text is semantically identical. By extracting metadata during ingestion, you can apply a hard pre-filter (e.g., WHERE Year = 2026) before FAISS even starts its semantic search.

4. Graph Extraction (GraphRAG) - The Heavyweight

How it works: During ingestion, you use a small LLM to extract entities (people, places, concepts) and their relationships from the text, building a Knowledge Graph.

Why it challenges the baseline: It allows for "multi-hop" reasoning across different documents, which vector search struggles with. (Note: I would advise against proposing this for Part A, as maintaining a graph database violates the "no DevOps" and "lightweight" constraints of your brief, but it is good to know conceptually).

## Reranker Strategy: Why Cross-Encoder on CPU Wins

1. The "Snappy" Latency Tradeoff (Math works in your favor)

The Cost: Running a lightweight cross-encoder (like ms-marco-MiniLM-L-6-v2) on a CPU to score just 20 documents takes about ~80ms.

The Savings: By aggressively filtering down from 20 candidates to the absolute best top-3, you drastically reduce the context window size sent to the LLM.

The Result: The time you spend on the CPU (80ms) is more than made up for by the GPU time you save. Processing fewer input tokens means a much faster Time To First Token (TTFT) from the LLM, keeping the system feeling "snappy" for those 20 concurrent users.

2. Protecting the GPU Bottleneck

Your GPU cluster behind the API gateway is your most constrained and expensive resource.

By offloading the embedding generation (bge-small-en) and the reranking (cross-encoder) entirely to the CPU, you ensure the GPU is exclusively dedicated to what it does best: heavy text generation.

3. Precision Lift (Combating "Lost in the Middle")

If you choose Option 2 (No reranker), you pass all the retrieval noise directly into the prompt. LLMs are notoriously susceptible to distraction; if the actual answer is buried in the middle of 10 irrelevant documents, the LLM might hallucinate or miss it entirely.

A cross-encoder processes the user's query and the document together (using self-attention across both sequences). It is vastly superior at identifying true semantic relevance compared to the independent vector matching of FAISS.

4. Why not Option 3 (RRF Only)?

Reciprocal Rank Fusion (RRF) is a great, low-cost way to mathematically merge the lists from BM25 and FAISS based on their rank positions. However, it is mathematically blind. It doesn't actually "read" the text to confirm relevance.

Best Practice: You actually use both. You use RRF to cleanly merge the initial BM25 + FAISS results to get your top-20 list, and then use the cross-encoder to scrutinize that list down to the final top-3.

### Why RRF Is the Only Valid Fusion Method Here — The Apples vs. Oranges Problem

To understand why RRF is necessary, you first have to understand the fundamental problem of hybrid search.

BM25 (Sparse Search) scores documents based on keyword frequency. Its scores are unbounded — they can look like 15.4, 32.1, or 8.9.

FAISS (Dense Search) scores documents based on vector distance (cosine similarity). Its scores are strictly bounded, usually between -1.0 and 1.0.

You cannot simply add 32.1 and 0.85 together. The BM25 score would completely overpower the FAISS score. The two scoring systems are measuring completely different things on completely different scales — apples vs. oranges.

### The RRF Solution: Ignore the Score, Look at the Rank

RRF completely discards the raw scores from both BM25 and FAISS. Instead, it looks exclusively at the position (rank) of the document in each respective list.

The formula for a single document:

RRF_Score = 1/(k + Rank_BM25) + 1/(k + Rank_FAISS)

k is a smoothing constant, almost universally set to 60. It prevents a document that ranks #1 in one list from mathematically overpowering everything else.

### What This Means in Practice

- A document ranked #1 by BM25 and #1 by FAISS gets the highest possible RRF score — both retrievers agree it is relevant.
- A document ranked #1 by BM25 but not found by FAISS at all still gets a partial score — sparse retrieval alone can surface it.
- The scale mismatch (32.1 vs 0.85) is completely irrelevant. Only rank positions matter.

This is why RRF needs no training data, no weight tuning, and no calibration — it sidesteps the incompatibility problem entirely by normalising both lists to ranks before combining them.

---

## Part A vs Part C — Why Both Mitigations Are Needed (Interview Defence)

This is a common interview trap: "Part A already has a health check, so why does Part C investigate the same thing?"

They do NOT investigate the same thing. They target two different failure modes:

### Part A failure mode + mitigation (split-brain)

Scenario: The monthly batch is interrupted mid-run (OOM kill, disk full, power cycle). FAISS rebuild finishes, BM25 rebuild does not — or vice versa. The two indexes are now out of sync. RRF pulls from both, blending chunks from two different document versions.

Mitigation: Write both indexes to a staging path. Atomically rename() both into production together as a single filesystem operation. Either both swap or neither does. Also check that FAISS and BM25 document counts match before committing the swap.

### Part C investigation 1 (per-file silent failures)

Scenario: The batch job runs to COMPLETION and exits with code 0. Both indexes are atomically swapped. But during the run, 50 individual files were silently skipped — a corrupted PDF raised an exception caught by a bare `except Exception: pass`, a file permission error was logged but not re-raised, a network timeout on a mounted drive was swallowed.

Both indexes are internally consistent (FAISS and BM25 match each other perfectly), but both are missing the same 50 documents. The atomic rename check passes because counts match — they're just both wrong.

Detection: md5sum (or sha256sum) every source file at ingest time and store the checksum with each chunk's metadata. A reconciliation script re-hashes source files and diffs against stored metadata. Any document whose current hash doesn't match its indexed version is a stale candidate.

### Summary

| Check                       | What it catches                          | What it misses                                     |
| --------------------------- | ---------------------------------------- | -------------------------------------------------- |
| Atomic rename + count check | FAISS and BM25 out of sync (split-brain) | Per-file silent skips where both indexes are wrong |
| md5sum reconciliation       | Per-file silent skips, partial ingestion | Split-brain between indexes                        |

Both are needed. They're complementary, not redundant.

### How RRF Works Hand-in-Hand with FAISS + BM25 — Step-by-Step Example

The apples vs. oranges problem in full:

- **BM25 (Sparse)** scores by keyword frequency — unbounded, e.g. 15.4, 32.1, 8.9
- **FAISS (Dense)** scores by cosine similarity — bounded between -1.0 and 1.0
- You cannot add 32.1 and 0.85 together. BM25 would always dominate.

**RRF solution:** discard raw scores entirely. Use rank positions only.

```
RRF_Score = 1/(k + Rank_BM25) + 1/(k + Rank_FAISS)
```

`k = 60` (industry standard smoothing constant — prevents a #1 rank from overpowering everything).

---

**Worked example — query: "What is the 2026 remote work policy?"**

**Step 1: Parallel search**

- BM25 searches exact keywords: "2026", "remote", "work", "policy" → returns top-50 by keyword frequency
- FAISS searches semantic meaning of the query → returns its own top-50 by cosine similarity

**Step 2: Score each candidate**

| Document | BM25 Rank | FAISS Rank | RRF Score |
| --- | --- | --- | --- |
| Doc A — 2026 HR manual | #3 (has the exact words) | #2 (meaning matches) | 1/(60+3) + 1/(60+2) ≈ **0.032** |
| Doc B — 2022 IT VPN memo | #1 (keyword stuffed) | #40 (poor semantic match) | 1/(60+1) + 1/(60+40) ≈ **0.026** |

**Step 3: Fusion**

Sort all candidates by RRF score descending. **Doc A wins** — even though Doc B ranked #1 in BM25, its weak FAISS rank pulls its combined score below Doc A, which both retrievers agreed was relevant.

---

## Extreme Low-Level: How Retrieval Actually Happens Internally

### BM25 Internals (rank_bm25 library)

**At ingest (index build):**

1. Tokenise each chunk: lowercase, split on whitespace/punctuation, strip stopwords
2. Build inverted index: `{ token → [(chunk_id, tf), ...] }` where `tf` = count of token in that chunk
3. Compute corpus statistics: `N` = total chunks, `avgdl` = mean token count across all chunks
4. Compute IDF per token:

```
IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

where `df(t)` = number of chunks containing token `t`. Rare terms get higher IDF.

**At query time:**

1. Tokenise the query the same way
2. For each query token, look up its IDF and the TF in each candidate chunk
3. Score each chunk against the full query:

```
BM25(chunk, query) = Σ IDF(t) × [ tf(t,chunk) × (k1 + 1) ] / [ tf(t,chunk) + k1 × (1 - b + b × dl/avgdl) ]
```

- `k1 = 1.5` — term frequency saturation. After a word appears a few times, extra occurrences add diminishing value
- `b = 0.75` — length normalisation. A long chunk mentioning a term once is penalised vs a short chunk
- `dl` = chunk length, `avgdl` = corpus average

4. Return top-N chunks sorted by BM25 score (unbounded, can be 0 to ~30+)

---

### FAISS Internals (IndexFlatIP)

**At ingest (index build):**

1. Pass each chunk text through bge-small-en-v1.5 → 384-dimensional float32 vector
2. L2-normalise each vector: `v = v / ‖v‖` (makes inner product = cosine similarity)
3. `faiss.IndexFlatIP` stores all vectors as a flat 2D array in RAM: `10,000 × 384 × 4 bytes ≈ 15 MB`
4. No tree, no clustering, no approximation — just a matrix

**At query time:**

1. Embed the query string with the same bge-small-en model → 384-dim float32 vector
2. L2-normalise the query vector
3. `IndexFlatIP.search(query_vec, k=10)` computes inner product between the query vector and **every** stored vector (brute-force matrix multiply)
4. `inner_product(q, v) = cosine_similarity(q, v)` because both are L2-normalised. Bounded between -1.0 and 1.0
5. Return top-k (chunk_id, cosine_score) sorted descending

**Why IndexFlatIP and not IndexIVFFlat (ANN):** At 10k vectors, brute-force completes in <5ms. ANN indexes (IVF, HNSW) partition the space into clusters to skip most vectors — essential at 100M+ scale, unnecessary overhead at 10k. `IndexFlatIP` guarantees 100% recall; IVF would introduce approximation error for no latency gain.

---

### RRF Fusion Internals

**Input:** two dicts `{chunk_id → rank}`, one from BM25, one from FAISS (rank starts at 1)

**Algorithm:**

```python
k = 60
scores = defaultdict(float)

for rank, chunk_id in enumerate(bm25_results, start=1):
    scores[chunk_id] += 1 / (k + rank)

for rank, chunk_id in enumerate(faiss_results, start=1):
    scores[chunk_id] += 1 / (k + rank)

fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
top_20 = fused[:20]
```

- Chunk appears in both lists → gets two additive contributions → ranked higher
- Chunk in only one list → gets one contribution (the other term = 0, not penalised by negative score)
- Raw BM25/FAISS scores are never used — only rank position

**What k=60 does numerically:**

| k value | rank #1 score | rank #2 score | rank #1 advantage |
| --- | --- | --- | --- |
| k=0 | 1.000 | 0.500 | 2× |
| k=10 | 0.091 | 0.083 | 1.09× |
| k=60 | 0.0164 | 0.0161 | 1.02× |
| k=200 | 0.005 | 0.0049 | 1.005× |

At k=60, being #1 vs #2 in one list barely matters. What matters is appearing in both lists. This is the correct behaviour for our use case — we want consensus between BM25 and FAISS, not dominance of one list's top result.

---

### Cross-Encoder Internals (ms-marco-MiniLM-L-6-v2)

**Why it is fundamentally different from bi-encoders (FAISS):**

FAISS (bi-encoder): encode query → vector, encode chunk → vector, compare independently. The two sequences never interact during encoding. Score = dot product of two independent representations.

Cross-encoder: concatenate query + chunk into one sequence, run a single BERT forward pass. Every token in the query attends to every token in the chunk via self-attention. The model actually reads them together.

**At inference (scoring 20 candidates):**

1. For each candidate chunk, construct input:
   ```
   [CLS] <query text> [SEP] <chunk text> [SEP]
   ```
2. Forward pass through 6-layer MiniLM transformer (full self-attention across the combined sequence)
3. Linear classifier head on the `[CLS]` token → single relevance logit (higher = more relevant)
4. Collect 20 logits, sort descending, return top-3 chunk_ids

**Why it catches what RRF misses:**

RRF rewarded Doc B (the 2022 IT VPN memo) partly because BM25 ranked it #1. The cross-encoder reads: "What is the 2026 remote work policy?" + "...old VPN configuration memo from 2022..." and its attention heads detect the year mismatch and policy-vs-config mismatch. Score drops. Doc A (the actual 2026 HR manual) scores high because the meaning aligns throughout. Cross-encoder corrects RRF's equal-weight flaw.

**CPU cost math:**

- MiniLM-L-6-v2 has 22M parameters, 6 layers, 384 hidden dim
- Each forward pass (query + one chunk, ~500 tokens combined) ≈ 4ms on CPU
- 20 candidates × 4ms = ~80ms total
- TTFT saving from shrinking context 20 chunks → 3 chunks: typically 300–600ms on a 13B LLM
- Net: spend 80ms on CPU, save 300–600ms on GPU. Always worth it.
