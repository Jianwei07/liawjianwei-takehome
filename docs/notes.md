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
