# Retrieval-Augmented Generation (RAG) — Overview

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI framework that enhances Large Language Model (LLM) outputs by grounding them in relevant external knowledge. Instead of relying solely on a model's pre-trained weights, RAG retrieves factual context from a curated document store before generating a response.

RAG was introduced by Lewis et al. (2020) in the paper *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* and has since become a foundational pattern for enterprise AI applications.

## Why RAG?

Standard LLMs have several limitations that RAG addresses:

- **Knowledge cutoff**: LLMs freeze their knowledge at training time. RAG enables up-to-date, domain-specific answers.
- **Hallucination**: Without a factual grounding signal, LLMs may invent plausible-sounding but incorrect facts. RAG anchors answers to retrieved passages.
- **Auditability**: RAG responses include source citations, making it easy to verify and trace the origin of every claim.
- **Cost-effective specialisation**: Fine-tuning a model on proprietary data is expensive. RAG achieves similar specialisation at a fraction of the cost.

## How RAG Works

A RAG pipeline has two distinct phases:

### 1. Indexing Phase (offline)

1. **Document ingestion** — Raw documents (PDF, TXT, Markdown, CSV) are loaded and parsed.
2. **Chunking** — Documents are split into overlapping chunks (e.g., 800 characters with 150-character overlap) to stay within the embedding model's context window.
3. **Embedding** — Each chunk is converted into a dense vector representation using an embedding model (e.g., `text-embedding-3-small`).
4. **Vector store** — Vectors are indexed in a fast approximate-nearest-neighbour store such as FAISS, Pinecone, or Chroma.

### 2. Query Phase (online)

1. **Query embedding** — The user's question is embedded using the same model.
2. **Retrieval** — The top-k most similar chunks are retrieved via cosine similarity (or MMR for diversity).
3. **Context assembly** — Retrieved passages are concatenated into a context string with source metadata.
4. **Generation** — An LLM (e.g., GPT-4o) generates an answer grounded exclusively in the retrieved context.
5. **Citation** — The response includes references to source filenames and page numbers.

## Chunking Strategies

Chunking strategy significantly affects retrieval quality:

| Strategy | Description | Best for |
|---|---|---|
| **Recursive** | Splits on paragraph, sentence, then word boundaries | Mixed content, general use |
| **Sentence** | Splits on sentence boundaries using NLTK | Long-form prose, articles |
| **Fixed** | Splits at exact character count | Structured data, speed-critical |

Larger chunks preserve more context but reduce recall precision. Smaller chunks improve precision but may lose surrounding context. An overlap of 10–20% of the chunk size helps bridge splits.

## Retrieval Techniques

### Cosine Similarity Search
Chunks are ranked by the cosine similarity between their embedding and the query embedding. L2-normalised vectors allow this to be computed efficiently as an inner product.

### Maximal Marginal Relevance (MMR)
MMR balances relevance with diversity. Given a retrieved candidate set, MMR iteratively selects the chunk that maximises:

```
MMR = λ · sim(chunk, query) − (1 − λ) · max_sim(chunk, already_selected)
```

A higher λ prioritises relevance; lower λ prioritises diversity. MMR prevents the model from receiving five near-identical passages.

## Confidence Scoring

A confidence score can be computed from the raw similarity scores of retrieved chunks:

```
confidence = (1.0 × score_1 + 0.6 × score_2 + 0.3 × score_3) / (1.0 + 0.6 + 0.3)
```

Scores below a threshold (e.g., 0.25) trigger a graceful fallback response rather than a low-quality grounded answer.

## Common RAG Failure Modes

1. **Chunking too small** — Critical context is split across chunk boundaries, degrading retrieval.
2. **No deduplication** — Near-identical chunks waste the LLM's context window.
3. **Prompt leakage** — Poorly designed prompts allow the LLM to use outside knowledge despite a grounding instruction.
4. **Score misinterpretation** — Raw inner-product scores from different embedding models are not directly comparable.
5. **Stale index** — The vector store is not updated when source documents change.

## Further Reading

- Lewis et al. (2020): *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — https://arxiv.org/abs/2005.11401
- Gao et al. (2023): *Retrieval-Augmented Generation for Large Language Models: A Survey* — https://arxiv.org/abs/2312.10997
- LangChain documentation: https://docs.langchain.com
- FAISS documentation: https://faiss.ai
