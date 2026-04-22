# PDF Chatbot with RAG for Document QA

Local question answering over a single text-based PDF. The app reads one PDF at a time, chunks it, embeds it, retrieves relevant evidence, reranks that evidence, answers only from the document, and returns a citation to the source page/chunk.

## MVP Scope

### Version 1 does

- load one text-based PDF at a time
- extract and clean page text locally
- create chunks with page-level metadata
- build embeddings and a FAISS index locally
- retrieve and rerank supporting chunks
- generate an answer grounded in retrieved text
- show the source page and chunk
- refuse unsupported questions with an out-of-scope message
- keep processing fully local

### Version 1 does not

- process multiple PDFs together
- handle scanned/image-only PDFs without OCR
- answer from outside the uploaded PDF
- use cloud APIs or external search
- reliably reason over complex visual layouts, tables, or charts

## Planned Architecture

```text
PDF -> page extraction -> cleaning -> chunking -> embeddings -> FAISS retrieval
   -> reranking -> QA model -> answer + citation / out-of-scope
```

## Proposed Project Structure

```text
pdf-chatbot-with-rag-for-document-qa/
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── main.py
├── app.py
├── pdf_reader.py
├── chunker.py
├── embedder.py
├── retriever.py
├── reranker.py
├── qa_engine.py
├── scope_checker.py
├── Project/
│   ├── Data/
│   │   ├── source/
│   │   ├── extracted/
│   │   └── normalized/
│   ├── tests/
│   │   └── test_smoke.py
│   ├── main.py
│   ├── pdf_reader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── qa_engine.py
│   ├── scope_checker.py
│   └── app.py
└── assets/
    └── screenshots/
```

## Core Libraries

- `PyMuPDF` for PDF text extraction
- `sentence-transformers` for chunk embeddings
- `faiss-cpu` for vector search
- `sentence-transformers` cross-encoder for reranking
- `transformers` for extractive QA
- `torch` for model execution
- `numpy` for vector operations
- `gradio` for the local interface
- `pytest` for smoke tests

## Initial Module Responsibilities

- `pdf_reader.py`: extract raw page text, normalize it page by page, skip empty pages after cleaning
- `chunker.py`: build page-bounded chunks with overlap and metadata
- `embedder.py`: load the embedding model and encode chunks/questions
- `retriever.py`: build and query the FAISS index
- `reranker.py`: reorder retrieved chunks by query relevance
- `qa_engine.py`: produce an answer from top evidence
- `scope_checker.py`: decide answer vs abstain
- `app.py`: local UI for upload, question input, answer, and citation
- `main.py`: CLI or smoke-entry script

## Expected Output Format

For supported questions:

```text
Answer: <grounded answer>
Source: page <n>, chunk <id>
```

For unsupported questions:

```text
This is outside the scope of the PDF.
```

## Local Setup Plan

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python Project/main.py --pdf <your-file.pdf>
```

## Suggested Starter `requirements.txt`

```text
PyMuPDF
sentence-transformers
faiss-cpu
transformers
torch
numpy
gradio
pytest
tqdm
```

## Milestones

1. Define the MVP clearly.
2. Create the runnable project skeleton.
3. Ingest the PDF page by page.
4. Chunk and annotate the text.
5. Generate embeddings.
6. Build retrieval.
7. Add reranking.
8. Generate grounded answers.
9. Add abstention logic.
10. Evaluate with in-scope and out-of-scope questions.
11. Wrap it in a local UI.
12. Finalize documentation and screenshots.

## Success Criteria

The first version is successful when:

- the app handles one text PDF end to end
- answers are grounded in retrieved evidence
- the source page/chunk is returned
- unsupported questions are rejected instead of hallucinated
- everything runs locally on the machine
