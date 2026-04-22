# PDF QA Project Plan

## Project Goal

Build a fully local question-answering app for **one text-based PDF at a time**. The system should answer questions only from the uploaded PDF, return the supporting page/chunk, and refuse unsupported questions with a clear out-of-scope response.

## Phase 1 - Define the MVP

### Version 1 Scope

Version 1 will:

- support one text-based PDF at a time
- extract text page by page
- split text into searchable chunks
- embed chunks locally
- retrieve relevant chunks with FAISS
- rerank retrieved chunks locally
- answer questions only from retrieved evidence
- return the answer with page number and chunk id
- say `"This is outside the scope of the PDF"` when evidence is insufficient
- run fully offline after local setup

Version 1 will not:

- support scanned/image-only PDFs without OCR
- support multiple PDFs in a shared knowledge base
- support tables, figures, charts, or layout-aware reasoning
- support citations beyond page/chunk text evidence
- support web search or external APIs
- guarantee perfect answers when the PDF text extraction is poor

### Folder Structure

```text
pdf-qa/
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
├── data/
│   ├── input/
│   ├── processed/
│   └── indexes/
├── tests/
│   ├── test_smoke.py
│   └── sample_questions.json
└── assets/
    └── screenshots/
```

### Planned Libraries

- `PyMuPDF` or `pypdf` for PDF text extraction
- `sentence-transformers` for embeddings and reranking
- `faiss-cpu` for vector search
- `transformers` for extractive QA
- `torch` as model runtime
- `numpy` for vector handling
- `gradio` or `streamlit` for a local UI
- `pytest` for testing
- `tqdm` for progress display

### Done When

You can explain exactly:

- what the first version does
- what it refuses to do
- what libraries it uses
- how the files are organized

## Phase 2 - Environment and Project Setup

### Tasks

- create project folder
- create Python virtual environment
- install libraries
- create starter files
- set up a simple test script

### Suggested Files

- `main.py`
- `pdf_reader.py`
- `chunker.py`
- `embedder.py`
- `retriever.py`
- `reranker.py`
- `qa_engine.py`
- `scope_checker.py`
- `ui.py` or `app.py`

### Deliverable

Runnable project skeleton

### Done When

The project runs without import errors.

## Phase 3 - PDF Ingestion

### Tasks

- load PDF
- extract text page by page
- preserve page numbers
- ignore empty pages
- clean raw text

### Deliverable

Structured page data:

- page number
- page text

### Done When

You can print extracted text for each page correctly.

## Phase 4 - Text Cleaning and Chunking

### Tasks

- normalize spaces/newlines
- split long text into chunks
- decide chunk size
- decide overlap
- assign metadata to each chunk:
  - chunk id
  - page number
  - source file

### Deliverable

List of chunks with metadata

### Done When

One PDF becomes a clean chunk dataset.

## Phase 5 - Embedding Generation

### Tasks

- load embedding model
- encode each chunk into embeddings
- verify output dimensions
- save embeddings in memory or file

### Deliverable

Embeddings for all chunks

### Done When

Every chunk has a usable vector representation.

## Phase 6 - Vector Indexing and Retrieval

### Tasks

- build FAISS index
- insert chunk embeddings
- save mapping from vector id to chunk metadata
- accept user question
- embed the question
- retrieve top-k nearest chunks

### Deliverable

Working retrieval system

### Done When

A question returns relevant chunks from the PDF.

## Phase 7 - Reranking

### Tasks

- take top retrieved chunks
- score query-chunk pairs with reranker
- reorder chunks by relevance
- select best 1-3 chunks for final answering

### Deliverable

Better ranked evidence chunks

### Done When

Retrieved context is noticeably more relevant than raw vector search alone.

## Phase 8 - Answer Generation

### Tasks

- pass question + top chunk(s) to QA model
- extract answer span
- keep the answer grounded in retrieved text
- return page/chunk citation with answer

### Deliverable

Answer + citation

### Done When

The system answers correctly for in-scope questions.

## Phase 9 - Out-of-Scope Detection

### Tasks

- define retrieval confidence threshold
- define reranker threshold
- use QA model confidence / no-answer behavior
- combine these signals into a decision rule
- return:
  - answer if supported
  - `"This is outside the scope of the PDF"` if unsupported

### Deliverable

Reliable abstention logic

### Done When

Unsupported questions do not get fake answers.

## Phase 10 - Evaluation and Testing

### Tasks

- create test questions from the PDF
- create out-of-scope questions
- measure:
  - retrieval quality
  - answer correctness
  - abstention correctness
- tune chunk size, overlap, top-k, thresholds

### Deliverable

Test set and evaluation notes

### Done When

You know the strengths and weaknesses of the system.

## Phase 11 - User Interface

### Tasks

- PDF upload
- question input box
- answer display
- citation display
- out-of-scope message
- loading/error states

### Deliverable

Usable local app

### Done When

Someone else can use it without help.

## Phase 12 - Documentation and Presentation

### Tasks

- write README
- explain architecture
- explain models used
- explain out-of-scope logic
- add screenshots
- write resume bullet
- prepare interview explanation

### Deliverable

GitHub-ready project

### Done When

You can present it as a serious ML/NLP project.

## Recommended First Technical Choices

To keep MVP complexity under control:

- use `PyMuPDF` for page-wise extraction
- use `sentence-transformers/all-MiniLM-L6-v2` for chunk embeddings
- use `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking
- use a lightweight extractive QA model from `transformers`
- use `faiss-cpu` for local vector search
- use `gradio` for the first local UI

These choices are practical, well-supported, and small enough for an MVP on a local machine.
