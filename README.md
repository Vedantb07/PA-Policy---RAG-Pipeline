# PA-Policy---RAG-Pipeline

# README

## Overview

This pipeline automatically extracts Prior Authorization (PA) policy criteria from PDF documents, calculates an Access Score, and writes the results back to an Excel file.

---

## Required Dependencies

Install the required packages before running the pipeline:

```bash
pip install pymupdf pdfplumber rapidfuzz rank-bm25
pip install sentence-transformers faiss-cpu transformers torch
pip install groq
pip install pandas numpy openpyxl
```

---

## Models Used

### Embedding Model

* **BAAI/bge-large-en-v1.5**


### Reranker Model

* **BAAI/bge-reranker-base**


### LLM

* Current model in pipeline - **llama-3.3-70b-versatile (groq)**
(use llama 8B parameters if have you have more than 8 test pdfs and needs to finished in one go)

---

## Configuration Required Before Running

Update the following variables before executing the pipeline:

```python
EXCEL_PATH = "path/to/input_excel.xlsx"
PDF_FOLDER = "path/to/pdf_folder"
SHEET_NAME = "Submissions"
GROQ_API_KEY = "your_groq_api_key"
```



