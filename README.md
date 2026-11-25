# PDF to Markdown Converter (Gemini)

Batch-converts PDF documents into clean Markdown using the Google GenAI SDK (`google-genai`), with automatic chunking for large PDFs, retry handling for transient errors, resumable part caching, and safe merging only after all parts succeed.

## Features

- **Folder-based batch processing:** Scans `Your_PDFs/` for `.pdf` files and skips already generated `(GeminiMarkdown) <name>.md` outputs.
- **Automatic chunking:** Splits large PDFs by page count (`MAX_PAGES_PER_CHUNK`) or file size (`MAX_SIZE_MB_PER_CHUNK`) using `pypdf`.
- **Resumable parts & safe merge:** Stores per-part Markdown files in `_md_parts_<name>/` and only creates the final merged Markdown file once every part succeeds.
- **Retry with backoff:** Retries transient errors (`429`, `500`, `503`, `UNAVAILABLE`, `RESOURCE_EXHAUSTED`, `INTERNAL`).
- **Aligned summary table:** Prints a formatted summary table at the end of each run.

## Model Configuration

- Default model: `gemini-3.8-flash` (`MODEL_NAME` in `PDFtoMarkdown.py`).
- Legacy API keys: If you have an older API key that still has access to `gemini-2.5-flash`, you can set `MODEL_NAME = "gemini-2.5-flash"` in `PDFtoMarkdown.py`.

## Prerequisites & API Key

1. Get an API key from [Google AI Studio](https://aistudio.google.com/).
2. Set the `GEMINI_API_KEY` environment variable before running the script:
   - **Windows (PowerShell):** `$env:GEMINI_API_KEY="YOUR_API_KEY"`
   - **Windows (CMD):** `set GEMINI_API_KEY=YOUR_API_KEY`
   - **Linux / macOS:** `export GEMINI_API_KEY="YOUR_API_KEY"`

## Installation & Usage

```bash
python -m venv .venv
pip install -r requirements.txt
mkdir Your_PDFs
python PDFtoMarkdown.py
```

## Author

**Milad Mehrnia** — M.Sc. Student in Mechatronics, University of Tehran  
GitHub: MiladMehr | LinkedIn: milad-mehrnia