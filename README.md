# RFC: Fact-Checking & Claim Verification System

A multi-model, multi-language pipeline for verifying claims using contextual evidence, relevance scoring, and stance detection.

## Features

- **Claim Analysis**: Checkability scoring and context classification
- **Two Pipeline Entrypoints**: `pipeline.claim_pipeline.ClaimPipeline` and `pipeline.document_pipeline.DocumentPipeline`
- **Evidence Retrieval**: Multi-source API and web search aggregation
- **Relevance Scoring**: Cross-encoder-based evidence prioritization
- **Stance Detection**: NLI models for support/refute/neutral classification
- **Multi-Language Support**: English and Indic language models
- **OCR Support**: Image-to-text processing with Tesseract
- **PDF Support**: Text-based PDF extraction for claim/document analysis
- **REST API**: FastAPI-based HTTP interface

## Quick Start

1. **Create 2 virtual environments (recommended)**:
   ```bash
   python -m venv .venv
   python -m venv .venv-gpu
   ```

2. **Install normal/runtime dependencies in `.venv`**:
   ```bash
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

3. **Install GPU/training dependencies in `.venv-gpu`**:
   ```bash
   .venv-gpu\Scripts\python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
   .venv-gpu\Scripts\python -m pip install -r requirements-gpu.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Prepare data**:
   ```bash
   bash scripts/download_datasets.sh
   python scripts/prepare_data.py
   ```

6. **Train models** (optional, use `.venv-gpu`):
   ```bash
   .venv-gpu\Scripts\python training/train_checkability.py
   .venv-gpu\Scripts\python training/train_relevance.py
   .venv-gpu\Scripts\python training/train_stance.py
   ```

7. **Run API** (use `.venv`):
   ```bash
   .venv\Scripts\python api/main.py
   ```

## API Usage

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d {
    "claim": "The moon is made of cheese",
    "language": "en"
  }
```

### Image Analysis
```bash
curl -X POST http://localhost:8000/api/analyze-image \
  -F "image=@sample.png" \
  -F "language=auto"
```

### PDF Analysis
```bash
curl -X POST http://localhost:8000/api/analyze-pdf \
  -F "pdf=@sample.pdf" \
  -F "language=auto"
```

## Project Structure

- `config/`: Configuration files (weights, routing, thresholds)
- `checkpoints/`: Pre-trained model weights
- `data/`: Raw datasets and processed splits
- `pipeline/`: Core verification logic
  - `claim_pipeline.py`: claim-level pipeline entry
  - `document_pipeline.py`: document-level pipeline entry
  - `ingestion/`: input extraction layer (image/pdf)
- `training/`: Model training scripts
- `api/`: REST API implementation
- `tests/`: Unit and integration tests
- `benchmarks/`: Evaluation scripts and results

## Current Roadmap Focus

- Context model (EN + Indic MT) is trained and promoted.
- Next top priority is stance training refresh:
  - EN staged curriculum: `MNLI -> FEVER -> VitaminC`
  - Indic path: `IndicNLI + translated FEVER`
  - detailed plan is in `training/README.md`.

## Current Model Status (April 2026)

- EN stance staged model is completed (`MNLI -> FEVER -> VitaminC`).
- Multi-Indic stance run finalized under:
  - `checkpoints/stance/multi/multi-indic-fever/checkpoint-11000`
  - evaluation artifacts under `checkpoints/stance/multi/multi-indic-fever/eval_checkpoint-11000`
- LLM verifier is kept ON by default for better benchmark accuracy.

## Runtime Notes

- Use `.venv-gpu` for training/eval tasks.
- If Playwright scraping is needed, run:
  - `.venv-gpu\Scripts\python -m playwright install chromium`
- PDF warnings:
  - Scanned/image-only PDFs may return low/empty extracted text without OCR fallback.
  - Default extraction is intentionally limited (`PDF_MAX_PAGES=5`, `PDF_MAX_EXTRACTED_CHARS=30000`) to keep runtime/cost bounded.
  - If claims get truncated, increase limits gradually instead of unlimited extraction.

## License

MIT
