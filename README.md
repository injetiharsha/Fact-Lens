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

## Project Structure

- `config/`: Configuration files (weights, routing, thresholds)
- `checkpoints/`: Pre-trained model weights
- `data/`: Raw datasets and processed splits
- `pipeline/`: Core verification logic
  - `claim_pipeline.py`: claim-level pipeline entry
  - `document_pipeline.py`: document-level pipeline entry
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

## License

MIT
