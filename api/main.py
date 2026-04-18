"""FastAPI application entry point."""

import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from dotenv import load_dotenv

# Allow `python api/main.py` by ensuring project root is importable.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

load_dotenv(override=True)

app = FastAPI(title="RFC Fact-Checking API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Import routes
from api.routes.claim import router as claim_router
from api.routes.health import router as health_router
from api.routes.claim import preload_claim_pipelines

app.include_router(claim_router)
app.include_router(health_router)


@app.on_event("startup")
async def preload_models_on_startup() -> None:
    preload_claim_pipelines()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
