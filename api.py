"""
api.py — Sentinel-DS FastAPI backend.
Endpoints: /ingest, /probe, /verify/mia
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import os

from src.auditor import PIIAuditor
from src.transformer import SyntheticTransformer
from src.membership_inference import run_mia

app = FastAPI(title="Sentinel-DS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
if os.path.isdir("frontend"):
    app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")

auditor = PIIAuditor()
transformer = SyntheticTransformer()


class Record(BaseModel):
    id: str
    content: str

class IngestRequest(BaseModel):
    records: List[Record]

class ProbeRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 80


@app.get("/")
def root():
    return {"status": "Sentinel-DS running", "docs": "/docs"}


@app.post("/api/v1/ingest")
def ingest(request: IngestRequest):
    """Audit PII and transform records into synthetic equivalents."""
    if not request.records:
        raise HTTPException(status_code=400, detail="No records provided.")

    results = []
    for rec in request.records:
        audit_result = auditor.audit(rec.content)
        synthetic_text = transformer.transform(rec.content, audit_result)
        results.append({
            "id": rec.id,
            "pii_entities_found": len(audit_result),
            "synthetic_preview": synthetic_text[:120] + "..."
        })

    return {
        "status": "success",
        "records_processed": len(results),
        "details": results
    }


@app.post("/api/v1/probe")
def probe(request: ProbeRequest):
    """Query the locally aggregated DistilGPT-2 model."""
    from src.llm_trainer import SISATrainer
    trainer = SISATrainer()
    try:
        response = trainer.generate(request.prompt, max_new_tokens=request.max_new_tokens)
        return {"prompt": request.prompt, "response": response}
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Aggregated model not found. Run: python main.py --mode train"
        )


@app.get("/api/v1/verify/mia")
def verify_mia():
    """Run Membership Inference Attack to verify unlearning."""
    score = run_mia()
    interpretation = (
        "UNLEARNED — model has statistically forgotten the data."
        if 0.45 <= score <= 0.55
        else "WARNING — data may still be memorized."
    )
    return {
        "mia_score": round(score, 4),
        "target_range": "0.45 – 0.55",
        "interpretation": interpretation
    }