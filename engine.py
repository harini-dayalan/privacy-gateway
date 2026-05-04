"""
engine.py — Uvicorn launcher for Sentinel-DS.
Forces host=0.0.0.0 and port=8000 to prevent Windows localhost binding issues.
"""
import uvicorn

if __name__ == "__main__":
    print("[Sentinel-DS Engine] Starting API server on http://0.0.0.0:8000 ...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )