from fastapi import FastAPI

from agent.config import get_settings

app = FastAPI(title="Astro Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": get_settings().app_env}
