import os

# Establece HF_HOME antes de importar transformers
os.environ["HF_HOME"] = "/tmp/hf_cache"

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline

class Req(BaseModel):
    text: str

app = FastAPI()

# Monta estáticos y sirve el index
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse("static/index.html")

# Health check
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}

# Carga pipeline SIN cache_dir
clf = pipeline(
    "text-classification",
    model="carlosalv12/deteccionem-model",
    tokenizer="carlosalv12/deteccionem-model",
    device=0
)

@app.post("/predict", tags=["predict"])
async def predict(req: Req):
    out = clf(req.text)
    top = out[0]
    return {"label": top["label"], "score": float(top["score"])}
