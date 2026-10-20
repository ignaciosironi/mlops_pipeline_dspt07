"""FastAPI service for payment-on-time predictions (Advance 3)."""
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"

class CreditRecord(BaseModel):
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tipo_laboral: str
    tendencia_ingresos: str
    tipo_credito: str

class PredictionBatch(BaseModel):
    data: list[CreditRecord] = Field(min_length=1)

app = FastAPI(title="Credit Payment Prediction", version="1.0.0")

@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Credit prediction service", "docs": "/docs"}

@app.get("/health")
def health():
    return {"model_file_available": MODEL_PATH.is_file()}

@app.post("/predict")
def predict(batch: PredictionBatch):
    try:
        records = [record.model_dump() for record in batch.data]
        frame = pd.DataFrame.from_records(records)
        predictions = get_model().predict(frame)
        return {"predictions": [value.item() if hasattr(value, "item") else value for value in predictions]}
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {error}") from error

@app.get("/evaluation")
def evaluation_image():
    try:
        # Import lazily so a missing or incompatible evaluation module does not prevent predictions.
        from src.model_evaluation_avance2 import evaluation
        image = evaluation()
        return Response(content=image.getvalue(), media_type="image/png")
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Evaluation image unavailable: {error}") from error
