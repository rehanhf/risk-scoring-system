from fastapi import FastAPI, HTTPException
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from api.schema import CreditApplication

# globals for memory persistence
preprocessor = None
model = None
OPTIMAL_THRESHOLD = 0.28


@asynccontextmanager
async def lifespan(app: FastAPI):
    global preprocessor, model
    # load strictly isolated serialized artifacts
    preprocessor = joblib.load("data/processed/preprocessor.joblib")
    model = lgb.Booster(model_file="src/models/lgbm_baseline.txt")
    yield
    # cleanup pas shutdown


app = FastAPI(lifespan=lifespan, title="Credit Risk Scoring Engine")


@app.post("/predict")
def predict_risk(app_data: CreditApplication):
    global preprocessor, model

    # strict type guard prevensi NoneType execution
    if preprocessor is None or model is None:
        raise HTTPException(
            status_code=503, detail="Models not fully loaded into memory."
        )

    try:
        df = pd.DataFrame([app_data.model_dump()])
        X_processed = preprocessor.transform(df)

        # cast explicitly prediksi ke numpy array agar memenuhi index type constraints
        preds = np.array(model.predict(X_processed))
        prob = float(preds[0])

        decision = "REJECT" if prob >= OPTIMAL_THRESHOLD else "APPROVE"

        if prob < 0.10:
            tier = "LOW"
        elif prob < OPTIMAL_THRESHOLD:
            tier = "MEDIUM"
        elif prob < 0.60:
            tier = "HIGH"
        else:
            tier = "CRITICAL"

        return {
            "probability": round(prob, 4),
            "threshold_applied": OPTIMAL_THRESHOLD,
            "decision": decision,
            "risk_tier": tier,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
