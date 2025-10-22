from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from . import predict

class MarketData(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int
    SMA_20: float
    SMA_100: float
    RSI_14: float
    BB_UPPER: float
    BB_MIDDLE: float
    BB_LOWER: float
    
class SequenceInput(BaseModel):
    sequence: List[MarketData]
    
app = FastAPI(title="Quant-Trading-Sequential-Prediction API", description="API REST pour la prédiction de signaux.", version="1.0.0")
    
@app.on_event("startup")
def startup_event():
    predict.load_attributes()
    if predict.model is None or predict.scaler is None:
        raise RuntimeError("Failed to load model or scaler during startup.")

@app.post("/signal", summary="Get trading signal")
async def get_trading_signal(input_data: SequenceInput):
    try:
        data_to_predict = [item.model_dump() for item in input_data.sequence]
        
        proba, signal = predict.predict_signal(data_to_predict)
        
        return {"prediction_probability": proba, "trading_signal": signal, "interpretation": "Buy" if signal == 1 else "Hold/Sell", "threshold": predict.PREDICTION_THRESHOLD}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error.")
    
    
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
    