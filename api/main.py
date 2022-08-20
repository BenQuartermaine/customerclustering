from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = joblib.load("model.joblib")


class MyModel(BaseModel):
    passengers: dict

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(m: MyModel):
    t = m.dict()
    X_dict = t['passengers']
    
    X_pred = pd.DataFrame(X_dict)
    pred = model.predict(X_pred)
    
    pr_int = {key: int(val) for key, val in zip(X_dict['userID'], pred)}
    
    return {'predictions': pr_int}

    
