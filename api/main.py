from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

model = joblib.load("models/model_v3_pca_Kmn.joblib")

# Used to specify what data structure an endpoint should recieve
class MyModel(BaseModel):
    passengers: dict

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(m: MyModel):
    # Convert m to a dict (it comes as a dict, idk why we need to do this)
    t = m.dict()
    
    # Get passengers dict
    X_dict = t['passengers']
    
    # Setup X for prediction
    X_pred = pd.DataFrame(X_dict)
    
    # Predict
    pred = model.predict(X_pred)
    
    # Create a dict with key user ID and val prediction
    pr_int = {key: int(val) for key, val in zip(X_dict['userID'], pred)}
    
    return { 'predictions': pr_int }

    
