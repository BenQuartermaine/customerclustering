from typing import Union

from fastapi import FastAPI
from customerclustering.get_training_data import GetTrainingData
from customerclustering.db_connection import Db

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

conn = Db.db_conn()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/test_route")
def func():
    h = GetTrainingData(conn, 30).get_training_data()
    return h 