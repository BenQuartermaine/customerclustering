from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = joblib.load("model.joblib")

#create notebook
#load joblib (sklearn pipeline once loaded, takes normal pipeline functions like predict)
#give it 1 - X rows of training data
#test if it works
#incorporate into predict 




@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict")
def func(Product, 
            Status, 
            userID, 
            stripeCustID, 
            num_subs, 
            account_age, 
            pProfileID, 
            typeOfPractice, 
            located, 
            specialities, 
            population, 
            focus, 
            complex, 
            autonomy, 
            access, 
            startDate, 
            endDate, 
            createDate, 
            country, 
            favActivityType, 
            secondFavActivityType, 
            minPerYear, 
            percentageOfLearningFromAusmed, 
            numQueued, 
            numCompletedFromQueue, 
            minQueued, 
            minCompleted, 
            RatioOfCompletion_num, 
            RatioOfCompletion_min, 
            event_cpd_day_diff, 
            doc_in_activation, 
            activated, 
            plan_type, 
            subscribe_days, 
            GoalsPerYear, 
            ratioOfAchivedGoals, 
            metaGoalTitle):
    dict(Product, Status, userID, stripeCustID, num_subs, account_age, pProfileID, typeOfPractice, located, specialities, population, focus, complex, autonomy, access, startDate, endDate, createDate, country, favActivityType, secondFavActivityType, minPerYear, percentageOfLearningFromAusmed, numQueued, numCompletedFromQueue, minQueued, minCompleted, RatioOfCompletion_num, RatioOfCompletion_min, event_cpd_day_diff, doc_in_activation, activated, plan_type, subscribe_days, GoalsPerYear, ratioOfAchivedGoals, metaGoalTitle)
    # model.predict(pd.dataFrane(dict))
    return 'hello'