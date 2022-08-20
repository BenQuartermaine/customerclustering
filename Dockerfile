FROM python:3.8.12-buster 

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY customerclustering customerclustering
COPY api api
COPY model.joblib model.joblib

CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT