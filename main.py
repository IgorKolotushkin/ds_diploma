import dill
import pandas as pd
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open('model_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    id: int
    result: str


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    result = model['model'].predict(df)

    return {
        'id': form.id,
        'result': result[0],
    }


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
