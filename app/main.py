from fastapi import FastAPI, BackgroundTasks
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from model.model import attach_drug_name, response,response_two,key_attach, predictions, predictions_two

DEVICE = "cuda"
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/predict_solubility')
async def post():
    return {'result': response}

@app.get('/predict')
async def predict(background_tasks: BackgroundTasks,solute,solvent):
    background_tasks.add_task(predictions,solute,solvent)
    return {'success'}
    
@app.get('/predict_solubility_json')
async def post():
    return {'result': attach_drug_name}


@app.get('/predict_two')
async def predict_two(background_tasks: BackgroundTasks,solute):
    background_tasks.add_task(predictions_two,solute)
    return {'success'}


if __name__ == "__main__":
    uvicorn.run("app.main:app", port=8000)
