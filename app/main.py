from fastapi import FastAPI, BackgroundTasks
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from model.model import attach_drug_name, response, predictions, predictions_two
from app.predictions import api_names

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

@app.get(api_names[0])
async def predict(background_tasks: BackgroundTasks,solute,solvent):
    background_tasks.add_task(predictions,solute,solvent)
    return {'success'}

@app.get(api_names[1])
async def post():
    return {'result': response}
    
@app.get(api_names[2])
async def predict_two(background_tasks: BackgroundTasks,solute):
    background_tasks.add_task(predictions_two,solute)
    return {'success'}

@app.get(api_names[3])
async def post():
    return {'result': attach_drug_name()}

if __name__ == "__main__":
    uvicorn.run("app.main:app", port=8000)
 