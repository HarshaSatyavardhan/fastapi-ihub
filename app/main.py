from fastapi import FastAPI, BackgroundTasks
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from model.model import attach_drug_name, response, predictions, predictions_two
from app.predictions import api_names, predict_dum

# from mangum import Mangum
import os

# stage = os.environ.get('STAGE', None)
# openapi_prefix = f"/{stage}" if stage else "/"
# openapi_prefix=openapi_prefix

DEVICE = "cuda"
app = FastAPI()
origins = ["*"]

# handler = Mangum(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(api_names[0])
async def predict(background_tasks: BackgroundTasks,solute,solvent):
    
    background_tasks.add_task(predict_dum.get(0),solute,solvent)
    return {'success'}

@app.get(api_names[1])
async def post():
    return {'result': response}
    
@app.get(api_names[2])
async def predict_two(background_tasks: BackgroundTasks,solute):
    background_tasks.add_task(predict_dum.get(1),solute)
    return {'success'}

@app.get(api_names[3])
async def post():
    return {'result': attach_drug_name()}

if __name__ == "__main__":
    uvicorn.run("app.main:app", port=8000)