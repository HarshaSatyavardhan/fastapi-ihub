from httpx import AsyncClient
import pytest
from app.main import app
from model.ml_model import predictions,response
data = ["success"]
data_two_here = {"result":{"interaction_map":[[15.0,5.0,14.0,15.0,15.0],[19.0,7.0,20.0,19.0,19.0],[13.0,6.0,18.0,13.0,13.0],[15.0,5.0,14.0,15.0,15.0],[15.0,5.0,14.0,15.0,15.0]],"predictions":-3.405024290084839}}
solute = 'CC(C)(C)Br'
solvent = 'CC(C)(C)Br'

@pytest.mark.anyio
async def test_root():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get('/predict?solute=CC(C)(C)Br&solvent=CC(C)(C)Br')
        response_two = await ac.get('/predict_solubility')
    assert response.status_code == 200
    assert response_two.status_code == 200
    assert response.json() == data
    assert response_two.json() == data_two_here


async def test_predict():
    res = await predictions(solute, solvent)
    response_final = response
    assert response_final == data_two_here

    