from httpx import AsyncClient
import pytest
from main import app
from sample_json_data import data_two
data = ["success"]

solute = 'CC(C)(C)Br'

@pytest.mark.anyio
async def test_root():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get('/predict_two?solute=CC(C)(C)Br')
        response_two = await ac.get('/predict_solubility_json')
    assert response.status_code == 200
    assert response_two.status_code == 200
    assert response.json() == data
    assert response_two.json() == data_two



    