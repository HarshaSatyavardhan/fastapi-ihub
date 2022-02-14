from httpx import AsyncClient
import pytest
from main import app
data = ["success"]
data_two = {"result":{"interaction_map":[[14.0,5.0,15.0,15.0,15.0],[20.0,7.0,19.0,19.0,19.0],[14.0,6.0,13.0,13.0,13.0],[14.0,5.0,15.0,15.0,15.0],[14.0,5.0,15.0,15.0,15.0]],"predictions":-3.4333341121673584}}
solute = 'CC(C)(C)Br'
solvent = 'CC(C)(C)O'

@pytest.mark.anyio
async def test_root():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get('/predict?solute=CC(C)(C)Br&solvent=CC(C)(C)O')
        response_two = await ac.get('/predict_solubility')
    assert response.status_code == 200
    assert response_two.status_code == 200
    assert response.json() == data
    assert response_two.json() == data_two

    