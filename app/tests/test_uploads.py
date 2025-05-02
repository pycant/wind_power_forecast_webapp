import pytest
from app.routes import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_allowed_file():
    assert app.config['ALLOWED_EXTENSIONS'] == {'csv'}
    assert allowed_file("data.csv") is True
    assert allowed_file("image.png") is False
