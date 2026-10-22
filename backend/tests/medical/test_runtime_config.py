from pathlib import Path
import pytest
from app.core.config import Settings, REPO_ROOT, DEV_JWT_SECRET


def test_config_dotenv_location_is_repository_root():
    assert REPO_ROOT == Path(__file__).resolve().parents[3]
    assert Settings.model_config['env_file'][-1] == str(REPO_ROOT/'deploy'/'.env')


@pytest.mark.parametrize('override', [
    {'database_url':''}, {'database_url':'sqlite:///test.db'},
    {'inline_jobs':True}, {'jwt_secret_key':DEV_JWT_SECRET}, {'jwt_secret_key':''},
])
def test_production_rejects_development_fallbacks(override):
    values={'app_env':'prod','database_url':'postgresql+psycopg://test:test@localhost/test',
            'jwt_secret_key':'config-test-secret-0123456789abcdef','inline_jobs':False}
    with pytest.raises(ValueError):Settings(_env_file=None,**{**values,**override})
