import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

# Build mock artifacts BEFORE app.config is imported (it reads env at import time).
_TMP = Path(tempfile.mkdtemp(prefix="rainfall_mock_"))
os.environ["DATA_ROOT"] = str(_TMP)
os.environ["SHOW_PROGRESS"] = "0"
os.environ["FRONTEND_DIST"] = str(_TMP / "no_frontend")

import make_mock_artifacts  # noqa: E402

make_mock_artifacts.generate(_TMP)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:   # runs the lifespan (loads models)
        yield c
