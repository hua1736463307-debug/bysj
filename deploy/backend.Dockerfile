# 毕业版后端镜像：默认只装核心依赖；worker target 额外安装 ML 依赖。
ARG PYTHON_BASE=python:3.12-slim
FROM ${PYTHON_BASE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend

WORKDIR /app

COPY backend/requirements-core-linux-py312.lock.txt /app/requirements-core-linux-py312.lock.txt
RUN python -m pip install --require-hashes -r /app/requirements-core-linux-py312.lock.txt

COPY backend /app/backend
COPY deploy/version.json /app/version.json

RUN python - <<'PY'
import json, pathlib
path = pathlib.Path('/app/version.json')
data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
data.setdefault('image', 'bysj-backend')
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
PY

FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

FROM base AS worker
# The single opencv-python wheel declared by Ultralytics needs these runtime
# libraries even when the application never opens a desktop window.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Missing ML lock is an error: a real worker must not silently become dummy.
RUN test -s /app/backend/requirements-ml-linux-py312-cu124.lock.txt \
    && python -m pip install --require-hashes -r /app/backend/requirements-ml-linux-py312-cu124.lock.txt \
    && python -m pip check
CMD ["python", "-m", "app.workers.trainer_worker"]
