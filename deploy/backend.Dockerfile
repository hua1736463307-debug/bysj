# 毕业版后端镜像：默认只装核心依赖；worker target 额外安装 ML 依赖。
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.lock.txt /app/requirements.lock.txt
RUN python -m pip install --require-hashes -r /app/requirements.lock.txt

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
ARG INSTALL_ML=true
RUN if [ "$INSTALL_ML" = "true" ] && [ -f /app/backend/requirements-ml.lock.txt ]; then \
        python -m pip install --require-hashes -r /app/backend/requirements-ml.lock.txt; \
    else \
        echo "ML lock missing: worker 只能执行 dummy 后端" >&2; \
    fi
CMD ["python", "-m", "app.workers.trainer_worker"]
