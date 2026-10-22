# Runtime-only assembly from separately validated clean core or ML dependencies.
# Both dependency images must originate from the pinned clean Python base.
ARG DEPENDENCY_IMAGE=bysj-core-deps:20261022
FROM ${DEPENDENCY_IMAGE}
ARG BYSJ_CODE_SHA
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/backend BYSJ_CODE_SHA=${BYSJ_CODE_SHA}
WORKDIR /app
COPY backend/app /app/backend/app
COPY backend/migrations /app/backend/migrations
COPY backend/alembic.ini /app/backend/alembic.ini
COPY backend/requirements-core-linux-py312.lock.txt backend/requirements-ml-linux-py312-cu124.lock.txt /app/backend/
RUN python - <<'PY'
import hashlib,json,os,pathlib,re
sha=os.environ['BYSJ_CODE_SHA']
assert re.fullmatch('[0-9a-f]{40}',sha), 'exact source commit required'
payload={'code_sha':sha,'schema_version':1,'image':'bysj-clean-runtime'}
pathlib.Path('/app/version.json').write_text(json.dumps(payload),encoding='utf-8')
import app
assert pathlib.Path(app.__file__).resolve().is_relative_to('/app/backend')
PY
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
