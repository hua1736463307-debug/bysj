# Optional offline dependency stage. Supply --build-context wheelhouse=<verified wheels>.
# This starts from clean Python, never from another application's runtime.
ARG PYTHON_BASE=python:3.12-slim
FROM ${PYTHON_BASE}
COPY backend/requirements-core-linux-py312.lock.txt /locks/core.txt
RUN --mount=type=bind,from=wheelhouse,target=/wheelhouse \
    python -m pip install --no-index --find-links=/wheelhouse --require-hashes -r /locks/core.txt \
    && python -m pip check
