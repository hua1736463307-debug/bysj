# Optional offline dependency stage. Supply --build-context wheelhouse=<verified wheels>.
# This starts from clean Python, never from another application's runtime.
ARG PYTHON_BASE=python:3.12-slim@sha256:dd29372629eeba2dd003fd9e9d35a5b8236c44727875a0364254b5127af88e65
FROM ${PYTHON_BASE}
COPY backend/requirements-core-linux-py312.lock.txt /locks/core.txt
RUN --mount=type=bind,from=wheelhouse,target=/wheelhouse \
    python -m pip install --no-index --find-links=/wheelhouse --require-hashes -r /locks/core.txt \
    && python -m pip check
