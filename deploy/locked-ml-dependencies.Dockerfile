# All Python artifacts are supplied by a verified, external wheelhouse context.
# Example: --build-context wheelhouse=.runtime/wheels
ARG CORE_BASE=bysj-core-deps:20261022
FROM ${CORE_BASE}
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=bind,from=wheelhouse,target=/wheels,readonly \
    python -m pip install --no-cache-dir --no-index --find-links=/wheels --require-hashes -r /wheels/ml-offline.lock.txt \
    && python -m pip check
