# LOCAL TEST ONLY. The parent contains another project's files; never publish this image.
ARG CANDIDATE_BASE=deepvision-backend:local
FROM ${CANDIDATE_BASE}
WORKDIR /opt/bysj
ENV PYTHONPATH=/opt/bysj/backend PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY backend/requirements.in backend/requirements-dev.in backend/constraints-candidate.txt /opt/bysj/dependencies/
RUN python -m pip install --report /opt/bysj/candidate-install-report.json \
    -r /opt/bysj/dependencies/requirements.in -r /opt/bysj/dependencies/requirements-dev.in \
    -c /opt/bysj/dependencies/constraints-candidate.txt && python -m pip check
COPY backend /opt/bysj/backend
COPY scripts /opt/bysj/scripts
COPY deploy/version.json /opt/bysj/version.json
ENTRYPOINT []
HEALTHCHECK NONE
CMD ["python", "-m", "app.workers.trainer_worker"]
