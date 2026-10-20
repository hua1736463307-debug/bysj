CREATE TABLE IF NOT EXISTS job_tasks (
    id VARCHAR(36) PRIMARY KEY,
    type VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    stage VARCHAR(64) NOT NULL DEFAULT 'queued',
    progress DOUBLE PRECISION NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    owner_id VARCHAR(64),
    owner_username VARCHAR(64) NOT NULL,
    worker_pid INTEGER,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    error_code VARCHAR(64),
    error_detail TEXT,
    input_json JSON NOT NULL DEFAULT '{}',
    result_json JSON NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    heartbeat_at TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_job_tasks_type ON job_tasks(type);
CREATE INDEX IF NOT EXISTS ix_job_tasks_status ON job_tasks(status);
CREATE INDEX IF NOT EXISTS ix_job_tasks_owner_id ON job_tasks(owner_id);
CREATE INDEX IF NOT EXISTS ix_job_tasks_owner_username ON job_tasks(owner_username);
CREATE INDEX IF NOT EXISTS ix_job_tasks_cancel_requested ON job_tasks(cancel_requested);
CREATE INDEX IF NOT EXISTS ix_job_tasks_heartbeat_at ON job_tasks(heartbeat_at);

CREATE TABLE IF NOT EXISTS job_events (
    id BIGSERIAL PRIMARY KEY,
    task_id VARCHAR(36) NOT NULL,
    sequence INTEGER NOT NULL,
    level VARCHAR(16) NOT NULL DEFAULT 'info',
    event_type VARCHAR(64) NOT NULL DEFAULT 'progress',
    message TEXT NOT NULL DEFAULT '',
    payload_json JSON NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_job_event_sequence UNIQUE(task_id, sequence)
);
CREATE INDEX IF NOT EXISTS ix_job_events_task_id ON job_events(task_id);
CREATE INDEX IF NOT EXISTS ix_job_events_event_type ON job_events(event_type);
CREATE INDEX IF NOT EXISTS ix_job_events_created_at ON job_events(created_at);
