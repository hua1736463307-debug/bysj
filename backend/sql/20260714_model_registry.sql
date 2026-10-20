CREATE TABLE IF NOT EXISTS model_registry (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    path TEXT NOT NULL UNIQUE,
    sha256 VARCHAR(64) NOT NULL,
    size BIGINT NOT NULL DEFAULT 0,
    source VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    format VARCHAR(32) NOT NULL DEFAULT 'pt',
    task_id VARCHAR(36),
    dataset_version_id VARCHAR(36),
    created_by VARCHAR(64) NOT NULL DEFAULT 'system',
    metadata_json JSON NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_model_registry_name ON model_registry(name);
CREATE INDEX IF NOT EXISTS ix_model_registry_sha256 ON model_registry(sha256);
CREATE INDEX IF NOT EXISTS ix_model_registry_source ON model_registry(source);
CREATE INDEX IF NOT EXISTS ix_model_registry_status ON model_registry(status);
CREATE INDEX IF NOT EXISTS ix_model_registry_task_id ON model_registry(task_id);
CREATE INDEX IF NOT EXISTS ix_model_registry_dataset_version_id ON model_registry(dataset_version_id);
CREATE INDEX IF NOT EXISTS ix_model_registry_created_at ON model_registry(created_at);
