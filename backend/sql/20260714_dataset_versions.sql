CREATE TABLE IF NOT EXISTS dataset_versions (
    id VARCHAR(36) PRIMARY KEY,
    dataset_id VARCHAR(255) NOT NULL,
    parent_version_id VARCHAR(36),
    dataset_role VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'ready',
    root_path TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    class_map_json JSON NOT NULL DEFAULT '{}',
    split_config_json JSON NOT NULL DEFAULT '{}',
    content_hash VARCHAR(64) NOT NULL,
    created_by VARCHAR(64) NOT NULL DEFAULT 'system',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_dataset_version_content UNIQUE (dataset_id, content_hash)
);

CREATE INDEX IF NOT EXISTS ix_dataset_versions_dataset_id ON dataset_versions (dataset_id);
CREATE INDEX IF NOT EXISTS ix_dataset_versions_parent_version_id ON dataset_versions (parent_version_id);
CREATE INDEX IF NOT EXISTS ix_dataset_versions_status ON dataset_versions (status);
CREATE INDEX IF NOT EXISTS ix_dataset_versions_content_hash ON dataset_versions (content_hash);

CREATE TABLE IF NOT EXISTS dataset_assets (
    id BIGSERIAL PRIMARY KEY,
    dataset_version_id VARCHAR(36) NOT NULL,
    relative_path TEXT NOT NULL,
    basename VARCHAR(255) NOT NULL,
    size BIGINT NOT NULL DEFAULT 0,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    label_path TEXT,
    source_group VARCHAR(255),
    content_hash VARCHAR(64) NOT NULL,
    annotations_json JSON NOT NULL DEFAULT '[]',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_dataset_asset_path UNIQUE (dataset_version_id, relative_path)
);

CREATE INDEX IF NOT EXISTS ix_dataset_assets_version_id ON dataset_assets (dataset_version_id);
CREATE INDEX IF NOT EXISTS ix_dataset_assets_basename ON dataset_assets (basename);
CREATE INDEX IF NOT EXISTS ix_dataset_assets_source_group ON dataset_assets (source_group);
CREATE INDEX IF NOT EXISTS ix_dataset_assets_content_hash ON dataset_assets (content_hash);
