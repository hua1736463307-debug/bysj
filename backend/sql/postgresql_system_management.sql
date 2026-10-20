-- deep-vision 平台管理 PostgreSQL 幂等迁移
-- 可重复执行；应用启动时 database.py 也会自动执行同等兼容迁移与种子初始化。

BEGIN;

ALTER TABLE users ADD COLUMN IF NOT EXISTS dept_id BIGINT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS user_no TEXT;
ALTER TABLE users ALTER COLUMN user_no TYPE TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS nickname VARCHAR(64);
ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(32);
ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(128);
ALTER TABLE users ADD COLUMN IF NOT EXISTS remark VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_ip VARCHAR(64);
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP;
ALTER TABLE users ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_until TIMESTAMP;

ALTER TABLE roles ADD COLUMN IF NOT EXISTS role_code VARCHAR(64);
ALTER TABLE roles ADD COLUMN IF NOT EXISTS authority_level INTEGER NOT NULL DEFAULT 100;
UPDATE roles SET role_code = CAST(id AS TEXT) WHERE role_code IS NULL OR BTRIM(role_code) = '';
ALTER TABLE roles ALTER COLUMN role_code SET NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_roles_role_code ON roles(role_code);

UPDATE users SET user_no = CAST(id AS TEXT) WHERE user_no IS NULL OR BTRIM(user_no) = '';
CREATE UNIQUE INDEX IF NOT EXISTS uq_users_user_no ON users(user_no);

ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS request_method VARCHAR(16);
ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS request_url VARCHAR(512);
ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS request_params TEXT;
ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS response_data TEXT;
ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS session_id VARCHAR(64);
ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS request_id VARCHAR(128);
ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS reason VARCHAR(64);
ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS user_agent TEXT;
ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS event_key VARCHAR(192);
CREATE INDEX IF NOT EXISTS ix_operation_logs_session_id ON operation_logs(session_id);
CREATE INDEX IF NOT EXISTS ix_operation_logs_request_id ON operation_logs(request_id);
CREATE INDEX IF NOT EXISTS ix_operation_logs_reason ON operation_logs(reason);
CREATE UNIQUE INDEX IF NOT EXISTS uq_operation_logs_event_key ON operation_logs(event_key) WHERE event_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS departments (
    id BIGSERIAL PRIMARY KEY,
    parent_id BIGINT,
    name VARCHAR(100) NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    leader VARCHAR(64),
    phone VARCHAR(32),
    email VARCHAR(128),
    status VARCHAR(16) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS roles (
    id BIGSERIAL PRIMARY KEY,
    role_code VARCHAR(64) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    role_key VARCHAR(64) NOT NULL UNIQUE,
    sort_order INTEGER NOT NULL DEFAULT 0,
    authority_level INTEGER NOT NULL DEFAULT 100,
    data_scope VARCHAR(16) NOT NULL DEFAULT 'self',
    status VARCHAR(16) NOT NULL DEFAULT 'active',
    remark VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS menus (
    id BIGSERIAL PRIMARY KEY,
    parent_id BIGINT,
    name VARCHAR(100) NOT NULL,
    path VARCHAR(255),
    component VARCHAR(255),
    permission VARCHAR(128),
    icon VARCHAR(64),
    menu_type VARCHAR(16) NOT NULL DEFAULT 'menu',
    visible VARCHAR(16) NOT NULL DEFAULT 'visible',
    status VARCHAR(16) NOT NULL DEFAULT 'active',
    cache VARCHAR(16) NOT NULL DEFAULT 'cache',
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Persistent marker: default menus are bootstrapped once, then administrator changes are authoritative.
CREATE TABLE IF NOT EXISTS system_seed_state (
    key VARCHAR(64) PRIMARY KEY,
    initialized_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_roles (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    role_id BIGINT NOT NULL,
    CONSTRAINT uq_user_role UNIQUE (user_id, role_id)
);

CREATE TABLE IF NOT EXISTS role_menus (
    id BIGSERIAL PRIMARY KEY,
    role_id BIGINT NOT NULL,
    menu_id BIGINT NOT NULL,
    CONSTRAINT uq_role_menu UNIQUE (role_id, menu_id)
);

CREATE TABLE IF NOT EXISTS role_departments (
    id BIGSERIAL PRIMARY KEY,
    role_id BIGINT NOT NULL,
    department_id BIGINT NOT NULL,
    CONSTRAINT uq_role_department UNIQUE (role_id, department_id)
);

CREATE TABLE IF NOT EXISTS dataset_resources (
    id VARCHAR(36) PRIMARY KEY,
    dataset_id VARCHAR(255) NOT NULL,
    dataset_role VARCHAR(32) NOT NULL,
    owner_user_id BIGINT,
    owner_username VARCHAR(64) NOT NULL DEFAULT 'system',
    owner_role_id BIGINT,
    visibility_mode VARCHAR(32) NOT NULL DEFAULT 'role_and_higher',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_dataset_resource_identity UNIQUE (dataset_id, dataset_role)
);

CREATE TABLE IF NOT EXISTS online_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL,
    username VARCHAR(64) NOT NULL,
    ip_address VARCHAR(64),
    location VARCHAR(128) NOT NULL DEFAULT '内网IP',
    browser VARCHAR(128),
    os VARCHAR(128),
    status VARCHAR(16) NOT NULL DEFAULT 'active',
    login_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    revoked_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_departments_parent_id ON departments(parent_id);
CREATE INDEX IF NOT EXISTS ix_departments_status ON departments(status);
CREATE INDEX IF NOT EXISTS ix_users_dept_id ON users(dept_id);
CREATE INDEX IF NOT EXISTS ix_users_phone ON users(phone);
CREATE INDEX IF NOT EXISTS ix_roles_status ON roles(status);
CREATE INDEX IF NOT EXISTS ix_roles_authority_level ON roles(authority_level);
CREATE INDEX IF NOT EXISTS ix_menus_parent_id ON menus(parent_id);
CREATE INDEX IF NOT EXISTS ix_online_sessions_status ON online_sessions(status);
CREATE INDEX IF NOT EXISTS ix_online_sessions_username ON online_sessions(username);
CREATE INDEX IF NOT EXISTS ix_dataset_resources_dataset_id ON dataset_resources(dataset_id);
CREATE INDEX IF NOT EXISTS ix_dataset_resources_dataset_role ON dataset_resources(dataset_role);
CREATE INDEX IF NOT EXISTS ix_dataset_resources_owner_user_id ON dataset_resources(owner_user_id);
CREATE INDEX IF NOT EXISTS ix_dataset_resources_owner_role_id ON dataset_resources(owner_role_id);

INSERT INTO departments (parent_id, name, sort_order, status)
SELECT NULL, '总公司', 0, 'active'
WHERE NOT EXISTS (SELECT 1 FROM departments WHERE parent_id IS NULL);

INSERT INTO roles (name, role_code, role_key, sort_order, authority_level, data_scope, status)
VALUES ('超级管理员', '1', 'admin', 1, 1000, 'all', 'active')
ON CONFLICT (role_key) DO NOTHING;

INSERT INTO roles (name, role_code, role_key, sort_order, authority_level, data_scope, status)
VALUES ('普通用户', '2', 'user', 2, 100, 'self', 'active')
ON CONFLICT (role_key) DO NOTHING;

UPDATE roles SET authority_level = 1000 WHERE role_key = 'admin';
UPDATE roles SET authority_level = 100 WHERE role_key = 'user';

UPDATE users
SET dept_id = (SELECT id FROM departments WHERE parent_id IS NULL ORDER BY id LIMIT 1)
WHERE dept_id IS NULL;

INSERT INTO user_roles (user_id, role_id)
SELECT u.id, r.id
FROM users u
JOIN roles r ON r.role_key = CASE WHEN u.role = 'admin' THEN 'admin' ELSE 'user' END
ON CONFLICT (user_id, role_id) DO NOTHING;

COMMIT;
