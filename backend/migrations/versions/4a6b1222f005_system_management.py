"""Add organization, custom roles and audit records."""
from alembic import op
import sqlalchemy as sa

revision = "4a6b1222f005"
down_revision = "4a6b1222f004"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index("uq_users_username_lower", "users", [sa.text("lower(username)")], unique=True)
    op.add_column("job_tasks", sa.Column("archived_at", sa.DateTime(), nullable=True))
    op.create_table("system_departments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("parent_id", sa.String(36), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False))
    op.create_table("system_roles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(64), nullable=False, unique=True),
        sa.Column("base_role", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("modules", sa.JSON(), nullable=False))
    op.create_table("system_user_profiles",
        sa.Column("user_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), sa.ForeignKey("users.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("department_id", sa.String(36), sa.ForeignKey("system_departments.id", ondelete="RESTRICT")),
        sa.Column("role_id", sa.String(36), sa.ForeignKey("system_roles.id", ondelete="RESTRICT")))
    op.create_table("system_audit_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("action", sa.String(160), nullable=False),
        sa.Column("status", sa.Integer(), nullable=False),
        sa.Column("client_ip", sa.String(64)),
        sa.Column("request_id", sa.String(64)),
        sa.Column("created_at", sa.DateTime(), nullable=False))
    op.create_table("system_menus",
        sa.Column("path", sa.String(128), primary_key=True),
        sa.Column("label", sa.String(64), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("visible", sa.Boolean(), nullable=False))
    op.create_index("ix_system_audit_logs_kind", "system_audit_logs", ["kind"])
    op.create_index("ix_system_audit_logs_created_at", "system_audit_logs", ["created_at"])


def downgrade():
    op.drop_index("uq_users_username_lower", table_name="users")
    op.drop_column("job_tasks", "archived_at")
    op.drop_table("system_menus")
    op.drop_table("system_audit_logs")
    op.drop_table("system_user_profiles")
    op.drop_table("system_roles")
    op.drop_table("system_departments")
