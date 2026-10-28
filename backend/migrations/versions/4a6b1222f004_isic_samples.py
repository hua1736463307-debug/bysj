"""ISIC dermoscopy samples (single-label, lesion-grouped)

Revision ID: 4a6b1222f004
Revises: 4a6b1222f003
Create Date: 2026-10-27

Increment only: one new table for the dermoscopy direction. CXR and detect
tables are untouched; `cxr_samples` keeps its four-state semantics and is not
reused for single-label data.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "4a6b1222f004"
down_revision = "4a6b1222f003"
branch_labels = None
depends_on = None

BIGINT_PK = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
BIGINT_FK = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
UUID = sa.String(length=36)


def upgrade() -> None:
    op.create_table(
        "isic_samples",
        sa.Column("id", BIGINT_PK, autoincrement=True, nullable=False),
        sa.Column("asset_id", BIGINT_FK, nullable=False),
        sa.Column("dataset_version_id", UUID, nullable=False),
        sa.Column("lesion_key", sa.String(length=64), nullable=True),
        sa.Column("group_key", sa.String(length=96), nullable=False),
        sa.Column("class_key", sa.String(length=32), nullable=False),
        sa.Column("class_index", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(length=32), nullable=False),
        sa.Column("pixel_hash", sa.String(length=64), nullable=False),
        sa.Column("source_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_isic_samples")),
        sa.ForeignKeyConstraint(
            ["asset_id"], ["dataset_assets.id"],
            name=op.f("fk_isic_samples_asset_id_dataset_assets"), ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("asset_id", name="uq_isic_sample_asset"),
    )
    with op.batch_alter_table("isic_samples", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_isic_samples_dataset_version_id"), ["dataset_version_id"], unique=False)
        batch_op.create_index("ix_isic_sample_version_group", ["dataset_version_id", "group_key"], unique=False)
        batch_op.create_index("ix_isic_sample_version_class", ["dataset_version_id", "class_index"], unique=False)


def downgrade() -> None:
    op.drop_table("isic_samples")
