"""CXR five-label tables (cxr_multilabel / cxr-five-v1)

Revision ID: 4a6b1222f003
Revises: 4a6b1222f002
Create Date: 2026-10-27

Increment only: five new tables for chest X-ray multi-label assistance.
No existing table is altered, dropped or restamped. Old detect data, models,
jobs and files keep their meaning; `dataset_resources.task_type` stays
`detect` by default.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "4a6b1222f003"
down_revision = "4a6b1222f002"
branch_labels = None
depends_on = None

BIGINT_PK = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
BIGINT_FK = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
UUID = sa.String(length=36)


def upgrade() -> None:
    op.create_table(
        "cxr_samples",
        sa.Column("id", BIGINT_PK, autoincrement=True, nullable=False),
        sa.Column("asset_id", BIGINT_FK, nullable=False),
        sa.Column("dataset_version_id", UUID, nullable=False),
        sa.Column("patient_key", sa.String(length=64), nullable=False),
        sa.Column("study_key", sa.String(length=64), nullable=False),
        sa.Column("component_key", sa.String(length=64), nullable=False),
        sa.Column("view", sa.String(length=8), nullable=False),
        sa.Column("raw_labels_json", sa.JSON(), nullable=False),
        sa.Column("schema_version", sa.String(length=32), nullable=False),
        sa.Column("source_hash", sa.String(length=64), nullable=False),
        sa.Column("pixel_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cxr_samples")),
        sa.ForeignKeyConstraint(
            ["asset_id"], ["dataset_assets.id"],
            name=op.f("fk_cxr_samples_asset_id_dataset_assets"), ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("asset_id", name="uq_cxr_sample_asset"),
    )
    with op.batch_alter_table("cxr_samples", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_cxr_samples_dataset_version_id"), ["dataset_version_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_samples_patient_key"), ["patient_key"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_samples_study_key"), ["study_key"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_samples_component_key"), ["component_key"], unique=False)
        batch_op.create_index("ix_cxr_sample_version_patient", ["dataset_version_id", "patient_key"], unique=False)
        batch_op.create_index("ix_cxr_sample_version_component", ["dataset_version_id", "component_key"], unique=False)

    op.create_table(
        "cxr_analysis_batches",
        sa.Column("id", UUID, nullable=False),
        sa.Column("owner_id", BIGINT_FK, nullable=False),
        sa.Column("owner_username", sa.String(length=64), nullable=False),
        sa.Column("job_id", UUID, nullable=True),
        sa.Column("model_id", UUID, nullable=False),
        sa.Column("model_sha256", sa.String(length=64), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("completed", sa.Integer(), nullable=False),
        sa.Column("failed", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cxr_analysis_batches")),
        sa.UniqueConstraint("owner_id", "idempotency_key", name="uq_cxr_batch_idempotency"),
    )
    with op.batch_alter_table("cxr_analysis_batches", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_cxr_analysis_batches_owner_id"), ["owner_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_analysis_batches_job_id"), ["job_id"], unique=True)
        batch_op.create_index(batch_op.f("ix_cxr_analysis_batches_model_id"), ["model_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_analysis_batches_status"), ["status"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_analysis_batches_created_at"), ["created_at"], unique=False)

    op.create_table(
        "cxr_analyses",
        sa.Column("id", UUID, nullable=False),
        sa.Column("batch_id", UUID, nullable=False),
        sa.Column("owner_id", BIGINT_FK, nullable=False),
        sa.Column("asset_id", BIGINT_FK, nullable=True),
        sa.Column("item_key", sa.String(length=160), nullable=False),
        sa.Column("upload_ref", sa.Text(), nullable=True),
        sa.Column("image_sha256", sa.String(length=64), nullable=True),
        sa.Column("image_width", sa.Integer(), nullable=True),
        sa.Column("image_height", sa.Integer(), nullable=True),
        sa.Column("model_id", UUID, nullable=False),
        sa.Column("model_sha256", sa.String(length=64), nullable=False),
        sa.Column("task", sa.String(length=32), nullable=False),
        sa.Column("label_schema", sa.String(length=32), nullable=False),
        sa.Column("raw_scores_json", sa.JSON(), nullable=False),
        sa.Column("calibrated_scores_json", sa.JSON(), nullable=True),
        sa.Column("thresholds_json", sa.JSON(), nullable=True),
        sa.Column("calibration_status", sa.String(length=32), nullable=False),
        sa.Column("threshold_source", sa.String(length=64), nullable=False),
        sa.Column("calibration_artifact_id", UUID, nullable=True),
        sa.Column("quality_json", sa.JSON(), nullable=False),
        sa.Column("explanation_artifact_id", UUID, nullable=True),
        sa.Column("explanation_json", sa.JSON(), nullable=False),
        sa.Column("backend", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("timing_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cxr_analyses")),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["cxr_analysis_batches.id"],
            name=op.f("fk_cxr_analyses_batch_id_cxr_analysis_batches"), ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["asset_id"], ["dataset_assets.id"],
            name=op.f("fk_cxr_analyses_asset_id_dataset_assets"), ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("batch_id", "item_key", "attempt", name="uq_cxr_analysis_item_attempt"),
    )
    with op.batch_alter_table("cxr_analyses", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_cxr_analyses_batch_id"), ["batch_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_analyses_owner_id"), ["owner_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_analyses_asset_id"), ["asset_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_analyses_model_id"), ["model_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_analyses_status"), ["status"], unique=False)

    op.create_table(
        "cxr_reviews",
        sa.Column("id", UUID, nullable=False),
        sa.Column("analysis_id", UUID, nullable=False),
        sa.Column("owner_id", BIGINT_FK, nullable=False),
        sa.Column("reviewer_id", BIGINT_FK, nullable=True),
        sa.Column("reviewer_username", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("items_json", sa.JSON(), nullable=False),
        sa.Column("model_snapshot_json", sa.JSON(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("submitted_at", sa.DateTime(), nullable=True),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("decided_by_id", BIGINT_FK, nullable=True),
        sa.Column("decided_by_username", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cxr_reviews")),
        sa.ForeignKeyConstraint(
            ["analysis_id"], ["cxr_analyses.id"],
            name=op.f("fk_cxr_reviews_analysis_id_cxr_analyses"), ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("analysis_id", "owner_id", name="uq_cxr_review_analysis_owner"),
    )
    with op.batch_alter_table("cxr_reviews", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_cxr_reviews_analysis_id"), ["analysis_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_reviews_owner_id"), ["owner_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_reviews_reviewer_id"), ["reviewer_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_reviews_state"), ["state"], unique=False)

    op.create_table(
        "cxr_reports",
        sa.Column("id", UUID, nullable=False),
        sa.Column("analysis_id", UUID, nullable=False),
        sa.Column("review_id", UUID, nullable=False),
        sa.Column("review_revision", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("supersedes_id", UUID, nullable=True),
        sa.Column("snapshot_json", sa.JSON(), nullable=False),
        sa.Column("snapshot_sha256", sa.String(length=64), nullable=False),
        sa.Column("artifact_id", UUID, nullable=True),
        sa.Column("owner_id", BIGINT_FK, nullable=False),
        sa.Column("issued_by_id", BIGINT_FK, nullable=True),
        sa.Column("issued_by_username", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("issued_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cxr_reports")),
        sa.ForeignKeyConstraint(
            ["analysis_id"], ["cxr_analyses.id"],
            name=op.f("fk_cxr_reports_analysis_id_cxr_analyses"), ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["review_id"], ["cxr_reviews.id"],
            name=op.f("fk_cxr_reports_review_id_cxr_reviews"), ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("analysis_id", "version", name="uq_cxr_report_version"),
    )
    with op.batch_alter_table("cxr_reports", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_cxr_reports_analysis_id"), ["analysis_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_reports_review_id"), ["review_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_reports_state"), ["state"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_reports_supersedes_id"), ["supersedes_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_reports_owner_id"), ["owner_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_cxr_reports_created_at"), ["created_at"], unique=False)


def downgrade() -> None:
    # Only the five CXR tables are removed. Old detect tables and their rows are
    # never touched; dropping a CXR table requires an explicit, reviewed backup.
    op.drop_table("cxr_reports")
    op.drop_table("cxr_reviews")
    op.drop_table("cxr_analyses")
    op.drop_table("cxr_analysis_batches")
    op.drop_table("cxr_samples")
