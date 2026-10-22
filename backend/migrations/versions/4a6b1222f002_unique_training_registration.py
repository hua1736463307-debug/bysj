"""One model registration per training job; retain null base registrations."""
from alembic import op

revision = "4a6b1222f002"
down_revision = "4a6b1222f001"
branch_labels = None
depends_on = None


def upgrade():
    # Existing duplicate job registrations require explicit reconciliation;
    # a unique-index failure must not silently delete historical models.
    op.create_index("uq_model_training_job", "model_registry", ["training_job_id"], unique=True)


def downgrade():
    op.drop_index("uq_model_training_job", table_name="model_registry")
