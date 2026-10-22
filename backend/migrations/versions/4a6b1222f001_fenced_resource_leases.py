"""Persistent lease generations and full worker identity lengths."""
from alembic import op
import sqlalchemy as sa

revision = "4a6b1222f001"
down_revision = "304f405af95f"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("resource_leases") as batch:
        batch.alter_column("resource", existing_type=sa.String(32), type_=sa.String(160))
        batch.alter_column("holder", existing_type=sa.String(64), type_=sa.String(160))
        batch.add_column(sa.Column("generation", sa.BigInteger(), nullable=False, server_default="1"))


def downgrade():
    # Keep expanded identity widths; narrowing could truncate live identities.
    with op.batch_alter_table("resource_leases") as batch:
        batch.drop_column("generation")
