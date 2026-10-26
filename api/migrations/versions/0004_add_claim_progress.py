"""Track live processing stage and progress for claims."""

from alembic import op
import sqlalchemy as sa

revision = "0004_add_claim_progress"
down_revision = "0003_add_cost_estimates"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("claims", sa.Column("stage", sa.String(length=200), nullable=True))
    op.add_column("claims", sa.Column("progress_pct", sa.Integer(), nullable=True))


def downgrade():
    op.drop_column("claims", "progress_pct")
    op.drop_column("claims", "stage")
