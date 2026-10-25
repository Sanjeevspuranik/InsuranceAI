"""Store provider-reported reasoning token counts."""

from alembic import op
import sqlalchemy as sa

revision = "0002_add_reasoning_tokens"
down_revision = "0001_create_claims"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("claims", sa.Column("reasoning_tokens", sa.Integer(), nullable=True))


def downgrade():
    op.drop_column("claims", "reasoning_tokens")
