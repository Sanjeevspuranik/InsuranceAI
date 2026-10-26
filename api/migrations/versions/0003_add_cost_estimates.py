"""Add repair-cost estimate, internal-damage estimate, policy input and claim breakdown."""

from alembic import op
import sqlalchemy as sa

revision = "0003_add_cost_estimates"
down_revision = "0002_add_reasoning_tokens"
branch_labels = None
depends_on = None

COLUMNS = ("cost_estimate", "internal_damage_estimate", "policy_input", "claim_breakdown")


def upgrade():
    for name in COLUMNS:
        op.add_column("claims", sa.Column(name, sa.JSON(), nullable=True))


def downgrade():
    for name in reversed(COLUMNS):
        op.drop_column("claims", name)
