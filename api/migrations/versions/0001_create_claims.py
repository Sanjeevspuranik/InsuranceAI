"""Create claims and database-backed job table."""

from alembic import op
import sqlalchemy as sa

revision = "0001_create_claims"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "claims",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("owner_subject", sa.String(length=256), nullable=False),
        sa.Column("source_image_key", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("detections", sa.JSON(), nullable=True),
        sa.Column("report_text", sa.Text(), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("pdf_key", sa.String(length=512), nullable=True),
        sa.Column("annotated_image_key", sa.String(length=512), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_claims_tenant_created", "claims", ["tenant_id", "created_at"]
    )
    op.create_index(
        "ix_claims_tenant_status_created",
        "claims",
        ["tenant_id", "status", "created_at"],
    )


def downgrade():
    op.drop_index("ix_claims_tenant_status_created", table_name="claims")
    op.drop_index("ix_claims_tenant_created", table_name="claims")
    op.drop_table("claims")
