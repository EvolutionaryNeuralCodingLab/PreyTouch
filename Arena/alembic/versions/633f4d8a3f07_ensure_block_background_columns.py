"""ensure block background columns exist

Revision ID: 633f4d8a3f07
Revises: bed84ac01e0c
Create Date: 2026-02-15

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '633f4d8a3f07'
down_revision = 'bed84ac01e0c'
branch_labels = None
depends_on = None


def _get_block_columns():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {col['name'] for col in inspector.get_columns('blocks')}


def upgrade() -> None:
    cols = _get_block_columns()
    if 'background_color' not in cols:
        op.add_column('blocks', sa.Column('background_color', sa.String(), nullable=True))
    if 'bug_mapped_background' not in cols:
        bind = op.get_bind()
        if bind.dialect.name == 'postgresql':
            col = sa.Column('bug_mapped_background', postgresql.JSON(astext_type=sa.Text()), nullable=True)
        else:
            col = sa.Column('bug_mapped_background', sa.JSON(), nullable=True)
        op.add_column('blocks', col)


def downgrade() -> None:
    cols = _get_block_columns()
    if 'bug_mapped_background' in cols:
        op.drop_column('blocks', 'bug_mapped_background')
    if 'background_color' in cols:
        op.drop_column('blocks', 'background_color')
