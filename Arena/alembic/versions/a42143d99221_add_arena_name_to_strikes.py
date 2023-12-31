"""add arena name to strikes

Revision ID: a42143d99221
Revises: f3dd993d5af6
Create Date: 2023-03-29 11:27:59.518912

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a42143d99221'
down_revision = 'f3dd993d5af6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('strikes', sa.Column('arena', sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('strikes', 'arena')
    # ### end Alembic commands ###
