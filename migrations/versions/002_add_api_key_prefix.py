"""Add api_keys.key_prefix for indexed authentication lookup.

Revision ID: 002
Revises: 001
Create Date: 2026-09-19
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("key_prefix", sa.String(12), nullable=True))
    op.create_index("ix_api_keys_key_prefix", "api_keys", ["key_prefix"])
    # Existing rows predate the prefix: they cannot be located by prefix and
    # will fail authentication. Rotate them (POST /v1/keys) after upgrading.
    op.execute(
        "COMMENT ON COLUMN api_keys.key_prefix IS "
        "'Leading characters of the raw key; NULL rows must be rotated (migration 002)'"
    )


def downgrade() -> None:
    op.drop_index("ix_api_keys_key_prefix")
    op.drop_column("api_keys", "key_prefix")