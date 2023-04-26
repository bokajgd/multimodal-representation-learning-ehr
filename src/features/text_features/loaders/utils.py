"""A set of misc. utility functions for free-text notes loaders.
"""

from pathlib import Path

from google.cloud import bigquery

RELATIVE_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = RELATIVE_PROJECT_ROOT / "data"

def load_sql_query(query: str) -> str:
    client = bigquery.Client()

    return client.query(query).to_dataframe()
