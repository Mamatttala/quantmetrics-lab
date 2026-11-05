"""Read-only inventory of local dataset classifications."""
import json
import sqlite3
from pathlib import Path

for path in Path("data").rglob("catalog.db"):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        print(path)
        for name, category, provenance, count in conn.execute(
            "SELECT name, category, provenance, row_count FROM datasets ORDER BY created_at"
        ):
            print(json.dumps([name, category, json.loads(provenance).get("operation"), count]))
