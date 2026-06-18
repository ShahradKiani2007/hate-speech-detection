import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from database_connection import get_engine  # noqa: E402

HERE = Path(__file__).resolve().parent
SQL_FILE = HERE / "queries.sql"
OUT = HERE / "query_outputs.txt"


def split_queries(sql):
    blocks = []
    title = None
    buf = []
    for line in sql.splitlines():
        m = re.match(r"--\s*(Q\d+:.*)", line)
        if m:
            if buf:
                blocks.append((title, "\n".join(buf).strip()))
                buf = []
            title = m.group(1).strip()
        elif line.strip():
            buf.append(line)
    if buf:
        blocks.append((title, "\n".join(buf).strip()))
    return blocks


def main():
    engine = get_engine()
    sql = SQL_FILE.read_text()
    lines = []
    for title, q in split_queries(sql):
        q = q.rstrip(";")
        df = pd.read_sql(q, engine)
        lines.append("=" * 70)
        lines.append(title or "query")
        lines.append("-" * 70)
        lines.append(df.to_string(index=False))
        lines.append("")
    text = "\n".join(lines)
    OUT.write_text(text)
    print(text)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
