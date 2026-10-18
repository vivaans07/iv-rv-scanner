"""DuckDB persistence.

Every table is written with "delete the matching keys, then insert", so
re-running any step for the same day replaces that day's rows instead of
duplicating them. Tables are created from the first DataFrame written to
them; later writes are aligned to the existing column order.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


class Store:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.path)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.con.execute("BEGIN TRANSACTION")
        try:
            yield
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        self.con.execute("COMMIT")

    def has_table(self, table: str) -> bool:
        row = self.con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()
        return bool(row and row[0])

    def columns(self, table: str) -> list[str]:
        rows = self.con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ? "
            "ORDER BY ordinal_position",
            [table],
        ).fetchall()
        return [r[0] for r in rows]

    def replace(self, table: str, df: pd.DataFrame, keys: Mapping[str, Any]) -> int:
        """Delete rows matching `keys` (column -> value or list of values), insert `df`."""
        with self.transaction():
            if self.has_table(table) and keys:
                clauses, params = [], []
                for col, val in keys.items():
                    if isinstance(val, list | tuple | set):
                        vals = list(val)
                        if not vals:
                            continue
                        clauses.append(f'"{col}" IN ({",".join("?" * len(vals))})')
                        params.extend(vals)
                    else:
                        clauses.append(f'"{col}" = ?')
                        params.append(val)
                if clauses:
                    self.con.execute(f"DELETE FROM {table} WHERE {' AND '.join(clauses)}", params)
            if df.empty:
                return 0
            self._insert(table, df)
        return len(df)

    def append(self, table: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        with self.transaction():
            self._insert(table, df)
        return len(df)

    def _insert(self, table: str, df: pd.DataFrame) -> None:
        frame = df.copy()
        # DuckDB infers object columns poorly; give it plain types.
        for col in frame.columns:
            if frame[col].dtype == object and frame[col].map(lambda v: isinstance(v, bool)).all():
                frame[col] = frame[col].astype(bool)
        self.con.register("_incoming", frame)
        try:
            if not self.has_table(table):
                self.con.execute(f"CREATE TABLE {table} AS SELECT * FROM _incoming")
                return
            existing = self.columns(table)
            for col in frame.columns:
                if col not in existing:
                    self.con.execute(f'ALTER TABLE {table} ADD COLUMN "{col}" VARCHAR')
                    existing.append(col)
            select = ", ".join(f'"{c}"' if c in frame.columns else "NULL" for c in existing)
            self.con.execute(f"INSERT INTO {table} SELECT {select} FROM _incoming")
        finally:
            self.con.unregister("_incoming")

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    def read(self, table: str, where: str = "", params: list[Any] | None = None) -> pd.DataFrame:
        if not self.has_table(table):
            return pd.DataFrame()
        # ASOF is a reserved word in DuckDB; quote bare uses of the column.
        where = re.sub(r'(?<!")\basof\b(?!")', '"asof"', where)
        sql = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "")
        return self.query(sql, params)
