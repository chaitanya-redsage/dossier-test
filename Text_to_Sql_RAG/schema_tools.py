from __future__ import annotations

from typing import Dict, Any, List, Optional

from db_connection import get_connection

SYSTEM_SCHEMAS = {"pg_catalog", "information_schema"}


def extract_schema_for_schemas(
    schemas: Optional[List[str]] = None,
    db_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract tables, columns, and foreign keys for the requested schemas.
    If schemas=None, discovers all non-system schemas.
    """
    with get_connection(db_url=db_url) as conn:
        with conn.cursor() as cur:
            if schemas is None:
                cur.execute(
                    """
                    SELECT schema_name
                    FROM information_schema.schemata
                    WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY schema_name;
                    """
                )
                schemas = [r[0] for r in cur.fetchall()]
            else:
                schemas = [s for s in schemas if s and s not in SYSTEM_SCHEMAS]

            out: Dict[str, Any] = {"schemas": {}}

            for sch in schemas:
                # Tables
                cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = %s
                      AND table_type = 'BASE TABLE'
                    ORDER BY table_name;
                    """,
                    (sch,),
                )
                table_names = [r[0] for r in cur.fetchall()]
                tables: Dict[str, Any] = {
                    t: {"columns": [], "foreign_keys": []} for t in table_names
                }

                # Columns
                cur.execute(
                    """
                    SELECT table_name, column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = %s
                    ORDER BY table_name, ordinal_position;
                    """,
                    (sch,),
                )
                for table_name, column_name, data_type, is_nullable in cur.fetchall():
                    if table_name in tables:
                        tables[table_name]["columns"].append(
                            {
                                "name": column_name,
                                "type": data_type,
                                "nullable": (is_nullable == "YES"),
                            }
                        )

                # Foreign keys
                cur.execute(
                    """
                    SELECT
                      tc.table_name,
                      kcu.column_name,
                      ccu.table_schema AS foreign_table_schema,
                      ccu.table_name   AS foreign_table_name,
                      ccu.column_name  AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                     AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                      AND tc.table_schema = %s;
                    """,
                    (sch,),
                )
                for table_name, column_name, f_sch, f_table, f_col in cur.fetchall():
                    if table_name in tables:
                        tables[table_name]["foreign_keys"].append(
                            {
                                "column": column_name,
                                "ref_schema": f_sch,
                                "ref_table": f_table,
                                "ref_column": f_col,
                            }
                        )

                out["schemas"][sch] = {"tables": tables}

    return out


def get_schema_payload(
    schemas: Optional[List[str]] = None,
    db_url: Optional[str] = None,
) -> Dict[str, Any]:
    return extract_schema_for_schemas(schemas=schemas, db_url=db_url)
