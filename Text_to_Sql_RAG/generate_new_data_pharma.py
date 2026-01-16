from __future__ import annotations

import os
import random
from datetime import date, datetime, timedelta, timezone

import psycopg2
from psycopg2.extras import execute_values, Json
from faker import Faker

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# -----------------------
# Settings / Scale knobs
# -----------------------
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

N_SITES = int(os.getenv("N_SITES", "2"))
DEPTS_PER_SITE = int(os.getenv("DEPTS_PER_SITE", "6"))
N_EMPLOYEES = int(os.getenv("N_EMPLOYEES", "60"))
N_USERS = int(os.getenv("N_USERS", "40"))

N_SUPPLIERS = int(os.getenv("N_SUPPLIERS", "25"))
N_MATERIALS = int(os.getenv("N_MATERIALS", "60"))
N_POS = int(os.getenv("N_POS", "40"))
N_RECEIPTS = int(os.getenv("N_RECEIPTS", "50"))
N_MATERIAL_LOTS = int(os.getenv("N_MATERIAL_LOTS", "120"))
N_LOCATIONS_PER_SITE = int(os.getenv("N_LOCATIONS_PER_SITE", "4"))
N_INV_TXNS = int(os.getenv("N_INV_TXNS", "800"))

N_PRODUCTS = int(os.getenv("N_PRODUCTS", "18"))
N_FORMULATIONS_PER_PRODUCT = int(os.getenv("N_FORMULATIONS_PER_PRODUCT", "2"))
N_BATCHES = int(os.getenv("N_BATCHES", "80"))
N_BATCH_STEPS = int(os.getenv("N_BATCH_STEPS", "8"))
N_CONSUMPTIONS = int(os.getenv("N_CONSUMPTIONS", "250"))

N_METHODS = int(os.getenv("N_METHODS", "12"))
N_TESTS = int(os.getenv("N_TESTS", "25"))
N_SPECS = int(os.getenv("N_SPECS", "120"))
N_SAMPLES = int(os.getenv("N_SAMPLES", "240"))
N_RESULTS = int(os.getenv("N_RESULTS", "800"))

N_STABILITY_STUDIES = int(os.getenv("N_STABILITY_STUDIES", "25"))

N_DEVIATIONS = int(os.getenv("N_DEVIATIONS", "35"))
N_CAPAS = int(os.getenv("N_CAPAS", "25"))
N_CHANGES = int(os.getenv("N_CHANGES", "30"))

N_DOCUMENTS = int(os.getenv("N_DOCUMENTS", "80"))
N_COURSES = int(os.getenv("N_COURSES", "25"))
N_TRAINING_RECORDS = int(os.getenv("N_TRAINING_RECORDS", "140"))

N_AUDITS = int(os.getenv("N_AUDITS", "20"))
N_FINDINGS = int(os.getenv("N_FINDINGS", "40"))

N_SUBMISSIONS = int(os.getenv("N_SUBMISSIONS", "25"))
N_CERTIFICATES = int(os.getenv("N_CERTIFICATES", "35"))

N_AUDIT_LOG = int(os.getenv("N_AUDIT_LOG", "800"))

# -----------------------
# Helpers
# -----------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def rand_past(days: int = 365) -> datetime:
    return utcnow() - timedelta(days=random.randint(0, days), minutes=random.randint(0, 1440))


def rand_future_date(days: int = 365) -> date:
    return (date.today() + timedelta(days=random.randint(1, days)))


def load_env(path: str = ".env") -> None:
    if load_dotenv:
        load_dotenv(path)
        return
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def connect():
    load_env()
    dbname = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    if not dbname or not user or not password:
        raise RuntimeError("Need DB_NAME, DB_USER, DB_PASSWORD (and optionally DB_HOST, DB_PORT) in .env")
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=dbname,
        user=user,
        password=password,
        options="-c search_path=new_data,public",
    )


def bulk_insert(cur, table: str, cols: list[str], rows: list[tuple]):
    if not rows:
        return
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s"
    execute_values(cur, sql, rows, page_size=2000)


def fetch_ids(cur, table: str, id_col: str) -> list[int]:
    cur.execute(f"SELECT {id_col} FROM {table}")
    return [r[0] for r in cur.fetchall()]


def safe_unique(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:05d}" for i in range(1, n + 1)]


def maybe(p=0.5):
    return random.random() < p


# -----------------------
# Generator
# -----------------------
def main():
    with connect() as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            # -----------------------
            # Company / Sites / Depts
            # -----------------------
            cur.execute(
                "INSERT INTO new_data.company (name, country) VALUES (%s,%s) ON CONFLICT (name) DO UPDATE SET country=EXCLUDED.country RETURNING company_id",
                ("Dossier Pharma Pty Ltd", "Australia"),
            )
            company_id = cur.fetchone()[0]

            sites = []
            for i in range(N_SITES):
                sites.append(
                    (
                        company_id,
                        f"{fake.city()} Site {i+1}",
                        random.choice(["manufacturing", "r&d", "warehouse"]),
                        fake.street_address(),
                        fake.city(),
                        fake.state(),
                        "Australia",
                        "Australia/Sydney",
                    )
                )
            bulk_insert(
                cur,
                "new_data.site",
                ["company_id", "name", "site_type", "address_line1", "city", "state", "country", "timezone"],
                sites,
            )
            site_ids = fetch_ids(cur, "new_data.site", "site_id")

            base_depts = ["QA", "QC", "Manufacturing", "Regulatory Affairs", "Supply Chain", "Engineering", "Validation"]
            dept_rows = []
            for sid in site_ids:
                picks = random.sample(base_depts, k=min(DEPTS_PER_SITE, len(base_depts)))
                for name in picks:
                    dept_rows.append((sid, name))
            bulk_insert(cur, "new_data.department", ["site_id", "name"], dept_rows)
            dept_ids = fetch_ids(cur, "new_data.department", "department_id")

            # -----------------------
            # People / Employees
            # -----------------------
            person_rows = []
            for _ in range(N_EMPLOYEES):
                person_rows.append((fake.first_name(), fake.last_name(), fake.unique.email(), fake.phone_number()))
            bulk_insert(cur, "new_data.person", ["first_name", "last_name", "email", "phone"], person_rows)
            person_ids = fetch_ids(cur, "new_data.person", "person_id")

            emp_rows = []
            for pid in person_ids[:N_EMPLOYEES]:
                emp_rows.append(
                    (
                        pid,
                        random.choice(site_ids),
                        random.choice(dept_ids),
                        random.choice(["Analyst", "Scientist", "Operator", "QA Associate", "Manager", "Engineer"]),
                        fake.date_between(start_date="-8y", end_date="today"),
                        True,
                    )
                )
            bulk_insert(
                cur,
                "new_data.employee",
                ["person_id", "site_id", "department_id", "title", "hire_date", "is_active"],
                emp_rows,
            )
            employee_ids = fetch_ids(cur, "new_data.employee", "employee_id")

            # -----------------------
            # Users / Roles / Permissions
            # -----------------------
            roles = ["admin", "qa", "qc", "manufacturing", "ra", "supply_chain", "viewer"]
            perms = [
                "product.read", "product.write",
                "batch.read", "batch.write",
                "qc.read", "qc.write",
                "docs.read", "docs.write",
                "reg.read", "reg.write",
                "inventory.read", "inventory.write",
            ]
            for r in roles:
                cur.execute("INSERT INTO new_data.role(name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (r,))
            for p in perms:
                cur.execute("INSERT INTO new_data.permission(name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (p,))
            role_ids = fetch_ids(cur, "new_data.role", "role_id")
            perm_ids = fetch_ids(cur, "new_data.permission", "permission_id")

            # map some permissions to each role
            for rid in role_ids:
                for pid in random.sample(perm_ids, k=random.randint(3, min(7, len(perm_ids)))):
                    cur.execute(
                        "INSERT INTO new_data.role_permission(role_id, permission_id) VALUES (%s,%s) ON CONFLICT (role_id, permission_id) DO NOTHING",
                        (rid, pid),
                    )

            user_rows = []
            usernames = safe_unique("user", N_USERS)

            # pick unique employees for users (because app_user.employee_id is UNIQUE)
            n_with_employee = min(int(N_USERS * 0.85), len(employee_ids), N_USERS)
            unique_emp_for_users = random.sample(employee_ids, k=n_with_employee)

            for i in range(N_USERS):
                emp_id = unique_emp_for_users[i] if i < n_with_employee else None
                user_rows.append((emp_id, usernames[i], fake.sha256(), rand_past(900), True))

            bulk_insert(
                cur,
                "new_data.app_user",
                ["employee_id", "username", "password_hash", "created_at", "is_active"],
                user_rows,
            )
            user_ids = fetch_ids(cur, "new_data.app_user", "app_user_id")

            # -----------------------
            # Documents + versions
            # -----------------------
            doc_types = ["SOP", "SPEC", "BMR", "Protocol", "Report"]
            doc_rows = []
            doc_codes = safe_unique("DOC-", N_DOCUMENTS)
            for i in range(N_DOCUMENTS):
                doc_rows.append(
                    (
                        doc_codes[i],
                        fake.sentence(nb_words=6),
                        random.choice(doc_types),
                        random.choice(dept_ids) if maybe(0.8) else None,
                        random.randint(1, 3),
                        random.choice(["draft", "effective", "obsolete"]),
                        rand_past(1200),
                    )
                )
            bulk_insert(
                cur,
                "new_data.document",
                ["doc_code", "title", "doc_type", "owner_department_id", "current_version", "status", "created_at"],
                doc_rows,
            )
            document_ids = fetch_ids(cur, "new_data.document", "document_id")

            dv_rows = []
            for did in document_ids:
                cur.execute("SELECT current_version FROM new_data.document WHERE document_id=%s", (did,))
                current_version = int(cur.fetchone()[0])
                for v in range(1, current_version + 1):
                    dv_rows.append(
                        (
                            did,
                            v,
                            fake.date_between(start_date="-3y", end_date="today") if v == current_version else None,
                            f"s3://fake/docs/{did}/v{v}.pdf",
                            fake.sha256(),
                            rand_past(1200),
                        )
                    )
            bulk_insert(
                cur,
                "new_data.document_version",
                ["document_id", "version_no", "effective_date", "storage_uri", "checksum_sha256", "created_at"],
                dv_rows,
            )

            # -----------------------
            # Suppliers / Materials / Supplier Material
            # -----------------------
            supplier_rows = []
            for i in range(N_SUPPLIERS):
                supplier_rows.append(
                    (
                        f"{fake.company()}",
                        random.choice(["raw_material", "packaging", "lab_service", "logistics"]),
                        random.choice(["Australia", "India", "USA", "Germany", "Singapore"]),
                        fake.company_email(),
                        fake.phone_number(),
                        random.choice(["approved", "approved", "conditional"]),
                    )
                )
            bulk_insert(
                cur,
                "new_data.supplier",
                ["name", "supplier_type", "country", "email", "phone", "status"],
                supplier_rows,
            )
            supplier_ids = fetch_ids(cur, "new_data.supplier", "supplier_id")

            material_types = ["raw", "excipient", "api", "packaging"]
            units = ["kg", "g", "L", "each"]
            mat_codes = safe_unique("MAT-", N_MATERIALS)
            mat_rows = []
            for i in range(N_MATERIALS):
                mat_rows.append(
                    (
                        mat_codes[i],
                        f"{fake.word().title()} {random.choice(['Powder','Granule','Solution','Film'])}",
                        random.choice(material_types),
                        random.choice(units),
                        random.choice(document_ids) if maybe(0.6) else None,
                        True,
                    )
                )
            bulk_insert(
                cur,
                "new_data.material",
                ["material_code", "name", "material_type", "unit", "specification_doc_id", "is_active"],
                mat_rows,
            )
            material_ids = fetch_ids(cur, "new_data.material", "material_id")

            # supplier_material mapping
            sm_rows = []
            for mid in material_ids:
                for sid in random.sample(supplier_ids, k=random.randint(1, 2)):
                    sm_rows.append((sid, mid, f"SKU-{sid}-{mid}", random.randint(7, 60), True))
            # insert with conflict ignore
            for sid, mid, sku, ltd, approved in sm_rows:
                cur.execute(
                    """
                    INSERT INTO new_data.supplier_material(supplier_id, material_id, supplier_sku, lead_time_days, approved)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (supplier_id, material_id) DO NOTHING
                    """,
                    (sid, mid, sku, ltd, approved),
                )

            # -----------------------
            # Inventory locations
            # -----------------------
            loc_rows = []
            for sid in site_ids:
                for j in range(N_LOCATIONS_PER_SITE):
                    loc_rows.append((sid, f"LOC-{sid}-{j+1}", random.choice(["warehouse", "cold_room", "qc_hold"])))
            bulk_insert(cur, "new_data.inventory_location", ["site_id", "name", "location_type"], loc_rows)
            location_ids = fetch_ids(cur, "new_data.inventory_location", "location_id")

            # -----------------------
            # Purchase orders + lines
            # -----------------------
            po_rows = []
            po_numbers = safe_unique("PO-", N_POS)
            for i in range(N_POS):
                po_rows.append((random.choice(supplier_ids), random.choice(site_ids), po_numbers[i], rand_past(900), random.choice(["open", "received", "partial"])))
            bulk_insert(cur, "new_data.purchase_order", ["supplier_id", "site_id", "po_number", "ordered_at", "status"], po_rows)
            po_ids = fetch_ids(cur, "new_data.purchase_order", "po_id")

            pol_rows = []
            for po_id in po_ids:
                for mid in random.sample(material_ids, k=random.randint(1, 4)):
                    pol_rows.append(
                        (po_id, mid, round(random.uniform(10, 500), 3), round(random.uniform(2, 80), 4), rand_future_date(120))
                    )
            # insert with conflict ignore
            for po_id, mid, qty, price, exp in pol_rows:
                cur.execute(
                    """
                    INSERT INTO new_data.purchase_order_line(po_id, material_id, qty_ordered, unit_price, expected_date)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (po_id, material_id) DO NOTHING
                    """,
                    (po_id, mid, qty, price, exp),
                )

            # -----------------------
            # Receipts + material lots + balances + txns
            # -----------------------
            receipt_rows = []
            for _ in range(N_RECEIPTS):
                receipt_rows.append((random.choice(po_ids) if maybe(0.75) else None, random.choice(site_ids), rand_past(800), random.choice(employee_ids), random.choice(["quarantine", "released", "rejected"])))
            bulk_insert(
                cur,
                "new_data.material_receipt",
                ["po_id", "site_id", "received_at", "received_by_employee_id", "status"],
                receipt_rows,
            )
            receipt_ids = fetch_ids(cur, "new_data.material_receipt", "receipt_id")

            lot_rows = []
            for i in range(N_MATERIAL_LOTS):
                mid = random.choice(material_ids)
                lot_rows.append(
                    (
                        mid,
                        random.choice(supplier_ids),
                        random.choice(receipt_ids) if maybe(0.8) else None,
                        f"LOT-{mid}-{i+1:04d}",
                        fake.date_between(start_date="-2y", end_date="today"),
                        fake.date_between(start_date="today", end_date="+2y"),
                        random.choice(["quarantine", "released", "released", "rejected", "consumed"]),
                    )
                )
            bulk_insert(
                cur,
                "new_data.material_lot",
                ["material_id", "supplier_id", "receipt_id", "lot_number", "manufacture_date", "expiry_date", "status"],
                lot_rows,
            )
            material_lot_ids = fetch_ids(cur, "new_data.material_lot", "material_lot_id")

            # balances
            bal_rows = []
            for lot in random.sample(material_lot_ids, k=min(len(material_lot_ids), 200)):
                bal_rows.append((random.choice(location_ids), lot, round(random.uniform(0, 800), 3)))
            for loc, lot, qty in bal_rows:
                cur.execute(
                    """
                    INSERT INTO new_data.inventory_balance(location_id, material_lot_id, qty_on_hand)
                    VALUES (%s,%s,%s)
                    ON CONFLICT (location_id, material_lot_id) DO NOTHING
                    """,
                    (loc, lot, qty),
                )

            # txns
            txn_rows = []
            for _ in range(N_INV_TXNS):
                lot = random.choice(material_lot_ids)
                loc = random.choice(location_ids)
                ttype = random.choice(["receive", "move", "consume", "adjust"])
                qty = round(random.uniform(1, 40), 3) * (1 if ttype == "receive" else -1)
                txn_rows.append((loc, lot, ttype, qty, rand_past(900), random.choice(["batch", "po", "adjustment"]), random.randint(1, 5000)))
            bulk_insert(
                cur,
                "new_data.inventory_txn",
                ["location_id", "material_lot_id", "txn_type", "qty_change", "occurred_at", "ref_entity", "ref_id"],
                txn_rows,
            )

            # -----------------------
            # Products / Formulations
            # -----------------------
            prod_rows = []
            dosage_forms = ["tablet", "capsule", "injection", "syrup"]
            areas = ["cardio", "metabolic", "neuro", "oncology", "respiratory"]
            for i in range(N_PRODUCTS):
                prod_rows.append(
                    (
                        f"PRD-{i+1:04d}",
                        f"{fake.word().title()} {random.choice(['XR','IR','Plus','Max'])}",
                        random.choice(dosage_forms),
                        random.choice(["10 mg", "20 mg", "50 mg", "100 mg", "250 mg"]),
                        random.choice(["oral", "IV", "IM"]),
                        random.choice(areas),
                        random.choice(["development", "clinical", "commercial"]),
                    )
                )
            bulk_insert(
                cur,
                "new_data.drug_product",
                ["product_code", "name", "dosage_form", "strength_label", "route", "therapeutic_area", "status"],
                prod_rows,
            )
            product_ids = fetch_ids(cur, "new_data.drug_product", "product_id")

            # formulations + components
            formulation_ids = []
            for pid in product_ids:
                for v in range(1, N_FORMULATIONS_PER_PRODUCT + 1):
                    cur.execute(
                        """
                        INSERT INTO new_data.formulation(product_id, version_no, effective_from, status)
                        VALUES (%s,%s,%s,%s)
                        RETURNING formulation_id
                        """,
                        (pid, v, fake.date_between(start_date="-2y", end_date="today"), "active" if v == N_FORMULATIONS_PER_PRODUCT else "inactive"),
                    )
                    formulation_ids.append(cur.fetchone()[0])

            # components: include 1 api + 3-6 excipients
            for fid in formulation_ids:
                api_mat = random.choice([m for m in material_ids])
                cur.execute(
                    """
                    INSERT INTO new_data.formulation_component(formulation_id, material_id, role, qty_per_unit, unit)
                    VALUES (%s,%s,'api',%s,'mg')
                    ON CONFLICT (formulation_id, material_id) DO NOTHING
                    """,
                    (fid, api_mat, round(random.uniform(5, 500), 6)),
                )
                for mid in random.sample(material_ids, k=random.randint(3, 6)):
                    cur.execute(
                        """
                        INSERT INTO new_data.formulation_component(formulation_id, material_id, role, qty_per_unit, unit)
                        VALUES (%s,%s,'excipient',%s,'mg')
                        ON CONFLICT (formulation_id, material_id) DO NOTHING
                        """,
                        (fid, mid, round(random.uniform(1, 200), 6)),
                    )

            # -----------------------
            # Equipment + Work orders + Batches + Steps + Consumption
            # -----------------------
            eq_rows = []
            for i in range(18):
                eq_rows.append((random.choice(site_ids), f"EQ-{i+1:04d}", f"{random.choice(['Mixer','Press','Coater','Filler'])} {i+1}", random.choice(["mixer","tablet_press","coater","filler"]), "active"))
            bulk_insert(cur, "new_data.equipment", ["site_id", "equipment_code", "name", "equipment_type", "status"], eq_rows)
            equipment_ids = fetch_ids(cur, "new_data.equipment", "equipment_id")

            wo_rows = []
            for i in range(max(20, N_BATCHES // 2)):
                pid = random.choice(product_ids)
                fid = random.choice([f for f in formulation_ids])
                wo_rows.append((random.choice(site_ids), pid, fid, round(random.uniform(1, 50), 3), "batch", rand_past(365), rand_past(365), random.choice(["planned","in_process","completed"])))
            bulk_insert(
                cur,
                "new_data.work_order",
                ["site_id","product_id","formulation_id","planned_qty","planned_unit","scheduled_start","scheduled_end","status"],
                wo_rows,
            )
            wo_ids = fetch_ids(cur, "new_data.work_order", "wo_id")

            batch_rows = []
            for i in range(N_BATCHES):
                wo = random.choice(wo_ids) if maybe(0.8) else None
                pid = random.choice(product_ids)
                mfg = fake.date_between(start_date="-1y", end_date="today")
                exp = mfg + timedelta(days=random.choice([365, 540, 730]))
                batch_rows.append(
                    (
                        wo,
                        random.choice(site_ids),
                        pid,
                        f"BATCH-{i+1:05d}",
                        mfg,
                        exp,
                        random.choice(["in_process","quarantine","released","rejected"]),
                    )
                )
            bulk_insert(
                cur,
                "new_data.batch",
                ["wo_id","site_id","product_id","batch_number","manufacture_date","expiry_date","status"],
                batch_rows,
            )
            batch_ids = fetch_ids(cur, "new_data.batch", "batch_id")

            step_rows = []
            for bid in batch_ids:
                for step_no in range(1, N_BATCH_STEPS + 1):
                    st = rand_past(365)
                    en = st + timedelta(minutes=random.randint(15, 240))
                    step_rows.append(
                        (
                            bid,
                            step_no,
                            f"Step {step_no}: {random.choice(['Weigh','Mix','Granulate','Dry','Compress','Coat','Fill','Pack'])}",
                            random.choice(equipment_ids),
                            st,
                            en,
                            random.choice(employee_ids),
                        )
                    )
            bulk_insert(
                cur,
                "new_data.batch_step",
                ["batch_id","step_no","step_name","equipment_id","started_at","ended_at","operator_employee_id"],
                step_rows,
            )

            cons_rows = []
            for _ in range(N_CONSUMPTIONS):
                cons_rows.append((random.choice(batch_ids), random.choice(material_lot_ids), round(random.uniform(0.1, 50), 3), rand_past(365)))
            bulk_insert(
                cur,
                "new_data.batch_material_consumption",
                ["batch_id","material_lot_id","qty_used","used_at"],
                cons_rows,
            )

            # -----------------------
            # QC: methods/tests/specs/samples/results
            # -----------------------
            method_rows = []
            for i in range(N_METHODS):
                method_rows.append((f"MTH-{i+1:04d}", f"{random.choice(['HPLC','GC','UV','Karl Fischer','Dissolution'])} Method {i+1}", random.randint(1, 3), fake.date_between(start_date="-3y", end_date="today"), "active"))
            bulk_insert(cur, "new_data.qc_method", ["method_code","name","version_no","effective_from","status"], method_rows)
            method_ids = fetch_ids(cur, "new_data.qc_method", "method_id")

            test_rows = []
            target_types = ["batch", "material_lot", "stability"]
            for i in range(N_TESTS):
                test_rows.append((f"TST-{i+1:04d}", f"{random.choice(['Assay','Impurities','Dissolution','Moisture','ID'])} Test {i+1}", random.choice(target_types), random.choice(method_ids)))
            bulk_insert(cur, "new_data.qc_test", ["test_code","name","target_type","method_id"], test_rows)
            test_ids = fetch_ids(cur, "new_data.qc_test", "test_id")

            spec_rows = []
            for _ in range(N_SPECS):
                if maybe(0.6):
                    spec_rows.append((random.choice(product_ids), None, random.choice(test_ids), random.choice(["95-105%", "NMT 0.5%", "Pass", "0.0-2.0%"]), fake.date_between(start_date="-2y", end_date="today"), "active"))
                else:
                    spec_rows.append((None, random.choice(material_ids), random.choice(test_ids), random.choice(["Pass", "Conforms", "NMT 1.0%"]), fake.date_between(start_date="-2y", end_date="today"), "active"))
            bulk_insert(cur, "new_data.qc_specification", ["product_id","material_id","test_id","limit_text","effective_from","status"], spec_rows)

            sample_rows = []
            sample_codes = safe_unique("SMP-", N_SAMPLES)
            for i in range(N_SAMPLES):
                sample_rows.append(
                    (
                        sample_codes[i],
                        random.choice(site_ids),
                        random.choice(batch_ids) if maybe(0.7) else None,
                        random.choice(material_lot_ids) if maybe(0.5) else None,
                        rand_past(365),
                        random.choice(employee_ids),
                    )
                )
            bulk_insert(
                cur,
                "new_data.qc_sample",
                ["sample_code","site_id","batch_id","material_lot_id","collected_at","collected_by_employee_id"],
                sample_rows,
            )
            sample_ids = fetch_ids(cur, "new_data.qc_sample", "sample_id")

            # results: ensure uniqueness (sample_id, test_id)
            seen = set()
            result_rows = []
            for _ in range(N_RESULTS):
                sid = random.choice(sample_ids)
                tid = random.choice(test_ids)
                if (sid, tid) in seen:
                    continue
                seen.add((sid, tid))
                pf = random.choice(["pass", "pass", "fail", "pending"])
                result_rows.append(
                    (
                        sid,
                        tid,
                        random.choice(employee_ids),
                        random.choice(["99.2%", "101.0%", "Pass", "Conforms", "0.3%"]),
                        pf,
                        rand_past(365) if pf != "pending" else None,
                    )
                )
            bulk_insert(
                cur,
                "new_data.qc_result",
                ["sample_id","test_id","analyst_employee_id","result_value","pass_fail","completed_at"],
                result_rows,
            )

            # -----------------------
            # Stability
            # -----------------------
            stability_rows = []
            for i in range(N_STABILITY_STUDIES):
                pid = random.choice(product_ids)
                bid = random.choice(batch_ids) if maybe(0.6) else None
                stability_rows.append(
                    (
                        pid,
                        bid,
                        f"STAB-PROT-{i+1:04d}",
                        random.choice(["25C/60%RH", "30C/65%RH", "40C/75%RH"]),
                        fake.date_between(start_date="-2y", end_date="today"),
                        None,
                        random.choice(["ongoing", "ongoing", "completed"]),
                    )
                )
            bulk_insert(
                cur,
                "new_data.stability_study",
                ["product_id","batch_id","protocol_code","condition_label","start_date","end_date","status"],
                stability_rows,
            )
            study_ids = fetch_ids(cur, "new_data.stability_study", "study_id")

            tp_rows = []
            for sid in study_ids:
                start_dt = fake.date_between(start_date="-2y", end_date="today")
                for m in [0, 3, 6, 9, 12]:
                    tp_rows.append((sid, m, start_dt + timedelta(days=30*m)))
            for sid, month_no, due_date in tp_rows:
                cur.execute(
                    """
                    INSERT INTO new_data.stability_timepoint(study_id, month_no, due_date)
                    VALUES (%s,%s,%s)
                    ON CONFLICT (study_id, month_no) DO NOTHING
                    """,
                    (sid, month_no, due_date),
                )
            timepoint_ids = fetch_ids(cur, "new_data.stability_timepoint", "timepoint_id")

            # make some stability samples by reusing qc_sample rows (must be unique)
            for tp in random.sample(timepoint_ids, k=min(len(timepoint_ids), 120)):
                # pick a qc_sample id that isn't already used in stability_sample
                cur.execute("SELECT sample_id FROM new_data.qc_sample ORDER BY random() LIMIT 1")
                sample_id = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO new_data.stability_sample(timepoint_id, sample_id)
                    VALUES (%s,%s)
                    ON CONFLICT (sample_id) DO NOTHING
                    """,
                    (tp, sample_id),
                )

            # -----------------------
            # Quality System: deviations / capa / change controls
            # -----------------------
            dev_rows = []
            for i in range(N_DEVIATIONS):
                dev_rows.append(
                    (
                        random.choice(site_ids),
                        random.choice(batch_ids) if maybe(0.6) else None,
                        random.choice(employee_ids),
                        rand_past(365),
                        random.choice(["minor","major","critical"]),
                        fake.paragraph(nb_sentences=2),
                        random.choice(["open","investigation","closed"]),
                    )
                )
            bulk_insert(
                cur,
                "new_data.deviation",
                ["site_id","batch_id","opened_by_employee_id","opened_at","severity","description","status"],
                dev_rows,
            )
            deviation_ids = fetch_ids(cur, "new_data.deviation", "deviation_id")

            capa_rows = []
            for i in range(N_CAPAS):
                capa_rows.append(
                    (
                        random.choice(deviation_ids) if maybe(0.7) else None,
                        rand_past(365),
                        random.choice(employee_ids),
                        fake.sentence(nb_words=8),
                        fake.paragraph(nb_sentences=2),
                        rand_future_date(180),
                        random.choice(["open","in_progress","verified","closed"]),
                    )
                )
            bulk_insert(
                cur,
                "new_data.capa",
                ["deviation_id","opened_at","owner_employee_id","root_cause","action_plan","due_date","status"],
                capa_rows,
            )
            capa_ids = fetch_ids(cur, "new_data.capa", "capa_id")

            change_rows = []
            for i in range(N_CHANGES):
                change_rows.append(
                    (
                        random.choice(site_ids),
                        random.choice(employee_ids),
                        rand_past(365),
                        random.choice(["process","equipment","method","document"]),
                        fake.paragraph(nb_sentences=2),
                        fake.sentence(nb_words=10),
                        random.choice(["proposed","approved","implemented","closed"]),
                    )
                )
            bulk_insert(
                cur,
                "new_data.change_control",
                ["site_id","requested_by_employee_id","requested_at","change_type","description","impact_assessment","status"],
                change_rows,
            )

            # -----------------------
            # Training
            # -----------------------
            course_rows = []
            for i in range(N_COURSES):
                course_rows.append((f"CRS-{i+1:04d}", fake.sentence(nb_words=5), random.choice(document_ids) if maybe(0.6) else None, random.choice([180, 365, 730, None])))
            bulk_insert(cur, "new_data.training_course", ["course_code","title","document_id","retrain_interval_days"], course_rows)
            course_ids = fetch_ids(cur, "new_data.training_course", "course_id")

            tr_rows = []
            for _ in range(N_TRAINING_RECORDS):
                emp = random.choice(employee_ids)
                crs = random.choice(course_ids)
                completed = fake.date_between(start_date="-2y", end_date="today")
                # expiry if retrain interval exists
                cur.execute("SELECT retrain_interval_days FROM new_data.training_course WHERE course_id=%s", (crs,))
                interval = cur.fetchone()[0]
                expires = (completed + timedelta(days=int(interval))) if interval else None
                status = "expired" if expires and expires < date.today() else "valid"
                tr_rows.append((emp, crs, completed, expires, status))
            # allow duplicates across different completed_on but unique constraint exists for (employee_id, course_id, completed_on)
            bulk_insert(cur, "new_data.training_record", ["employee_id","course_id","completed_on","expires_on","status"], tr_rows)

            # -----------------------
            # Audits + Findings
            # -----------------------
            audit_rows = []
            for i in range(N_AUDITS):
                audit_rows.append(
                    (
                        random.choice(site_ids),
                        random.choice(["internal","supplier","regulatory"]),
                        random.choice(supplier_ids) if maybe(0.4) else None,
                        fake.date_between(start_date="-1y", end_date="today"),
                        fake.date_between(start_date="-1y", end_date="today"),
                        random.choice(employee_ids),
                        random.choice(["planned","performed","closed"]),
                    )
                )
            bulk_insert(
                cur,
                "new_data.audit",
                ["site_id","audit_type","audited_party_supplier_id","planned_date","performed_date","lead_auditor_employee_id","status"],
                audit_rows,
            )
            audit_ids = fetch_ids(cur, "new_data.audit", "audit_id")

            finding_rows = []
            for i in range(N_FINDINGS):
                finding_rows.append(
                    (
                        random.choice(audit_ids),
                        f"FND-{i+1:05d}",
                        random.choice(["minor","major","critical"]),
                        fake.paragraph(nb_sentences=2),
                        random.choice(capa_ids) if maybe(0.4) else None,
                        random.choice(["open","closed"]),
                    )
                )
            bulk_insert(
                cur,
                "new_data.audit_finding",
                ["audit_id","finding_code","severity","description","capa_id","status"],
                finding_rows,
            )

            # -----------------------
            # Regulatory: agencies/markets/submissions + linking docs
            # -----------------------
            agencies = [("FDA","USA"), ("EMA","EU"), ("TGA","Australia"), ("MHRA","UK")]
            for name, country in agencies:
                cur.execute("INSERT INTO new_data.reg_agency(name,country) VALUES (%s,%s) ON CONFLICT (name) DO NOTHING", (name, country))
            agency_ids = fetch_ids(cur, "new_data.reg_agency", "agency_id")

            markets = ["USA", "Australia", "UK", "Germany", "India", "Singapore"]
            for c in markets:
                cur.execute("INSERT INTO new_data.market(country) VALUES (%s) ON CONFLICT (country) DO NOTHING", (c,))
            # market_ids not used further (yet), but table exists

            sub_rows = []
            for i in range(N_SUBMISSIONS):
                sub_rows.append(
                    (
                        random.choice(product_ids),
                        random.choice(agency_ids),
                        random.choice(["IND","NDA","ANDA","CTA","DMF"]),
                        fake.date_between(start_date="-2y", end_date="today"),
                        random.choice(["draft","submitted","under_review","approved","rejected"]),
                    )
                )
            bulk_insert(
                cur,
                "new_data.submission",
                ["product_id","agency_id","submission_type","submitted_on","status"],
                sub_rows,
            )
            submission_ids = fetch_ids(cur, "new_data.submission", "submission_id")

            # link docs to submissions (unique)
            for sid in random.sample(submission_ids, k=min(len(submission_ids), 20)):
                for did in random.sample(document_ids, k=random.randint(1, 4)):
                    cur.execute(
                        """
                        INSERT INTO new_data.submission_document(submission_id, document_id)
                        VALUES (%s,%s)
                        ON CONFLICT (submission_id, document_id) DO NOTHING
                        """,
                        (sid, did),
                    )

            # certificates
            cert_rows = []
            cert_types = ["GMP", "ISO", "CoA", "License"]
            for i in range(N_CERTIFICATES):
                cert_rows.append(
                    (
                        random.choice(cert_types),
                        random.choice(site_ids) if maybe(0.6) else None,
                        random.choice(supplier_ids) if maybe(0.6) else None,
                        random.choice(agency_ids) if maybe(0.6) else None,
                        f"CERT-{i+1:06d}",
                        fake.date_between(start_date="-3y", end_date="today"),
                        fake.date_between(start_date="today", end_date="+3y") if maybe(0.85) else fake.date_between(start_date="-2y", end_date="today"),
                        "valid",
                    )
                )
            bulk_insert(
                cur,
                "new_data.certificate",
                ["cert_type","site_id","supplier_id","agency_id","certificate_no","issued_on","expires_on","status"],
                cert_rows,
            )
            # update certificate status based on expiry
            cur.execute(
                """
                UPDATE new_data.certificate
                SET status = CASE
                  WHEN expires_on IS NOT NULL AND expires_on < CURRENT_DATE THEN 'expired'
                  ELSE 'valid'
                END
                """
            )

            # -----------------------
            # Audit log
            # -----------------------
            ent = ["company","site","material","supplier","po","receipt","material_lot","batch","qc_sample","qc_result","document","submission","certificate","deviation","capa"]
            log_rows = []
            for _ in range(N_AUDIT_LOG):
                log_rows.append(
                    (
                        rand_past(900),
                        random.choice(user_ids) if maybe(0.8) else None,
                        random.choice(["CREATE","UPDATE","VIEW","DELETE"]),
                        random.choice(ent),
                        random.randint(1, 200000),
                        Json({"seed": SEED, "ip": fake.ipv4(), "note": fake.word()}),
                    )
                )
            bulk_insert(
                cur,
                "new_data.audit_log",
                ["occurred_at","actor_user_id","action","entity","entity_id","details"],
                log_rows,
            )

            conn.commit()

            # -----------------------
            # Print quick summary
            # -----------------------
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='new_data' AND table_type='BASE TABLE'"
            )
            tables = cur.fetchone()[0]
            print(f"✅ Done. new_data populated. Tables in schema: {tables}")

            for q, label in [
                ("SELECT COUNT(*) FROM new_data.employee", "employees"),
                ("SELECT COUNT(*) FROM new_data.supplier", "suppliers"),
                ("SELECT COUNT(*) FROM new_data.material", "materials"),
                ("SELECT COUNT(*) FROM new_data.purchase_order", "purchase_orders"),
                ("SELECT COUNT(*) FROM new_data.material_lot", "material_lots"),
                ("SELECT COUNT(*) FROM new_data.drug_product", "products"),
                ("SELECT COUNT(*) FROM new_data.batch", "batches"),
                ("SELECT COUNT(*) FROM new_data.qc_sample", "qc_samples"),
                ("SELECT COUNT(*) FROM new_data.qc_result", "qc_results"),
                ("SELECT COUNT(*) FROM new_data.submission", "submissions"),
                ("SELECT COUNT(*) FROM new_data.certificate", "certificates"),
            ]:
                cur.execute(q)
                print(f"  - {label}: {cur.fetchone()[0]}")

if __name__ == "__main__":
    main()
