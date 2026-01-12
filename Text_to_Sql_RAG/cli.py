import json
from llm_agent_rag import answer_nl

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (natural language). Type 'exit' to quit:\n> ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        try:
            out = answer_nl(q)

            print("\n--- RETRIEVED TABLES ---")
            for h in out["retrieved"][:10]:
                p = h["payload"] or {}
                print(f"- {p.get('doc_id')} (score={h['score']:.4f}) :: {p.get('description','')[:90]}")

            print("\n--- PLAN ---")
            print(json.dumps(out["plan"], indent=2, ensure_ascii=False))

            print("\n--- SQL ---")
            print(out["sql"])

            print("\n--- RESULT ---")
            cols = out["result"]["columns"]
            rows = out["result"]["rows"]
            print(cols)
            for r in rows[:20]:
                print(r)
            if len(rows) > 20:
                print(f"... ({len(rows)} rows total, showing first 20)")

        except Exception as e:
            print(f"\nERROR: {e}")

