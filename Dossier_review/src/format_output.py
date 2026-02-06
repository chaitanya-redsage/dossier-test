from __future__ import annotations

from typing import Any, Dict, List


def format_result_to_json(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts your current detect_expiry() + proofread output into:
    {
      "spelling_mistakes": [...],
      "grammatical_errors": [...],
      "validity_issues": {...}
    }
    """

    proofread = result.get("proofread") or {}

    # 1) spelling mistakes
    spelling: List[Dict[str, Any]] = []
    for item in (proofread.get("spelling") or []):
        spelling.append({
            "word": item.get("word"),
            "suggestions": item.get("suggestions", []),
        })

    # 2) grammatical errors
    grammar: List[Dict[str, Any]] = []
    for item in (proofread.get("grammar") or []):
        grammar.append({
            "message": item.get("message"),
            "offset": item.get("offset"),
            "length": item.get("errorLength"),
            "rule_id": item.get("ruleId"),
            "suggestions": item.get("replacements", []),
        })

    # 3) validity issues (expiry pipeline)
    evidence = result.get("evidence") or {}
    issue_date = evidence.get("issue_date") or {}
    duration = evidence.get("duration") or {}

    validity_issues: Dict[str, Any] = {
        "type": result.get("type"),
        "confidence": result.get("confidence"),
        "expiry_date": result.get("expiry_date"),
        "issue_date": issue_date.get("parsed_iso") or issue_date.get("raw"),
        "valid_for": (
            {
                "value": duration.get("value"),
                "unit": duration.get("unit"),
                "context": duration.get("context"),
                "page": duration.get("page"),
            }
            if duration else None
        ),
        "evidence": {
            "issue_date": (
                {
                    "raw": issue_date.get("raw"),
                    "parsed_iso": issue_date.get("parsed_iso"),
                    "context": issue_date.get("context"),
                    "page": issue_date.get("page"),
                }
                if issue_date else None
            ),
            "duration": (
                {
                    "value": duration.get("value"),
                    "unit": duration.get("unit"),
                    "context": duration.get("context"),
                    "page": duration.get("page"),
                }
                if duration else None
            ),
        },
        "reason": result.get("reason", ""),
    }

    # remove None blocks
    if validity_issues.get("valid_for") is None:
        validity_issues.pop("valid_for", None)

    return {
        "spelling_mistakes": spelling,
        "grammatical_errors": grammar,
        "validity_issues": validity_issues,
    }
