import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path so we can import deed_pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env for OPENAI_API_KEY before importing deed_pipeline
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import deed_pipeline as dp


BLOCK_RE = re.compile(r"\*\*\*\s*RECORDING\s+REQ\s*\*\*\*(.*?)\*\*\*\s*END\s*\*\*\*", re.IGNORECASE | re.DOTALL)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def extract_blocks(all_text: str) -> List[str]:
    matches = BLOCK_RE.findall(all_text)
    blocks: List[str] = []
    for m in matches:
        blocks.append(("*** RECORDING REQ ***" + m + "*** END ***").strip())
    return blocks


def _flatten_failures(failures: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Flatten group->list into [(group, failure), ...] for order-independent matching."""
    out: List[Dict[str, Any]] = []
    for group, lst in failures.items():
        for f in lst:
            out.append({**f, "_group": group})
    return out


def compare_case(expected_case: Dict[str, Any], actual_doc: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Compare only deterministic outputs: status, and failures (code + fields).
    Ignores details, summary, evidence, and extracted values (those vary with LLM).
    """
    problems: List[str] = []
    exp_status = expected_case["expect_doc_status"]
    act_status = actual_doc.get("status")
    if act_status != exp_status:
        problems.append(f"status mismatch: expected={exp_status} actual={act_status}")

    exp_fails = expected_case.get("expect_failures", [])
    act_flat = _flatten_failures(actual_doc.get("failures", {}))
    used = [False] * len(act_flat)

    for ef in exp_fails:
        group = ef["group"]
        code = ef["code"]
        fields = ef.get("fields")

        matched = False
        for i, c in enumerate(act_flat):
            if used[i]:
                continue
            if c.get("_group") != group or c.get("code") != code:
                continue
            if fields is not None:
                c_fields = c.get("fields")
                if c_fields is None or sorted(c_fields) != sorted(fields):
                    continue
            used[i] = True
            matched = True
            break

        if not matched:
            problems.append(f"missing failure: group={group} code={code}" + (f" fields={fields}" if fields else ""))

    return (len(problems) == 0), problems


def _test_counties_file_missing() -> bool:
    """Test COUNTIES_FILE_MISSING when counties.json is missing. Returns True if pass."""
    out = dp.run_pipeline_on_blocks(
        blocks=[dp.DEFAULT_RAW_TEXT],
        counties_path="/nonexistent/counties_404.json",
        model_id=dp.DEFAULT_MODEL_ID,
        include_audit=False,
    )
    schema_failures = out.get("failures", {}).get("schema", [])
    codes = [f.get("code") for f in schema_failures]
    return "COUNTIES_FILE_MISSING" in codes and out.get("status") == "FAIL"


def _test_counties_file_invalid() -> bool:
    """Test COUNTIES_FILE_INVALID when counties.json is malformed. Returns True if pass."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{ invalid json }")
        invalid_path = f.name

    try:
        out = dp.run_pipeline_on_blocks(
            blocks=[dp.DEFAULT_RAW_TEXT],
            counties_path=invalid_path,
            model_id=dp.DEFAULT_MODEL_ID,
            include_audit=False,
        )
        schema_failures = out.get("failures", {}).get("schema", [])
        codes = [f.get("code") for f in schema_failures]
        return "COUNTIES_FILE_INVALID" in codes and out.get("status") == "FAIL"
    finally:
        try:
            os.unlink(invalid_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save output.json and results.json to unitTest/")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    input_path = os.path.join(here, "input.txt")
    expected_path = os.path.join(here, "expected.json")

    output_path = os.path.join(here, "output.json")
    results_path = os.path.join(here, "results.json")

    # Phase 0: Reference data tests (no API calls)
    runner_tests: List[Tuple[str, bool]] = []
    runner_tests.append(("COUNTIES_FILE_MISSING", _test_counties_file_missing()))
    runner_tests.append(("COUNTIES_FILE_INVALID", _test_counties_file_invalid()))

    raw = read_text(input_path)
    blocks = extract_blocks(raw)
    expected_cases = json.loads(read_text(expected_path))

    if len(blocks) != len(expected_cases):
        print(f"[warn] blocks({len(blocks)}) != expected_cases({len(expected_cases)})")

    # Run pipeline with real LLM calls (no monkeypatching)
    counties_path = os.path.join(here, "counties_test.json")
    if not os.path.exists(counties_path):
        raise FileNotFoundError(f"Missing unit test counties file: {counties_path}")
    out_obj = dp.run_pipeline_on_blocks(
        blocks,
        counties_path=counties_path,
        model_id=dp.DEFAULT_MODEL_ID,
        include_audit=True,
    )

    if args.save:
        write_json(output_path, out_obj)

    docs = out_obj.get("documents", [])
    matched = 0
    mismatched = 0
    details: List[Dict[str, Any]] = []

    for i, exp in enumerate(expected_cases):
        name = exp.get("name", f"case_{i}")
        act_doc = docs[i]["doc"] if i < len(docs) else None
        if act_doc is None:
            mismatched += 1
            details.append({"name": name, "ok": False, "problems": ["missing actual doc output"]})
            continue

        ok, problems = compare_case(exp, act_doc)
        if ok:
            matched += 1
        else:
            mismatched += 1
        details.append({"name": name, "ok": ok, "problems": problems})

    total = matched + mismatched
    pass_rate = (matched / total) if total else 0.0

    # Also print pass/fail stats across docs (pipeline-level)
    doc_pass = sum(1 for d in docs if d["doc"].get("status") == "PASS")
    doc_fail = sum(1 for d in docs if d["doc"].get("status") == "FAIL")
    doc_total = len(docs)
    doc_pass_rate = (doc_pass / doc_total) if doc_total else 0.0

    summary = {
        "matched": matched,
        "mismatched": mismatched,
        "total": total,
        "pass_rate": pass_rate,
        "doc_pass": doc_pass,
        "doc_fail": doc_fail,
        "doc_total": doc_total,
        "doc_pass_rate": doc_pass_rate,
        "details": details,
        "output_file": os.path.basename(output_path),
    }

    if args.save:
        write_json(results_path, summary)

    runner_pass = sum(1 for _, ok in runner_tests if ok)
    runner_total = len(runner_tests)

    print("\n==== UNIT TEST SUMMARY ====")
    print("\n---- Reference data (Phase 0) ----")
    for name, ok in runner_tests:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"  Phase 0 pass rate: {runner_pass}/{runner_total}")

    print("\n---- Document cases (Phase 1) ----")
    print(f"Matched expected:   {matched}/{total}")
    print(f"Mismatched:         {mismatched}/{total}")
    print(f"Test pass rate:     {pass_rate:.2%}")
    print("\n---- Pipeline doc stats ----")
    print(f"Docs PASS:          {doc_pass}/{doc_total}")
    print(f"Docs FAIL:          {doc_fail}/{doc_total}")
    print(f"Doc pass rate:      {doc_pass_rate:.2%}")

    if mismatched:
        print("\n==== MISMATCH DETAILS ====")
        for d in details:
            if d["ok"]:
                continue
            print(f"\n[{d['name']}]")
            for p in d["problems"]:
                print(f"  - {p}")

    if args.save:
        print(f"\nWrote output:  {output_path}")
        print(f"Wrote results: {results_path}")


if __name__ == "__main__":
    main()