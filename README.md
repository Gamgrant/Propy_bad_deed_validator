# Deed Pipeline

A fail-closed pipeline that extracts deed fields from OCR text, enriches them with reference data, and validates them. Each document gets PASS/FAIL with structured error codes.

**Flow:** Raw text → LLM extraction → presence detection (line-by-line) → normalization → enrichment (county→tax_rate) → validation. Deterministic rules drive most failures; LLM loopbacks handle ambiguity (multiple dates, amounts, or currencies in the text).

---

## LangExtract & design choices

**Why LangExtract?** We use [LangExtract](https://github.com/google-deepmind/langextract) instead of raw OpenAI completion APIs because it gives structured extractions with exact character spans. That lets us tie each extracted value back to its source span for auditability and alignment checks.

**Design choices:**
- **Few-shot prompting** — Example inputs/outputs steer the LLM to exact-span extraction (no paraphrasing).
- **Dual-layer presence** — A deterministic line-by-line presence pass runs alongside LLM extraction; if the OCR line has a value but the LLM returns nothing, we flag `EXTRACTED_VALUE_MISSING`. This catches LLM omissions without trusting extraction alone.
- **Loopbacks only for ambiguity** — Main extraction is one shot. We re-call the LLM only when we need candidate lists (e.g., all date mentions) to detect ambiguous cases.
- **Deterministic validation first** — Date/amount parsing, bounds, and comparisons are rule-based. LLM is used for extraction and disambiguation, not for validation decisions.

---

## Layer architecture

| Layer | Function(s) | Purpose |
|-------|-------------|---------|
| **1. Extraction** | `langextract_deed()` | LLM extracts raw spans (doc_id, county, dates, amount, etc.) |
| **2. Presence** | `detect_field_presence()` | Line-by-line: ABSENT / PRESENT_EMPTY / PRESENT_WITH_VALUE |
| **3. Normalization** | `parse_iso_date()`, `parse_money_digits_any()`, `words_to_int_us()`, `normalize_county()` | Raw strings → typed values |
| **4. Enrichment** | `match_county()`, `langextract_candidate_counties()` | County → tax_rate via fuzzy match; LLM suggests expansion if needed |
| **5. Validation** | 6 sub-layers | Presence → Schema → County → Date → Amount → Loopback (ambiguity) |

**Validation order:** Presence (FIELD_ABSENT, FIELD_PRESENT_EMPTY) → Schema (UNKNOWN_FIELD) → County (UNKNOWN_COUNTY, COUNTY_EXPANSION_NOT_IN_REFERENCE) → Date (parse, bounds, order, ambiguity) → Amount (parse, reconcile, ambiguity, currency) → Loopback flags (AMBIGUOUS_DATE, AMBIGUOUS_AMOUNT, AMBIGUOUS_CURRENCY).

---

## Edge cases & error codes

| Category | Conditions | Codes |
|----------|------------|-------|
| **Input** | Field absent or label present but empty | FIELD_ABSENT, FIELD_PRESENT_EMPTY |
| **Reference data** | Missing or malformed `counties.json` | COUNTIES_FILE_MISSING, COUNTIES_FILE_INVALID |
| **Schema** | LLM returns unknown field | UNKNOWN_FIELD_EXTRACTED |
| **County** | No match or expansion not in ref | UNKNOWN_COUNTY, COUNTY_EXPANSION_NOT_IN_REFERENCE |
| **Dates** | Parse fail, out of bounds, recorded < signed, multiple candidates | DATE_PARSE_ERROR, DATE_UNREASONABLE, DATE_ORDER_ERROR, AMBIGUOUS_DATE |
| **Amounts** | Parse fail, digits ≠ words, multiple amounts/currencies | AMOUNT_*_PARSE_ERROR, AMOUNT_MISMATCH_ERROR, AMBIGUOUS_AMOUNT, AMBIGUOUS_CURRENCY, CURRENCY_MISMATCH |

**Amount fallback:** If the LLM omits `amount_digits` but the presence layer finds a value (e.g., `TBD`), we try to parse it. Parse failure → `AMOUNT_DIGITS_PARSE_ERROR`.

---

## Design Q&A

**How does date validation work?** Deterministic: `parse_iso_date()` enforces YYYY-MM-DD; `date_reasonableness_error()` checks 1900–today+30 days; recorded must be ≥ signed. The loopback `langextract_candidate_dates()` finds all date mentions; if there are multiple, we flag `AMBIGUOUS_DATE`.

**How does "S. Clara" resolve?** `normalize_county()` does regex cleanup; `match_county()` uses rapidfuzz (WRatio) against `counties.json`. If no match, LLM loopback suggests a normalized candidate; we retry. Still no match → UNKNOWN_COUNTY or COUNTY_EXPANSION_NOT_IN_REFERENCE.

---

## Requirements

```bash
pip install -r requirements.txt
```

> **Required for pipeline and unit tests:** Set `OPENAI_API_KEY` or `openAI_API_key` in your environment (e.g. via `.env`).

---

## How to run

### Pipeline (real LLM calls)

```bash
# Run with built-in default deed text (no file needed)
python deed_pipeline.py

# Read from file, print doc stats + JSON
python deed_pipeline.py --input_file unitTest/input.txt

# Save output
python deed_pipeline.py --input_file unitTest/input.txt --output out.json

# Full options
python deed_pipeline.py --input_file path.txt --counties counties.json --output out.json --audit --verbose
```

With no `--text` or `--input_file`, the pipeline uses the hardcoded `DEFAULT_RAW_TEXT`. Input must be deed blocks wrapped in `*** RECORDING REQ ***` … `*** END ***`.

| Flag | Default | Description |
|------|---------|-------------|
| `--input_file` | — | Path to .txt with deed blocks |
| `--text` | — | Inline text (overrides file) |
| `--counties` | `counties.json` | Path to counties reference |
| `--model` | `gpt-4o` | LLM model ID |
| `--output` | — | Write JSON to file (omit = print only) |
| `--audit` | off | Include forensic audit in output |
| `--verbose` | off | LangExtract progress/warnings |

Pipeline always prints doc stats (Docs PASS, Docs FAIL, Doc pass rate).

### Unit tests (real LLM)

Unit tests run the **full pipeline with real API calls**. They compare only deterministic outputs: `status` and `failures` (code + fields); details, summary, and extracted values are ignored.

**Phase 0** (no API calls) validates reference-data failures: `COUNTIES_FILE_MISSING`, `COUNTIES_FILE_INVALID`.  
**Phase 1** runs each deed block through the pipeline and asserts expected failure codes.

```bash
# Run tests, print summary + doc stats (no files written)
python unitTest/run_tests.py

# Save output.json and results.json
python unitTest/run_tests.py --save
```

| Flag | Description |
|------|-------------|
| `--save` | Write output.json and results.json to unitTest/ |

**Test coverage vs edge cases:**

| README edge case | Test name(s) |
|-----------------|--------------|
| Input ABSENT | REQUIRED_AMOUNT_LABEL_ABSENT |
| Input PRESENT_EMPTY | REQUIRED_AMOUNT_PRESENT_EMPTY, COUNTY_PRESENT_EMPTY |
| Reference: missing counties | Phase 0: COUNTIES_FILE_MISSING |
| Reference: invalid counties | Phase 0: COUNTIES_FILE_INVALID |
| County: no match | COUNTY_UNKNOWN, COUNTY_EXPANSION_NOT_IN_REFERENCE |
| County: expansion not in ref | COUNTY_EXPANSION_NOT_IN_REFERENCE |
| Date parse fail | DATE_PARSE_ERROR |
| Date too old/future | DATE_UNREASONABLE_TOO_OLD, DATE_UNREASONABLE_TOO_FUTURE |
| Date order error | BASELINE_DATE_AND_AMOUNT_MISMATCH |
| Date ambiguous | AMBIGUOUS_DATE_WARNING_PLUS_FAIL |
| Amount digits parse fail | AMOUNT_DIGITS_PARSE_ERROR |
| Amount words parse fail | AMOUNT_WORDS_PARSE_ERROR |
| Amount mismatch | BASELINE_DATE_AND_AMOUNT_MISMATCH, AMBIGUOUS_* |
| Amount ambiguous | AMBIGUOUS_AMOUNT_WARNING_PLUS_FAIL |
| Currency ambiguous | AMBIGUOUS_CURRENCY_WARNING_PLUS_FAIL |
| Currency mismatch | CURRENCY_MISMATCH |
| Schema: unknown field | Not tested (LLM-dependent) |

---
