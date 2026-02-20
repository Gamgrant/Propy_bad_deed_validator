# deed_pipeline.py
from __future__ import annotations

import argparse
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import langextract as lx
from rapidfuzz import fuzz, process

# =============================
# CONFIG
# =============================
DEFAULT_COUNTIES_FILE = os.path.join(os.path.dirname(__file__), "counties.json")
DEFAULT_MODEL_ID = "gpt-4o"

DEFAULT_RAW_TEXT = """*** RECORDING REQ ***
Doc: DEED-TRUST-0042
County: S. Clara  |  State: CA
Date Signed: 2024-01-15
Date Recorded: 2024-01-10
Grantor:  T.E.S.L.A. Holdings LLC
Grantee:  John  &  Sarah  Connor
Amount: $1,250,000.00 (One Million Two Hundred Thousand Dollars)
APN: 992-001-XA
Status: PRELIMINARY
*** END ***"""

# Fail-closed on unknown fields extracted by LLM (warning by default; flip to "error" if desired)
FAIL_CLOSED_ON_UNKNOWN_FIELDS = True

# Always run loopback passes (ambiguity + evidence)
ALWAYS_RUN_LOOPBACK = True

# Date reasonableness bounds
DATE_MIN = date(1900, 1, 1)
DATE_MAX_FUTURE_DAYS = 30

# County fuzzy match threshold
COUNTY_MIN_SCORE = 85
COUNTY_RETRY_MARGIN = 5

# Presence status constants
ABSENT = "ABSENT"
PRESENT_EMPTY = "PRESENT_EMPTY"
PRESENT_WITH_VALUE = "PRESENT_WITH_VALUE"

# Which fields must be present with a value (hard fail otherwise)
REQUIRED_FIELDS = {
    "doc_id",
    "county",
    "state",
    "date_signed",
    "date_recorded",
    "grantor",
    "grantee",
    "amount_digits",
    "amount_words",
    "apn",
}

# Optional fields (never hard-fail purely for presence)
OPTIONAL_FIELDS = {"status", "currency"}

KNOWN_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS

DEFAULT_INCLUDE_AUDIT = False  # audit OFF by default

# =============================
# LOGGING
# =============================
def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.ERROR)
    for name in ["absl", "langextract"]:
        logging.getLogger(name).setLevel(logging.DEBUG if verbose else logging.ERROR)


# =============================
# Domain errors
# =============================
class DeedValidationError(Exception):
    """Base class for deed validation errors."""


class UnknownCountyError(DeedValidationError):
    pass


# =============================
# Reference data
# =============================
def load_counties(path: str = DEFAULT_COUNTIES_FILE) -> List[Dict[str, Any]]:
    """
    Strict loader:
      - file must exist
      - must be JSON list of {name, tax_rate}
      - tax_rate must be numeric
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing counties.json at '{path}'. Put counties.json in the same folder as this script "
            f"(or pass --counties /path/to/counties.json)."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("counties.json must be a JSON list of objects")

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"counties.json row {i} must be an object")
        if "name" not in row or "tax_rate" not in row:
            raise ValueError(f"counties.json row {i} must contain 'name' and 'tax_rate'")
        try:
            float(row["tax_rate"])
        except Exception:
            raise ValueError(f"counties.json row {i} has invalid 'tax_rate': {row['tax_rate']}")

    return data


# =============================
# Date parsing / validation
# =============================
def parse_iso_date(s: str) -> date:
    s = s.strip()
    return datetime.strptime(s, "%Y-%m-%d").date()


def date_reasonableness_error(d: date, field: str) -> Optional[Dict[str, Any]]:
    max_d = date.today() + timedelta(days=DATE_MAX_FUTURE_DAYS)
    if d < DATE_MIN:
        return {
            "code": "DATE_UNREASONABLE",
            "message": f"{field}={d.isoformat()} is before {DATE_MIN.isoformat()}",
            "fields": [field],
        }
    if d > max_d:
        return {
            "code": "DATE_UNREASONABLE",
            "message": f"{field}={d.isoformat()} is too far in the future (>{max_d.isoformat()})",
            "fields": [field],
        }
    return None


# =============================
# Currency parsing
# =============================
_SYMBOL_TO_ISO = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₩": "KRW",
    "₺": "TRY",
    "₽": "RUB",
    "C$": "CAD",
    "A$": "AUD",
}

_WORD_TO_ISO = {
    # dollars
    "DOL": "USD",
    "DOL.": "USD",
    "DOLLAR": "USD",
    "DOLLARS": "USD",
    "USD": "USD",
    "US$": "USD",
    "US DOLLAR": "USD",
    "US DOLLARS": "USD",
    "U.S. DOLLAR": "USD",
    "U.S. DOLLARS": "USD",
    "UNITED STATES DOLLAR": "USD",
    "UNITED STATES DOLLARS": "USD",
    # euros
    "EUR": "EUR",
    "EURO": "EUR",
    "EUROS": "EUR",
    # pounds
    "GBP": "GBP",
    "POUND": "GBP",
    "POUNDS": "GBP",
    "STERLING": "GBP",
    # yen
    "JPY": "JPY",
    "YEN": "JPY",
    # cad/aud
    "CAD": "CAD",
    "CANADIAN DOLLAR": "CAD",
    "CANADIAN DOLLARS": "CAD",
    "AUD": "AUD",
    "AUSTRALIAN DOLLAR": "AUD",
    "AUSTRALIAN DOLLARS": "AUD",
}


def normalize_currency_token(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).upper()

    if s in _SYMBOL_TO_ISO:
        return _SYMBOL_TO_ISO[s]

    s2 = s.replace(",", "")
    if s2 in _SYMBOL_TO_ISO:
        return _SYMBOL_TO_ISO[s2]

    s3 = re.sub(r"[^A-Z\s\$\€\£\¥\₹\₩\₺\₽]", "", s2).strip()
    if s3 in _WORD_TO_ISO:
        return _WORD_TO_ISO[s3]
    if re.fullmatch(r"[A-Z]{3}", s3):
        return s3

    return s3 or s


_money_any_re = re.compile(
    r"""
    ^\s*
    (?:(?P<prefix_ccy>([A-Za-z]{2,15}(\s+[A-Za-z]{2,15})?)|(\$|€|£|¥|₹|₩|₺|₽|C\$|A\$))\s*)?
    (?P<num>[0-9][0-9,]*([.][0-9]+)?)
    (?:\s*(?P<suffix_ccy>([A-Za-z]{2,15}(\s+[A-Za-z]{2,15})?)|(\$|€|£|¥|₹|₩|₺|₽|C\$|A\$)))?
    \s*$
    """,
    re.VERBOSE,
)


def parse_money_digits_any(s: str, extracted_currency_hint: Optional[str] = None) -> Tuple[Decimal, Optional[str]]:
    s = s.strip().replace("\u00a0", " ")
    m = _money_any_re.match(s)
    if not m:
        raise ValueError(f"Unrecognized money format: {s}")

    num = m.group("num").replace(",", "")
    amount = Decimal(num)

    prefix = m.group("prefix_ccy")
    suffix = m.group("suffix_ccy")
    ccy = (
        normalize_currency_token(prefix)
        or normalize_currency_token(suffix)
        or normalize_currency_token(extracted_currency_hint)
    )
    return amount, ccy


def infer_currency_from_words(amount_words: Optional[str]) -> Optional[str]:
    if not amount_words:
        return None
    w = amount_words.lower()
    if "euro" in w:
        return "EUR"
    if "pound" in w or "sterling" in w:
        return "GBP"
    if "yen" in w:
        return "JPY"
    if "rupee" in w:
        return "INR"
    if "dollar" in w or re.search(r"\bdol\b", w):
        return "USD"
    return None


# =============================
# Words-to-number (strict)
# =============================
_SMALL = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_MULT = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}


def words_to_int_us(words: str) -> int:
    w = re.sub(r"[^a-zA-Z\s-]", " ", words).lower()
    w = w.replace("-", " ")
    tokens = [
        t
        for t in w.split()
        if t
        not in {
            "and",
            "dollars",
            "dollar",
            "usd",
            "euros",
            "euro",
            "eur",
            "pounds",
            "pound",
            "gbp",
            "yen",
            "jpy",
            "rupees",
            "rupee",
            "inr",
        }
    ]
    if not tokens:
        raise ValueError("No number words found")

    total = 0
    current = 0
    for t in tokens:
        if t in _SMALL:
            current += _SMALL[t]
        elif t == "hundred":
            if current == 0:
                current = 1
            current *= 100
        elif t in ("thousand", "million", "billion"):
            total += current * _MULT[t]
            current = 0
        else:
            raise ValueError(f"Unknown number token: {t}")
    return total + current


# =============================
# County normalization + matching
# =============================
def normalize_county(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"\s+", " ", s)
    s2 = re.sub(r"[^A-Za-z\s]", "", s).strip()
    s2 = re.sub(r"^St\s+", "Saint ", s2, flags=re.IGNORECASE)
    return s2


def match_county(raw_county: str, counties: List[Dict[str, Any]], min_score: int = COUNTY_MIN_SCORE) -> Tuple[str, float]:
    norm = normalize_county(raw_county)
    choices = [c["name"] for c in counties]
    match = process.extractOne(norm, choices, scorer=fuzz.WRatio)
    if not match:
        raise UnknownCountyError(f"County '{raw_county}' could not be matched")
    name, score, _idx = match
    if score < min_score:
        raise UnknownCountyError(f"County '{raw_county}' matched '{name}' with low score={score}")
    return name, float(score)


# =============================
# Presence detection: ABSENT vs PRESENT_EMPTY vs PRESENT_WITH_VALUE
# =============================
# Line-by-line label parsing (avoids regex multiline/$ edge cases across platforms)
_LABEL_PREFIXES = [
    ("doc_id", r"^\s*Doc\s*:\s*", None),
    ("county_line", r"^\s*County\s*:\s*", None),
    ("date_signed", r"^\s*Date\s*Signed\s*:\s*", None),
    ("date_recorded", r"^\s*Date\s*Recorded\s*:\s*", None),
    ("grantor", r"^\s*Grantor\s*:\s*", None),
    ("grantee", r"^\s*Grantee\s*:\s*", None),
    ("amount_line", r"^\s*Amount\s*:\s*", None),
    ("apn", r"^\s*APN\s*:\s*", None),
    ("status", r"^\s*Status\s*:\s*", None),
]
_LABEL_RE = {k: re.compile(pat, re.I) for k, pat, _ in _LABEL_PREFIXES}


def _get_label_value(raw_text: str, label_key: str) -> Tuple[str, Optional[str]]:
    """Get (status, raw_value) for a label. Line-by-line to avoid cross-line capture."""
    pat = _LABEL_RE.get(label_key)
    if not pat:
        return ABSENT, None
    for line in raw_text.splitlines():
        m = pat.match(line)
        if m:
            val = line[m.end() :].strip()
            if val == "":
                return PRESENT_EMPTY, ""
            return PRESENT_WITH_VALUE, val
    return ABSENT, None


def _presence_from_match(m: Optional[re.Match]) -> Tuple[str, Optional[str]]:
    if not m:
        return ABSENT, None
    raw = (m.group("val") or "")
    if raw.strip() == "":
        return PRESENT_EMPTY, ""
    return PRESENT_WITH_VALUE, raw.strip()


def _split_county_state_from_line(raw_line: str) -> Tuple[Optional[str], Optional[str]]:
    line = raw_line.strip()
    m = re.search(r"(?i)\bState\s*:\s*([A-Z]{2})\b", line)
    state = m.group(1).strip().upper() if m else None

    county_part = line
    if "|" in county_part:
        county_part = county_part.split("|", 1)[0]
    county_part = re.sub(r"(?i)\bState\s*:\s*[A-Z]{2}\b", "", county_part).strip()
    county_part = county_part if county_part.strip() else None
    return county_part, state


def _extract_amount_components(amount_val: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    s = amount_val.strip()
    m_words = re.search(r"\((.*?)\)", s)
    words = m_words.group(1).strip() if m_words and m_words.group(1).strip() else None

    before = s.split("(", 1)[0].strip() if "(" in s else s.strip()
    digits = before if before else None

    currency = None
    if digits:
        try:
            _amt, ccy = parse_money_digits_any(digits, extracted_currency_hint=None)
            currency = ccy
        except Exception:
            currency = normalize_currency_token(digits)

    return digits, words, currency


def detect_field_presence(raw_text: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    st, val = _get_label_value(raw_text, "doc_id")
    out["doc_id"] = {"status": st, "raw_value": val}

    county_line_status, county_line_val = _get_label_value(raw_text, "county_line")
    county_val = None
    state_val = None
    if county_line_status == PRESENT_WITH_VALUE and county_line_val is not None:
        county_val, state_val = _split_county_state_from_line(county_line_val)

    if county_line_status == ABSENT:
        out["county"] = {"status": ABSENT, "raw_value": None}
        out["state"] = {"status": ABSENT, "raw_value": None}
    elif county_line_status == PRESENT_EMPTY:
        out["county"] = {"status": PRESENT_EMPTY, "raw_value": ""}
        out["state"] = {"status": PRESENT_EMPTY, "raw_value": ""}
    else:
        out["county"] = {"status": PRESENT_WITH_VALUE if (county_val and county_val.strip()) else PRESENT_EMPTY, "raw_value": county_val or ""}
        out["state"] = {"status": PRESENT_WITH_VALUE if (state_val and state_val.strip()) else PRESENT_EMPTY, "raw_value": state_val or ""}

    for k in ["date_signed", "date_recorded", "grantor", "grantee", "apn", "status"]:
        st, val = _get_label_value(raw_text, k)
        out[k] = {"status": st, "raw_value": val}

    amt_line_status, amt_line_val = _get_label_value(raw_text, "amount_line")

    if amt_line_status == ABSENT:
        out["amount_digits"] = {"status": ABSENT, "raw_value": None}
        out["amount_words"] = {"status": ABSENT, "raw_value": None}
        out["currency"] = {"status": ABSENT, "raw_value": None}
    elif amt_line_status == PRESENT_EMPTY:
        out["amount_digits"] = {"status": PRESENT_EMPTY, "raw_value": ""}
        out["amount_words"] = {"status": PRESENT_EMPTY, "raw_value": ""}
        out["currency"] = {"status": PRESENT_EMPTY, "raw_value": ""}
    else:
        digits, words, currency = _extract_amount_components(amt_line_val or "")
        out["amount_digits"] = {"status": PRESENT_WITH_VALUE if digits else PRESENT_EMPTY, "raw_value": digits or ""}
        out["amount_words"] = {"status": PRESENT_WITH_VALUE if words else PRESENT_EMPTY, "raw_value": words or ""}
        out["currency"] = {"status": PRESENT_WITH_VALUE if currency else PRESENT_EMPTY, "raw_value": currency or ""}

    for f in KNOWN_FIELDS:
        out.setdefault(f, {"status": ABSENT, "raw_value": None})

    return out


# =============================
# Extraction schema with spans
# =============================
@dataclass
class Span:
    text: Optional[str]
    start: Optional[int]
    end: Optional[int]


@dataclass
class DeedExtracted:
    doc_id: Optional[Span]
    county_raw: Optional[Span]
    state: Optional[Span]
    date_signed: Optional[Span]
    date_recorded: Optional[Span]
    grantor: Optional[Span]
    grantee: Optional[Span]
    amount_digits: Optional[Span]
    amount_words: Optional[Span]
    currency: Optional[Span]
    apn: Optional[Span]
    status: Optional[Span]
    extractions: List[Dict[str, Any]]
    unknown_extractions: List[Dict[str, Any]]


def _flatten_extractions(result: Any) -> List[Dict[str, Any]]:
    extracted_items = getattr(result, "extractions", None) or getattr(result, "data", None) or result
    flat: List[Dict[str, Any]] = []
    if isinstance(extracted_items, list):
        for e in extracted_items:
            ci = getattr(e, "char_interval", None)
            flat.append(
                {
                    "class": getattr(e, "extraction_class", None),
                    "text": getattr(e, "extraction_text", None),
                    "start": getattr(ci, "start_pos", None) if ci else None,
                    "end": getattr(ci, "end_pos", None) if ci else None,
                    "alignment_status": getattr(e, "alignment_status", None).value
                    if getattr(e, "alignment_status", None)
                    else None,
                    "attributes": getattr(e, "attributes", {}) or {},
                }
            )
    return flat


def _get_first_span(flat: List[Dict[str, Any]], cls: str) -> Optional[Span]:
    for item in flat:
        if item["class"] == cls:
            return Span(text=item["text"], start=item["start"], end=item["end"])
    return None


def _api_key() -> Optional[str]:
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("openAI_API_key") or os.environ.get("LANGEXTRACT_API_KEY")


def langextract_deed(text: str, model_id: str = DEFAULT_MODEL_ID) -> DeedExtracted:
    prompt = (
        "Extract deed fields from the text.\n"
        "Rules:\n"
        "- Use exact text spans; do NOT paraphrase.\n"
        "- Only extract fields that have actual values. Do NOT extract empty or missing fields.\n"
        "- If a field label exists but has no value (e.g., 'County: ' with nothing after), skip it entirely.\n"
        "- Amount must be split into: amount_digits and amount_words.\n"
        "- Extract currency if present (ISO code, symbol, or words like Dollar/Dollars/Dol.).\n"
        "- Dates must be extracted exactly as written.\n"
        "Fields: doc_id, county, state, date_signed, date_recorded, grantor, grantee, "
        "amount_digits, amount_words, currency, apn, status."
    )

    examples = [
        lx.data.ExampleData(
            text=(
                "*** RECORDING REQ ***\n"
                "Doc: DEED-0001\n"
                "County: San Mateo | State: CA\n"
                "Date Signed: 2024-01-01\n"
                "Date Recorded: 2024-01-02\n"
                "Grantor: Example LLC\n"
                "Grantee: Alice Example\n"
                "Amount: 100,000.00 Dollars (One Hundred Thousand Dollars)\n"
                "APN: 111-222-XYZ\n"
                "Status: FINAL\n"
                "*** END ***\n"
            ),
            extractions=[
                lx.data.Extraction("doc_id", "DEED-0001", attributes={}),
                lx.data.Extraction("county", "San Mateo", attributes={}),
                lx.data.Extraction("state", "CA", attributes={}),
                lx.data.Extraction("date_signed", "2024-01-01", attributes={}),
                lx.data.Extraction("date_recorded", "2024-01-02", attributes={}),
                lx.data.Extraction("grantor", "Example LLC", attributes={}),
                lx.data.Extraction("grantee", "Alice Example", attributes={}),
                lx.data.Extraction("amount_digits", "100,000.00 Dollars", attributes={}),
                lx.data.Extraction("amount_words", "One Hundred Thousand Dollars", attributes={}),
                lx.data.Extraction("currency", "Dollars", attributes={}),
                lx.data.Extraction("apn", "111-222-XYZ", attributes={}),
                lx.data.Extraction("status", "FINAL", attributes={}),
            ],
        ),
        lx.data.ExampleData(
            text=(
                "*** RECORDING REQ ***\n"
                "Doc: DEED-0002\n"
                "County: \n"
                "Date Signed: \n"
                "Date Recorded: 2024-01-10\n"
                "Grantor: Test LLC\n"
                "Grantee: Test Person\n"
                "Amount: $500,000.00 (Five Hundred Thousand Dollars)\n"
                "APN: 123-456-AB\n"
                "Status: PRELIMINARY\n"
                "*** END ***\n"
            ),
            extractions=[
                lx.data.Extraction("doc_id", "DEED-0002", attributes={}),
                lx.data.Extraction("date_recorded", "2024-01-10", attributes={}),
                lx.data.Extraction("grantor", "Test LLC", attributes={}),
                lx.data.Extraction("grantee", "Test Person", attributes={}),
                lx.data.Extraction("amount_digits", "$500,000.00", attributes={}),
                lx.data.Extraction("amount_words", "Five Hundred Thousand Dollars", attributes={}),
                lx.data.Extraction("apn", "123-456-AB", attributes={}),
                lx.data.Extraction("status", "PRELIMINARY", attributes={}),
            ],
        ),
    ]

    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        api_key=_api_key(),
        fence_output=True,
        use_schema_constraints=False,
    )

    flat = _flatten_extractions(result)
    unknown = [x for x in flat if x.get("class") and x["class"] not in KNOWN_FIELDS]

    return DeedExtracted(
        doc_id=_get_first_span(flat, "doc_id"),
        county_raw=_get_first_span(flat, "county"),
        state=_get_first_span(flat, "state"),
        date_signed=_get_first_span(flat, "date_signed"),
        date_recorded=_get_first_span(flat, "date_recorded"),
        grantor=_get_first_span(flat, "grantor"),
        grantee=_get_first_span(flat, "grantee"),
        amount_digits=_get_first_span(flat, "amount_digits"),
        amount_words=_get_first_span(flat, "amount_words"),
        currency=_get_first_span(flat, "currency"),
        apn=_get_first_span(flat, "apn"),
        status=_get_first_span(flat, "status"),
        extractions=flat,
        unknown_extractions=unknown,
    )


# =============================
# Loopback passes (candidates)
# =============================
def langextract_candidate_dates(text: str, model_id: str = DEFAULT_MODEL_ID) -> List[Dict[str, Any]]:
    prompt = (
        "Return ALL candidate date mentions in the text.\n"
        "Rules:\n"
        "- Use exact text spans only.\n"
        "- Extraction class: date_mention.\n"
        "- Attributes must include: kind in {signed, recorded, other}, and raw_line_hint.\n"
        "- If you are unsure of kind, use 'other'.\n"
    )
    examples = [
        lx.data.ExampleData(
            text="Date Signed: 2024-01-01\nDate Recorded: 2024-01-02\n",
            extractions=[
                lx.data.Extraction("date_mention", "2024-01-01", attributes={"kind": "signed", "raw_line_hint": "Date Signed"}),
                lx.data.Extraction("date_mention", "2024-01-02", attributes={"kind": "recorded", "raw_line_hint": "Date Recorded"}),
            ],
        )
    ]
    res = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        api_key=_api_key(),
        fence_output=True,
        use_schema_constraints=False,
    )
    return _flatten_extractions(res)


def langextract_candidate_amounts(text: str, model_id: str = DEFAULT_MODEL_ID) -> List[Dict[str, Any]]:
    prompt = (
        "Return ALL candidate amount mentions in the text.\n"
        "Rules:\n"
        "- Use exact text spans only.\n"
        "- Extraction class must be one of: amount_digits, amount_words, currency.\n"
        "- Capture currency tokens even if they appear after the number (e.g., '1000 Dollars').\n"
    )
    examples = [
        lx.data.ExampleData(
            text="Amount: 100,000.00 EUR (One Hundred Thousand Euros)",
            extractions=[
                lx.data.Extraction("amount_digits", "100,000.00 EUR", attributes={}),
                lx.data.Extraction("currency", "EUR", attributes={}),
                lx.data.Extraction("amount_words", "One Hundred Thousand Euros", attributes={}),
            ],
        )
    ]
    res = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        api_key=_api_key(),
        fence_output=True,
        use_schema_constraints=False,
    )
    return _flatten_extractions(res)


def langextract_candidate_counties(text: str, model_id: str = DEFAULT_MODEL_ID) -> List[Dict[str, Any]]:
    prompt = (
        "Return ALL county-like mentions in the text.\n"
        "Rules:\n"
        "- Use exact text spans only.\n"
        "- Extraction class: county_mention.\n"
        "- Attributes should include: normalized_candidate and raw_line_hint.\n"
        "- normalized_candidate must be derived from the text (abbreviation expansion), not invented.\n"
    )
    examples = [
        lx.data.ExampleData(
            text="County: S. Clara | State: CA\n",
            extractions=[
                lx.data.Extraction(
                    "county_mention",
                    "S. Clara",
                    attributes={"normalized_candidate": "Santa Clara", "raw_line_hint": "County"},
                ),
            ],
        )
    ]
    res = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        api_key=_api_key(),
        fence_output=True,
        use_schema_constraints=False,
    )
    return _flatten_extractions(res)


# =============================
# Loopback analysis (ambiguity)
# =============================
def analyze_dates_ambiguity(
    candidate_dates: List[Dict[str, Any]],
    primary_signed: Optional[str],
    primary_recorded: Optional[str],
) -> Optional[Dict[str, Any]]:
    parsed: List[Tuple[str, str]] = []
    for item in candidate_dates:
        if item.get("class") != "date_mention":
            continue
        txt = item.get("text")
        if not txt:
            continue
        try:
            d = parse_iso_date(txt).isoformat()
        except Exception:
            continue
        kind = (item.get("attributes") or {}).get("kind") or "other"
        kind = kind.lower().strip()
        if kind not in {"signed", "recorded", "other"}:
            kind = "other"
        parsed.append((kind, d))

    if not parsed:
        return None

    by_kind: Dict[str, set] = {"signed": set(), "recorded": set(), "other": set()}
    for kind, d in parsed:
        by_kind[kind].add(d)

    if len(by_kind["signed"]) > 1:
        return {"signed": sorted(by_kind["signed"])}
    if len(by_kind["recorded"]) > 1:
        return {"recorded": sorted(by_kind["recorded"])}

    expected = set()
    if primary_signed:
        try:
            expected.add(parse_iso_date(primary_signed).isoformat())
        except Exception:
            pass
    if primary_recorded:
        try:
            expected.add(parse_iso_date(primary_recorded).isoformat())
        except Exception:
            pass

    extra_other = sorted([d for d in by_kind["other"] if d not in expected])
    if extra_other:
        return {"other": extra_other}
    return None


def analyze_amounts_ambiguity(candidate_amounts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    uniq = set()
    for item in candidate_amounts:
        if item.get("class") != "amount_digits":
            continue
        txt = item.get("text")
        if not txt:
            continue
        try:
            amt, ccy = parse_money_digits_any(txt, extracted_currency_hint=None)
            uniq.add((str(amt), ccy or ""))
        except Exception:
            continue

    if len(uniq) > 1:
        return {"candidates": [{"amount": a, "currency": c or None} for (a, c) in sorted(uniq)]}
    return None


def analyze_currency_mentions(candidate_amounts: List[Dict[str, Any]]) -> List[str]:
    ccys = set()
    for item in candidate_amounts:
        if item.get("class") != "currency":
            continue
        txt = item.get("text")
        if not txt:
            continue
        c = normalize_currency_token(txt)
        if c:
            ccys.add(c)
    return sorted(ccys)


# =============================
# Helpers: output formatting
# =============================
def compact_field(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    return v if v else None


def make_failure(
    code: str,
    summary: str,
    details: Optional[str] = None,
    fields: Optional[List[str]] = None,
    evidence: Optional[Dict[str, Any]] = None,
    severity: str = "error",
) -> Dict[str, Any]:
    f: Dict[str, Any] = {"code": code, "severity": severity, "summary": summary}
    if details:
        f["details"] = details
    if fields:
        f["fields"] = fields
    if evidence:
        f["evidence"] = evidence
    return f


def group_failures(failures: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = {"presence": [], "dates": [], "amounts": [], "identity": [], "schema": [], "other": []}
    for f in failures:
        code = f.get("code", "") or ""
        if code.startswith("FIELD_"):
            groups["presence"].append(f)
        elif code.startswith("DATE_") or code in {"AMBIGUOUS_DATE"}:
            groups["dates"].append(f)
        elif code.startswith("AMOUNT_") or code.startswith("CURRENCY_") or code in {"AMBIGUOUS_AMOUNT", "AMBIGUOUS_CURRENCY"}:
            groups["amounts"].append(f)
        elif code.startswith("UNKNOWN_COUNTY") or code.startswith("COUNTY_"):
            groups["identity"].append(f)
        elif code.startswith("COUNTIES_") or code.startswith("UNKNOWN_FIELD"):
            groups["schema"].append(f)
        else:
            groups["other"].append(f)
    return {k: v for k, v in groups.items() if v}


def extracted_value(s: Optional["Span"]) -> Optional[str]:
    return compact_field(s.text) if (s and s.text) else None

def is_blank_text(s: Optional[str]) -> bool:
    return s is None or s.strip() == ""


def extracted_blank(span: Optional["Span"]) -> bool:
    if span is None:
        return True
    return is_blank_text(span.text)

def extracted_span_for_field(deed: "DeedExtracted", field: str) -> Optional["Span"]:
    # Required fields are in "normalized/derived" names (county, amount_digits, etc.)
    # Map them to the actual extracted spans produced by langextract_deed().
    mapping = {
        "doc_id": deed.doc_id,
        "county": deed.county_raw,          # extracted as "county" → stored in county_raw
        "state": deed.state,
        "date_signed": deed.date_signed,
        "date_recorded": deed.date_recorded,
        "grantor": deed.grantor,
        "grantee": deed.grantee,
        "amount_digits": deed.amount_digits,
        "amount_words": deed.amount_words,
        "apn": deed.apn,
        # optional ones
        "status": deed.status,
        "currency": deed.currency,
    }
    return mapping.get(field)

def unknown_values(unknown: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for u in unknown:
        cls = u.get("class") or "UNKNOWN"
        txt = (u.get("text") or "").strip()
        if not txt:
            continue
        out.setdefault(cls, []).append(txt)
    return {k: sorted(list(set(v))) for k, v in out.items()}


# =============================
# Validation + enrichment
# =============================
def validate_and_enrich(
    raw_text: str,
    deed: "DeedExtracted",
    counties: List[Dict[str, Any]],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    include_audit: bool = DEFAULT_INCLUDE_AUDIT,
) -> Dict[str, Any]:
    """
    Fail-closed validation pipeline.

    Major checks:
      1) Presence (ABSENT / PRESENT_EMPTY / PRESENT_WITH_VALUE) for required fields -> hard fail
      2) Extraction completeness: if raw OCR has value but primary extraction returned blank/None -> hard fail
      3) Schema drift: unknown extraction classes -> warning (configurable)
      4) County enrichment + verification + expansion-not-in-reference -> hard fail
      5) Date parsing/reasonableness/order -> hard fail
      6) Amount parsing + words parsing + reconciliation -> hard fail
      7) Loopback ambiguity checks -> warnings; currency mismatch -> hard fail
    """
    failures: List[Dict[str, Any]] = []

    # -----------------------------
    # Presence check (raw OCR)
    # -----------------------------
    presence = detect_field_presence(raw_text)

    for field in sorted(REQUIRED_FIELDS):
        st = presence.get(field, {}).get("status", ABSENT)

        if st == ABSENT:
            failures.append(
                make_failure(
                    code="FIELD_ABSENT",
                    summary=f"Required field is absent: {field}",
                    details="Label not found in raw OCR text (fail-closed).",
                    fields=[field],
                    evidence={"field": field, "presence": st} if include_audit else {"field": field},
                    severity="error",
                )
            )
        elif st == PRESENT_EMPTY:
            failures.append(
                make_failure(
                    code="FIELD_PRESENT_EMPTY",
                    summary=f"Required field is present but empty: {field}",
                    details="Label found but value missing/blank (fail-closed).",
                    fields=[field],
                    evidence={"field": field, "presence": st} if include_audit else {"field": field},
                    severity="error",
                )
            )

    # -----------------------------
    # Extraction completeness (fail-closed)
    # If OCR has value, but extractor returned blank/None -> error
    # -----------------------------
    def _is_blank_text(s: Optional[str]) -> bool:
        return s is None or s.strip() == ""

    def _span_for_field(f: str) -> Optional["Span"]:
        mapping = {
            "doc_id": deed.doc_id,
            "county": deed.county_raw,  # stored as county_raw span
            "state": deed.state,
            "date_signed": deed.date_signed,
            "date_recorded": deed.date_recorded,
            "grantor": deed.grantor,
            "grantee": deed.grantee,
            "amount_digits": deed.amount_digits,
            "amount_words": deed.amount_words,
            "apn": deed.apn,
            "status": deed.status,
            "currency": deed.currency,
        }
        return mapping.get(f)

    for field in sorted(REQUIRED_FIELDS):
        st = presence.get(field, {}).get("status", ABSENT)
        if st != PRESENT_WITH_VALUE:
            continue  # already failed or not present with value

        sp = _span_for_field(field)
        extracted_txt = None if sp is None else sp.text
        if _is_blank_text(extracted_txt):
            failures.append(
                make_failure(
                    code="EXTRACTED_VALUE_MISSING",
                    summary=f"Extractor failed to extract required field with value: {field}",
                    details="Raw OCR has a value for this field, but primary extraction returned blank/None (fail-closed).",
                    fields=[field],
                    evidence=(
                        {
                            "field": field,
                            "presence": st,
                            "raw_value": presence.get(field, {}).get("raw_value"),
                            "extracted_text": extracted_txt,
                        }
                        if include_audit
                        else {"field": field}
                    ),
                    severity="error",
                )
            )

    # -----------------------------
    # Schema drift / unknown fields (LLM)
    # -----------------------------
    unk_vals = unknown_values(deed.unknown_extractions)
    if FAIL_CLOSED_ON_UNKNOWN_FIELDS and deed.unknown_extractions:
        failures.append(
            make_failure(
                code="UNKNOWN_FIELD_EXTRACTED",
                severity="warning",  # flip to "error" if you want hard fail
                summary="Extractor returned unexpected field(s).",
                details="Fail-closed policy: new fields must be reviewed before accepting.",
                evidence={"unknown_fields": sorted(list(unk_vals.keys()))},
            )
        )

    # -----------------------------
    # County enrichment (+ verifier)
    # -----------------------------
    county_raw = extracted_value(deed.county_raw)
    county_name: Optional[str] = None
    county_score: Optional[float] = None
    tax_rate: Optional[float] = None

    # Track expansion attempt (audit)
    expansion_attempted = False
    expansion_candidates: List[str] = []
    expansion_best_candidate: Optional[str] = None

    # Try direct match first
    if county_raw:
        try:
            county_name, county_score = match_county(county_raw, counties)
            tax_rate = float(next(c["tax_rate"] for c in counties if c["name"] == county_name))
        except UnknownCountyError:
            county_name, county_score = None, None

    needs_county_retry = county_name is None or (
        county_score is not None and county_score < COUNTY_MIN_SCORE + COUNTY_RETRY_MARGIN
    )

    county_loopback_candidates = None
    if needs_county_retry:
        try:
            county_loopback_candidates = langextract_candidate_counties(raw_text, model_id=model_id)
            expansion_attempted = True

            best = (county_name, county_score, tax_rate)

            for cand in county_loopback_candidates:
                if cand.get("class") != "county_mention":
                    continue
                attrs = cand.get("attributes") or {}
                normcand = attrs.get("normalized_candidate")
                rawcand = cand.get("text")

                for _source, option in [("normalized_candidate", normcand), ("raw_text", rawcand)]:
                    if not option:
                        continue
                    if option not in expansion_candidates:
                        expansion_candidates.append(option)

                    try:
                        name2, score2 = match_county(option, counties)
                        if score2 >= COUNTY_MIN_SCORE and (best[1] is None or score2 > best[1]):
                            rate2 = float(next(c["tax_rate"] for c in counties if c["name"] == name2))
                            best = (name2, score2, rate2)
                            expansion_best_candidate = option
                    except UnknownCountyError:
                        pass

            county_name, county_score, tax_rate = best

            # A+B: if normalized_candidate was present but nothing matched -> explicit error
            normalized_present = any(
                (c.get("attributes") or {}).get("normalized_candidate") for c in (county_loopback_candidates or [])
            )
            if county_name is None and normalized_present:
                failures.append(
                    make_failure(
                        code="COUNTY_EXPANSION_NOT_IN_REFERENCE",
                        summary="County expansion candidate(s) were not found in counties.json.",
                        details="LLM suggested an expanded/normalized county name, but no match exists in reference data (fail-closed).",
                        fields=["county"],
                        evidence=(
                            {"county_raw": county_raw, "expansion_candidates": sorted(expansion_candidates)}
                            if include_audit
                            else {"county_raw": county_raw}
                        ),
                        severity="error",
                    )
                )

        except Exception as e:
            failures.append(
                make_failure(
                    code="COUNTY_LOOPBACK_FAILED",
                    summary="County verification pass failed.",
                    details=str(e),
                    fields=["county"],
                    severity="warning",
                )
            )

    if county_name is None and not any(f.get("code") == "COUNTY_EXPANSION_NOT_IN_REFERENCE" for f in failures):
        failures.append(
            make_failure(
                code="UNKNOWN_COUNTY",
                summary="County could not be matched to reference list.",
                details="Cannot compute tax_rate without a verified county.",
                fields=["county"],
                evidence={"county_raw": county_raw} if include_audit else {"county_raw": county_raw},
                severity="error",
            )
        )

    # -----------------------------
    # Dates
    # -----------------------------
    date_signed_raw = extracted_value(deed.date_signed)
    date_recorded_raw = extracted_value(deed.date_recorded)

    signed: Optional[date] = None
    recorded: Optional[date] = None

    if date_signed_raw:
        try:
            signed = parse_iso_date(date_signed_raw)
            maybe = date_reasonableness_error(signed, "date_signed")
            if maybe:
                failures.append(
                    make_failure(
                        code=maybe["code"],
                        summary="Signed date is not reasonable.",
                        details=maybe["message"],
                        fields=maybe.get("fields") or ["date_signed"],
                    )
                )
        except Exception as e:
            failures.append(
                make_failure(
                    code="DATE_PARSE_ERROR",
                    summary="Signed date could not be parsed.",
                    details=str(e),
                    fields=["date_signed"],
                    evidence={"date_signed_raw": date_signed_raw},
                )
            )

    if date_recorded_raw:
        try:
            recorded = parse_iso_date(date_recorded_raw)
            maybe = date_reasonableness_error(recorded, "date_recorded")
            if maybe:
                failures.append(
                    make_failure(
                        code=maybe["code"],
                        summary="Recorded date is not reasonable.",
                        details=maybe["message"],
                        fields=maybe.get("fields") or ["date_recorded"],
                    )
                )
        except Exception as e:
            failures.append(
                make_failure(
                    code="DATE_PARSE_ERROR",
                    summary="Recorded date could not be parsed.",
                    details=str(e),
                    fields=["date_recorded"],
                    evidence={"date_recorded_raw": date_recorded_raw},
                )
            )

    if signed and recorded and recorded < signed:
        failures.append(
            make_failure(
                code="DATE_ORDER_ERROR",
                summary="Recorded date occurs before signed date (impossible).",
                details="This suggests OCR/LLM extraction error or fraudulent / inconsistent document.",
                fields=["date_recorded", "date_signed"],
                evidence={"date_signed": signed.isoformat(), "date_recorded": recorded.isoformat()},
            )
        )

    # -----------------------------
    # Amounts + currency
    # -----------------------------
    amount_digits_raw = extracted_value(deed.amount_digits)
    amount_words_raw = extracted_value(deed.amount_words)
    extracted_currency_hint = extracted_value(deed.currency)

    # Fallback: when LLM did not extract amount_digits but presence layer found a value, use it for parse attempt
    if not amount_digits_raw:
        presence_digits = presence.get("amount_digits", {}).get("raw_value")
        if presence_digits and presence.get("amount_digits", {}).get("status") == PRESENT_WITH_VALUE:
            amount_digits_raw = presence_digits

    amt_digits: Optional[Decimal] = None
    amt_digits_ccy: Optional[str] = None
    amt_words: Optional[Decimal] = None

    if amount_digits_raw:
        try:
            amt_digits, amt_digits_ccy = parse_money_digits_any(
                amount_digits_raw, extracted_currency_hint=extracted_currency_hint
            )
        except (InvalidOperation, ValueError) as e:
            failures.append(
                make_failure(
                    code="AMOUNT_DIGITS_PARSE_ERROR",
                    summary="Digit amount could not be parsed.",
                    details=str(e),
                    fields=["amount_digits"],
                    evidence={"amount_digits_raw": amount_digits_raw},
                )
            )

    digit_ccy = amt_digits_ccy or normalize_currency_token(extracted_currency_hint)

    if amount_words_raw:
        try:
            amt_words = Decimal(words_to_int_us(amount_words_raw))
        except ValueError as e:
            failures.append(
                make_failure(
                    code="AMOUNT_WORDS_PARSE_ERROR",
                    summary="Written amount could not be parsed deterministically.",
                    details=str(e),
                    fields=["amount_words"],
                    evidence={"amount_words_raw": amount_words_raw},
                )
            )

    if amt_digits is not None and amt_words is not None and amt_digits != amt_words:
        diff = amt_digits - amt_words
        failures.append(
            make_failure(
                code="AMOUNT_MISMATCH_ERROR",
                summary="Digit amount does not match written amount.",
                details="System cannot safely choose which one is correct without review.",
                fields=["amount_digits", "amount_words"],
                evidence={"digits": str(amt_digits), "words": str(amt_words), "difference": str(diff)},
            )
        )

    # -----------------------------
    # Loopback (ambiguity + currency mismatch)
    # -----------------------------
    loopback = None
    loopback_analysis = None
    if ALWAYS_RUN_LOOPBACK:
        try:
            candidate_dates = langextract_candidate_dates(raw_text, model_id=model_id)
            candidate_amounts = langextract_candidate_amounts(raw_text, model_id=model_id)
            loopback = {"candidate_dates": candidate_dates, "candidate_amounts": candidate_amounts}

            date_amb = analyze_dates_ambiguity(
                candidate_dates, primary_signed=date_signed_raw, primary_recorded=date_recorded_raw
            )
            amt_amb = analyze_amounts_ambiguity(candidate_amounts)
            ccys = analyze_currency_mentions(candidate_amounts)

            loopback_analysis = {"date_ambiguity": date_amb, "amount_ambiguity": amt_amb, "currency_mentions": ccys}

            if date_amb:
                failures.append(
                    make_failure(
                        code="AMBIGUOUS_DATE",
                        severity="warning",
                        summary="Multiple plausible date candidates detected.",
                        details="Requires review to confirm which dates are authoritative.",
                        fields=["date_signed", "date_recorded"],
                        evidence=date_amb,
                    )
                )

            if amt_amb:
                failures.append(
                    make_failure(
                        code="AMBIGUOUS_AMOUNT",
                        severity="warning",
                        summary="Multiple digit-amount candidates detected.",
                        details="Requires review to confirm which amount is authoritative.",
                        fields=["amount_digits"],
                        evidence=amt_amb,
                    )
                )

            words_ccy = infer_currency_from_words(amount_words_raw)
            words_ccy_final = words_ccy or (ccys[0] if len(ccys) == 1 else None)

            if digit_ccy and words_ccy_final and digit_ccy != words_ccy_final:
                failures.append(
                    make_failure(
                        code="CURRENCY_MISMATCH",
                        summary="Currency inconsistency between digits and words.",
                        details="Potential OCR error or tampering; requires review.",
                        fields=["amount_digits", "amount_words", "currency"],
                        evidence={"digit_currency": digit_ccy, "words_currency": words_ccy_final, "mentions": ccys},
                    )
                )

            if len(ccys) > 1:
                failures.append(
                    make_failure(
                        code="AMBIGUOUS_CURRENCY",
                        severity="warning",
                        summary="Multiple different currency mentions detected.",
                        details="Requires review to confirm currency.",
                        fields=["currency"],
                        evidence={"mentions": ccys},
                    )
                )

        except Exception as e:
            failures.append(
                make_failure(
                    code="LOOPBACK_FAILED",
                    severity="warning",
                    summary="Loopback verification failed.",
                    details=str(e),
                )
            )

    # -----------------------------
    # Final output
    # -----------------------------
    passed = all(f.get("severity") != "error" for f in failures)

    extracted_fields: Dict[str, Any] = {
        "doc_id": extracted_value(deed.doc_id),
        "county": county_name,
        "county_raw": county_raw if (county_raw and county_name and county_raw.lower() != county_name.lower()) else None,
        "state": extracted_value(deed.state),
        "date_signed": date_signed_raw,
        "date_recorded": date_recorded_raw,
        "grantor": extracted_value(deed.grantor),
        "grantee": extracted_value(deed.grantee),
        "amount_digits": amount_digits_raw,
        "amount_words": amount_words_raw,
        "currency": digit_ccy,
        "apn": extracted_value(deed.apn),
        "status": extracted_value(deed.status),
        "tax_rate": tax_rate,
    }
    extracted_fields = {k: v for k, v in extracted_fields.items() if v is not None}

    out: Dict[str, Any] = {
        "status": "PASS" if passed else "FAIL",
        "failures": group_failures(failures) if not passed else {},
        "extracted": extracted_fields,
        "unknown_fields": unk_vals,
    }

    if include_audit:
        out["audit"] = {
            "presence": {k: {"status": v["status"], "raw_value": v.get("raw_value")} for k, v in presence.items()},
            "county_resolution": {
                "expansion_attempted": expansion_attempted,
                "expansion_candidates": sorted(expansion_candidates),
                "expansion_best_candidate": expansion_best_candidate,
                "matched_county": county_name,
                "match_score": county_score,
            },
            "spans": {
                "doc_id": {"value": deed.doc_id.text, "start": deed.doc_id.start, "end": deed.doc_id.end}
                if deed.doc_id
                else None,
                "county_raw": {"value": deed.county_raw.text, "start": deed.county_raw.start, "end": deed.county_raw.end}
                if deed.county_raw
                else None,
                "date_signed": {"value": deed.date_signed.text, "start": deed.date_signed.start, "end": deed.date_signed.end}
                if deed.date_signed
                else None,
                "date_recorded": {
                    "value": deed.date_recorded.text,
                    "start": deed.date_recorded.start,
                    "end": deed.date_recorded.end,
                }
                if deed.date_recorded
                else None,
                "amount_digits": {
                    "value": deed.amount_digits.text,
                    "start": deed.amount_digits.start,
                    "end": deed.amount_digits.end,
                }
                if deed.amount_digits
                else None,
                "amount_words": {
                    "value": deed.amount_words.text,
                    "start": deed.amount_words.start,
                    "end": deed.amount_words.end,
                }
                if deed.amount_words
                else None,
                "currency": {"value": deed.currency.text, "start": deed.currency.start, "end": deed.currency.end}
                if deed.currency
                else None,
            },
            "langextract_raw_extractions": deed.extractions,
            "county_loopback_candidates": county_loopback_candidates,
            "loopback_analysis": loopback_analysis,
            "loopback_raw": loopback,
        }

    return out


# =============================
# Input parsing: default text or .txt file with 1+ deed blocks
# =============================
_BLOCK_RE = re.compile(r"\*\*\*\s*RECORDING\s+REQ\s*\*\*\*(.*?)\*\*\*\s*END\s*\*\*\*", re.IGNORECASE | re.DOTALL)


def extract_blocks_from_text(all_text: str) -> List[str]:
    matches = _BLOCK_RE.findall(all_text)
    blocks: List[str] = []
    for m in matches:
        block = "*** RECORDING REQ ***" + m + "*** END ***"
        block = block.replace("\r\n", "\n").strip()
        blocks.append(block)
    return blocks


def read_input_blocks(text: Optional[str], input_file: Optional[str]) -> List[str]:
    if text and text.strip():
        blocks = extract_blocks_from_text(text)
        return blocks if blocks else [text.strip()]

    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = extract_blocks_from_text(content)
        return blocks if blocks else [content.strip()]

    blocks = extract_blocks_from_text(DEFAULT_RAW_TEXT)
    return blocks if blocks else [DEFAULT_RAW_TEXT.strip()]


# =============================
# Runner
# =============================
def run_pipeline_on_blocks(
    blocks: List[str],
    counties_path: str,
    model_id: str,
    include_audit: bool,
) -> Dict[str, Any]:
    # A: structured failure for missing/invalid counties.json
    try:
        counties = load_counties(counties_path)
    except FileNotFoundError as e:
        return {
            "status": "FAIL",
            "document_count": 0,
            "documents": [],
            "failures": {
                "schema": [
                    {
                        "code": "COUNTIES_FILE_MISSING",
                        "severity": "error",
                        "summary": "counties.json not found.",
                        "details": str(e),
                    }
                ]
            },
        }
    except (ValueError, json.JSONDecodeError) as e:
        return {
            "status": "FAIL",
            "document_count": 0,
            "documents": [],
            "failures": {
                "schema": [
                    {
                        "code": "COUNTIES_FILE_INVALID",
                        "severity": "error",
                        "summary": "counties.json is invalid.",
                        "details": str(e),
                    }
                ]
            },
        }

    docs: List[Dict[str, Any]] = []
    for idx, raw_block in enumerate(blocks):
        extracted = langextract_deed(raw_block, model_id=model_id)
        tidy = validate_and_enrich(raw_block, extracted, counties, model_id=model_id, include_audit=include_audit)
        docs.append({"index": idx, "doc": tidy})

    overall_pass = all(d["doc"]["status"] == "PASS" for d in docs)
    return {"status": "PASS" if overall_pass else "FAIL", "document_count": len(docs), "documents": docs}


# =============================
# Main
# =============================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counties", default=DEFAULT_COUNTIES_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--text", default=None, help="Inline text input (single or multiple deed blocks)")
    parser.add_argument("--input_file", default=None, help="Path to .txt containing 1+ deed blocks")
    parser.add_argument("--output", default=None, help="Write output JSON to file (otherwise print)")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Include forensic audit details (presence/spans/raw extractions/loopback dumps)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logs (includes LangExtract warnings/progress)")
    args = parser.parse_args()

    configure_logging(args.verbose)

    blocks = read_input_blocks(args.text, args.input_file)
    output_obj = run_pipeline_on_blocks(blocks, counties_path=args.counties, model_id=args.model, include_audit=args.audit)

    doc_pass = sum(1 for d in output_obj.get("documents", []) if d.get("doc", {}).get("status") == "PASS")
    doc_fail = sum(1 for d in output_obj.get("documents", []) if d.get("doc", {}).get("status") == "FAIL")
    doc_total = output_obj.get("document_count", 0)
    doc_pass_rate = (doc_pass / doc_total) if doc_total else 0.0

    print("---- Pipeline doc stats ----")
    print(f"Docs PASS:          {doc_pass}/{doc_total}")
    print(f"Docs FAIL:          {doc_fail}/{doc_total}")
    print(f"Doc pass rate:      {doc_pass_rate:.2%}")

    out_json = json.dumps(output_obj, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"\n[ok] wrote {args.output}")
    else:
        print("\n")
        print(out_json)


if __name__ == "__main__":
    main()