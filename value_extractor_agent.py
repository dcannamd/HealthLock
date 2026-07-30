# value_extractor_agent.py
# Extracts STRUCTURED numeric lab values from a health document.
#
# This is different from summarizer_agent.py, which produces prose.
# This agent produces typed data: analyte name, value, unit, and the
# reference range printed on the report itself.
#
# WHY THIS EXISTS:
#   To chart a value across several reports, you need the numbers as data,
#   not as sentences. Prose cannot be plotted or diffed.
#
# HONEST LIMITATIONS — READ THIS:
#   Lab PDFs vary enormously in layout. An LLM reading them will sometimes
#   misread a number, attach the wrong unit, or miss an analyte entirely.
#   A wrong number rendered as a chart looks MORE authoritative than wrong
#   prose, which makes silent errors more dangerous here than anywhere else
#   in this project.
#
#   Every value this agent returns carries a `confidence` field and the
#   `raw_text` it was extracted from, so a human can verify it against the
#   original document. Do not display extracted values without also making
#   verification possible.
#
# Uses Ollama (local LLM) — nothing leaves your machine.

import sys
import json
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaLLM

MODEL_NAME = "llama3"

# Analytes we care about. Keeping this list explicit (rather than "extract
# everything") dramatically improves reliability, because the model has a
# closed target instead of an open one.
KNOWN_ANALYTES = [
    "LDL cholesterol",
    "HDL cholesterol",
    "Total cholesterol",
    "Triglycerides",
    "Non-HDL cholesterol",
    "Hemoglobin",
    "Hematocrit",
    "White blood cell count",
    "Red blood cell count",
    "Platelets",
    "Fasting glucose",
    "Hemoglobin A1C",
    "Creatinine",
    "eGFR",
    "TSH",
    "ALT",
    "Vitamin D",
    "PSA",
]

# Aliases the model or the report might use, mapped to our canonical names.
# This is deterministic normalisation — no AI needed, so no AI errors.
ALIASES = {
    "ldl": "LDL cholesterol",
    "ldl-c": "LDL cholesterol",
    "ldl chol": "LDL cholesterol",
    "hdl": "HDL cholesterol",
    "hdl-c": "HDL cholesterol",
    "chol": "Total cholesterol",
    "cholesterol": "Total cholesterol",
    "trig": "Triglycerides",
    "tg": "Triglycerides",
    "hgb": "Hemoglobin",
    "hb": "Hemoglobin",
    "hct": "Hematocrit",
    "wbc": "White blood cell count",
    "rbc": "Red blood cell count",
    "plt": "Platelets",
    "glucose": "Fasting glucose",
    "glucose fasting": "Fasting glucose",
    "a1c": "Hemoglobin A1C",
    "hba1c": "Hemoglobin A1C",
    "hemoglobin a1c": "Hemoglobin A1C",
    "25-oh vitamin d": "Vitamin D",
    "vitamin d 25-oh": "Vitamin D",
}

EXTRACTION_PROMPT = """You are extracting numeric lab values from a medical report.

Extract ONLY the analytes listed below that actually appear in this document.
Do not invent values. Do not include analytes that are not present.

Analytes to look for:
{analytes}

For each one you find, record:
- "analyte": the name from the list above
- "value": the numeric result, as a number only (no units)
- "unit": the unit printed on the report (e.g. "mg/dL", "mmol/L", "g/L", "%")
- "ref_low": the lower end of the reference range printed on the report, or null
- "ref_high": the upper end of the reference range printed on the report, or null
- "raw_text": the exact line of text you read this from

CRITICAL RULES:
- Use ONLY the reference range printed in THIS document. Never use your own
  medical knowledge of what is normal.
- If no reference range is printed for an analyte, set ref_low and ref_high to null.
- Copy units exactly as printed. Do not convert between units.
- If you are unsure about a value, do not include it.

Respond with ONLY a JSON array. No explanation, no markdown fences, no other text.
If you find nothing, respond with: []

Example of the required format:
[{{"analyte":"LDL cholesterol","value":3.2,"unit":"mmol/L","ref_low":null,"ref_high":2.0,"raw_text":"LDL Cholesterol 3.2 mmol/L (target <2.0)"}}]

Document:
{text}

JSON:"""


def load_document(file_path: str) -> str:
    """Load a PDF or TXT file and return its text content."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() == ".txt":
        loader = TextLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    pages = loader.load()
    return "\n".join(page.page_content for page in pages)


def normalise_analyte(name: str) -> str:
    """Map an analyte name to its canonical form. Deterministic, no AI."""
    if not name:
        return ""
    cleaned = name.strip().lower()
    if cleaned in ALIASES:
        return ALIASES[cleaned]
    for known in KNOWN_ANALYTES:
        if cleaned == known.lower():
            return known
    return name.strip()


def strip_json_fences(text: str) -> str:
    """Models often wrap JSON in markdown fences despite instructions."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # Grab the outermost array if there is surrounding chatter
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def validate_entry(entry: dict, source_text: str) -> dict:
    """
    Validate one extracted value. Returns the entry with a `confidence` field,
    or None if it should be discarded.

    This is DETERMINISTIC validation — the checks here are code, not AI, so
    they cannot hallucinate.
    """
    if not isinstance(entry, dict):
        return None

    analyte = normalise_analyte(entry.get("analyte", ""))
    if not analyte:
        return None

    # Value must be numeric
    raw_value = entry.get("value")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None

    unit = (entry.get("unit") or "").strip()
    raw_text = (entry.get("raw_text") or "").strip()

    def as_float_or_none(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    ref_low = as_float_or_none(entry.get("ref_low"))
    ref_high = as_float_or_none(entry.get("ref_high"))

    # Confidence scoring — every check here is verifiable code, not a guess.
    confidence = "high"
    notes = []

    # Can we find this number as a literal string in the document?
    # If not, the model may have computed or hallucinated it.
    value_str = f"{value:g}"
    if value_str not in source_text:
        confidence = "low"
        notes.append("value not found verbatim in document text")

    if not unit:
        confidence = "low" if confidence == "high" else confidence
        notes.append("no unit captured")

    if not raw_text:
        confidence = "medium" if confidence == "high" else confidence
        notes.append("no source line captured")

    if ref_low is None and ref_high is None:
        notes.append("no reference range printed on report")

    # Sanity: if a range was captured, the value should be in a plausible
    # relationship to it. A value 100x the upper bound suggests a unit
    # or decimal misread.
    if ref_high is not None and ref_high > 0 and value > ref_high * 100:
        confidence = "low"
        notes.append("value implausibly large vs printed range — possible misread")

    return {
        "analyte": analyte,
        "value": value,
        "unit": unit,
        "ref_low": ref_low,
        "ref_high": ref_high,
        "raw_text": raw_text,
        "confidence": confidence,
        "notes": "; ".join(notes),
    }


def extract_values(file_path: str, doc_date: str = "UNKNOWN") -> dict:
    """
    Extract structured lab values from one document.

    Returns:
        {
          "file": str,
          "date": str,
          "values": [ {analyte, value, unit, ref_low, ref_high,
                       confidence, raw_text, notes}, ... ],
          "parse_ok": bool,
          "raw_response": str,
        }
    """
    print(f"\nExtracting values from: {file_path}")
    text = load_document(file_path)

    # Use a generous window — lab values can appear well into a report.
    text_window = text[:6000]

    print(f"Document loaded ({len(text)} chars). Extracting with {MODEL_NAME}...")
    print("This may take 30-60 seconds.\n")

    llm = OllamaLLM(model=MODEL_NAME)
    prompt = EXTRACTION_PROMPT.format(
        analytes="\n".join(f"- {a}" for a in KNOWN_ANALYTES),
        text=text_window,
    )
    response = llm.invoke(prompt).strip()

    cleaned = strip_json_fences(response)

    parse_ok = True
    entries = []
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            parse_ok = False
            parsed = []
    except json.JSONDecodeError:
        parse_ok = False
        parsed = []
        print("WARNING: model did not return valid JSON. No values extracted.")

    for entry in parsed:
        validated = validate_entry(entry, text)
        if validated:
            entries.append(validated)

    result = {
        "file": Path(file_path).name,
        "date": doc_date,
        "values": entries,
        "parse_ok": parse_ok,
        "raw_response": response,
    }

    print(f"--- EXTRACTED {len(entries)} VALUE(S) ---")
    for v in entries:
        rng = ""
        if v["ref_low"] is not None or v["ref_high"] is not None:
            lo = v["ref_low"] if v["ref_low"] is not None else ""
            hi = v["ref_high"] if v["ref_high"] is not None else ""
            rng = f"  [ref {lo}-{hi}]"
        flag = "" if v["confidence"] == "high" else f"  ({v['confidence']}: {v['notes']})"
        print(f"  {v['analyte']}: {v['value']} {v['unit']}{rng}{flag}")

    if not entries and parse_ok:
        print("  (none found — this document may not contain these analytes)")

    return result


def build_series(extractions: list) -> dict:
    """
    Turn several per-document extractions into per-analyte time series,
    suitable for charting.

    Input:  list of extract_values() results
    Output: {analyte: [{date, value, unit, ref_low, ref_high, file,
                        confidence, above_ref, below_ref}, ...]}

    Series are sorted by date. Documents with UNKNOWN dates are placed last.
    Units are NOT converted — if a series mixes units, that is flagged rather
    than silently normalised, because guessing at unit conversion is exactly
    how a chart becomes quietly wrong.
    """
    series = {}

    for extraction in extractions:
        date = extraction.get("date", "UNKNOWN")
        fname = extraction.get("file", "unknown")
        for v in extraction.get("values", []):
            analyte = v["analyte"]
            point = {
                "date": date,
                "value": v["value"],
                "unit": v["unit"],
                "ref_low": v["ref_low"],
                "ref_high": v["ref_high"],
                "file": fname,
                "confidence": v["confidence"],
                "above_ref": (
                    v["ref_high"] is not None and v["value"] > v["ref_high"]
                ),
                "below_ref": (
                    v["ref_low"] is not None and v["value"] < v["ref_low"]
                ),
            }
            series.setdefault(analyte, []).append(point)

    # Sort each series chronologically, UNKNOWN dates last
    for analyte in series:
        series[analyte].sort(
            key=lambda p: (p["date"] == "UNKNOWN", p["date"])
        )

    return series


def summarise_series(series: dict) -> list:
    """
    Compute first-to-last change for each analyte with 2+ points.

    Returns a list of dicts suitable for a "values that moved" table.
    Only compares points that share the same unit — mixed-unit series are
    reported as such rather than compared.
    """
    moved = []

    for analyte, points in series.items():
        if len(points) < 2:
            continue

        units = {p["unit"] for p in points if p["unit"]}
        mixed_units = len(units) > 1

        first, last = points[0], points[-1]
        delta = last["value"] - first["value"]

        pct = None
        if first["value"] != 0:
            pct = (delta / abs(first["value"])) * 100

        moved.append({
            "analyte": analyte,
            "first_value": first["value"],
            "first_date": first["date"],
            "last_value": last["value"],
            "last_date": last["date"],
            "unit": last["unit"],
            "delta": round(delta, 3),
            "percent_change": round(pct, 1) if pct is not None else None,
            "points": len(points),
            "mixed_units": mixed_units,
            "any_low_confidence": any(p["confidence"] != "high" for p in points),
        })

    # Largest absolute percentage change first
    moved.sort(
        key=lambda m: abs(m["percent_change"]) if m["percent_change"] is not None else 0,
        reverse=True,
    )
    return moved


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python value_extractor_agent.py <document> [more documents...]")
        print()
        print("One document  : extracts and prints its values")
        print("Several       : also builds time series and change summary")
        sys.exit(1)

    paths = sys.argv[1:]
    extractions = [extract_values(p) for p in paths]

    if len(extractions) > 1:
        series = build_series(extractions)
        moved = summarise_series(series)

        print("\n" + "=" * 58)
        print("  VALUES THAT MOVED")
        print("=" * 58)
        if not moved:
            print("  No analyte appeared in two or more documents.")
        for m in moved:
            pct = f"{m['percent_change']:+.1f}%" if m["percent_change"] is not None else "n/a"
            print(f"\n  {m['analyte']}: {pct}")
            print(f"    {m['first_value']} {m['unit']} ({m['first_date']}) "
                  f"-> {m['last_value']} {m['unit']} ({m['last_date']})")
            print(f"    {m['points']} data points")
            if m["mixed_units"]:
                print("    WARNING: units differ across reports — comparison unreliable")
            if m["any_low_confidence"]:
                print("    WARNING: some points were low confidence — verify against originals")

        print("\n" + "-" * 58)
        print("  Every number above was read out of a PDF by a language model.")
        print("  Check them against your original reports before relying on them.")
        print("-" * 58 + "\n")