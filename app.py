import streamlit as st
import json
import os
import re
from typing import List, Optional

from pydantic import BaseModel
from google import genai
from google.genai import types

_api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
client = genai.Client(api_key=_api_key)

GEMINI_MODEL = "gemini-3.5-flash"

st.set_page_config(page_title="Invoice Extractor", layout="wide")

GSTIN_PATTERN = r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]{1}\b'

KNOWN_VENDOR_LABELS = [
    "sold by", "seller", "vendor", "supplier", "from", "issued by",
    "dispatched by",
]
KNOWN_CUSTOMER_LABELS = [
    "billing address", "bill to", "buyer", "customer", "to", "sold to",
    "consignee", "client",
]
KNOWN_TAX_LABELS = [
    "cgst", "sgst", "igst", "vat", "gst", "service tax", "cess", "tax",
    "sales tax", "output tax",
]

# Tolerance (in currency units) below which a total-mismatch is treated as
# rounding noise rather than a real discrepancy.
TALLY_TOLERANCE = 1.0


def _normalize_label(label):
    return re.sub(r'[^a-z ]', '', label.lower()).strip()


def _strip_rate_suffix(label):
    return re.sub(r'[@(]?\s*[\d.]+\s*%\)?', '', label).strip()


def _is_known(normalized, known_list):
    if normalized in known_list:
        return True
    return any(re.search(rf'\b{re.escape(known)}\b', normalized) for known in known_list)


def audit_labels(final):
    entries = []

    v_label = final.get("vendor_label_used")
    if v_label:
        known = _is_known(_normalize_label(v_label), KNOWN_VENDOR_LABELS)
        entries.append({"category": "Vendor", "label": v_label, "known": known})

    c_label = final.get("customer_label_used")
    if c_label:
        known = _is_known(_normalize_label(c_label), KNOWN_CUSTOMER_LABELS)
        entries.append({"category": "Customer", "label": c_label, "known": known})

    for tax_line in final.get("taxes") or []:
        label = tax_line.get("label") if isinstance(tax_line, dict) else None
        if not label:
            continue
        known = _is_known(_normalize_label(_strip_rate_suffix(label)), KNOWN_TAX_LABELS)
        entries.append({"category": "Tax", "label": label, "known": known})

    return entries


class TaxLine(BaseModel):
    label: Optional[str] = None
    rate_percent: Optional[str] = None
    amount: Optional[str] = None


class HsnSummaryLine(BaseModel):
    hsn_code: Optional[str] = None
    quantity: Optional[str] = None
    bill_amount: Optional[str] = None
    cgst_amount: Optional[str] = None
    sgst_amount: Optional[str] = None
    igst_amount: Optional[str] = None
    total_amount: Optional[str] = None


class InvoiceItem(BaseModel):
    sl_no: Optional[str] = None
    description: Optional[str] = None
    hsn_sac_code: Optional[str] = None
    quantity: Optional[str] = None
    unit_price: Optional[str] = None
    discount_amount: Optional[str] = None
    amount: Optional[str] = None
    tax_rate_percent: Optional[str] = None
    tax_amount: Optional[str] = None
    # Only filled when THIS row explicitly prints a CGST/SGST/IGST split of
    # its own (e.g. "OUTPUT CGST 9%" / "OUTPUT SGST 9%" against this exact
    # row). Leave null when the row only shows one combined tax figure —
    # that case stays in tax_rate_percent / tax_amount above.
    igst_rate_percent: Optional[str] = None
    igst_amount: Optional[str] = None
    cgst_rate_percent: Optional[str] = None
    cgst_amount: Optional[str] = None
    sgst_rate_percent: Optional[str] = None
    sgst_amount: Optional[str] = None
    total_amount: Optional[str] = None


class InvoiceData(BaseModel):
    vendor_name: Optional[str] = None
    vendor_gstin: Optional[str] = None
    vendor_pan: Optional[str] = None
    vendor_phone: Optional[str] = None
    vendor_email: Optional[str] = None
    vendor_website: Optional[str] = None
    vendor_bank_name: Optional[str] = None
    vendor_account_number: Optional[str] = None
    vendor_ifsc: Optional[str] = None
    vendor_label_used: Optional[str] = None

    customer_name: Optional[str] = None
    customer_gstin: Optional[str] = None
    customer_phone: Optional[str] = None
    customer_email: Optional[str] = None
    customer_label_used: Optional[str] = None

    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    payment_terms: Optional[str] = None
    place_of_supply: Optional[str] = None
    currency: Optional[str] = None

    subtotal: Optional[str] = None
    invoice_discount_percent: Optional[str] = None
    invoice_discount_amount: Optional[str] = None
    taxes: List[TaxLine] = []
    hsn_summary: List[HsnSummaryLine] = []
    total_tax: Optional[str] = None
    total_amount: Optional[str] = None
    amount_in_words: Optional[str] = None
    hsn_sac_code: Optional[str] = None

    items: List[InvoiceItem] = []


EXTRACTION_PROMPT = """Extract structured invoice data from the attached document. It may be a
digital PDF, a scanned/photographed invoice, a handwritten bill, a receipt, or any
country/company/layout — read whatever is actually on the page. Only fill a field if
that exact information is printed on the invoice. Never invent, derive, or
reconstruct a value that isn't separately and explicitly shown.

VENDOR = the seller who issued the invoice, CUSTOMER = the billed party. Common
labels: vendor is "Sold By"/"Seller"/"Vendor"/"From"/"Issued By"; customer is
"Billing Address"/"Bill To"/"Buyer"/"Customer"/"To:"/"Consignee". A "Shipping
Address" is just a delivery point, not the customer identity — don't confuse them.
Also report the EXACT label text printed on this invoice next to each party (e.g.
"Sold By", "To:", "Consignee") in vendor_label_used / customer_label_used, verbatim
as printed — this lets unfamiliar wording be caught and reviewed rather than
silently guessed at.

GSTIN — CRITICAL: vendor_gstin and customer_gstin are two DIFFERENT numbers printed
near each party's own name block. Never return the same value for both. If you can
only clearly find one GSTIN on the page, leave the other one null rather than
guessing or duplicating it.

LINE ITEMS: read each table's column HEADERS to map fields correctly — never assume
column position, different invoices order columns differently.
- "quantity" = the units/count column. "unit_price" = the per-unit rate, before tax
  and before discount.
- "amount" = the row's PRE-TAX net/taxable value — if the table has a column
  literally named "Taxable Amount"/"Net Amount", use that one specifically. If
  the table only has ONE total-like column with no separate pre/post-tax split,
  use that single column here instead (don't invent a taxable-amount split that
  doesn't exist).
- "total_amount" (item-level, optional) = ONLY fill this if the SAME row also
  prints its own separate GRAND TOTAL column (i.e. the row shows two distinct
  numbers: a pre-tax amount AND a separate post-tax total, e.g. "Taxable
  Amount" next to "Total"). If the row only has one total-like figure, leave
  this null — don't duplicate "amount" into it.
- Tax per row: if the row shows a tax RATE (e.g. "GST% = 5.00"), put that number
  in tax_rate_percent. If the row shows a tax AMOUNT in currency directly, put
  that in tax_amount. If the row instead prints BOTH a pre-tax amount and a
  separate post-tax total (as above) but no explicit tax-amount column, COMPUTE
  tax_amount as (total_amount − amount) for that row — this is not invented,
  it's arithmetic on two numbers already printed on that exact row. Otherwise,
  if only a rate or only a single combined figure is shown, leave tax_amount
  null rather than guessing. Most invoices do NOT break tax into separate lines
  per row (that split, if present at all, is usually only in an invoice-level
  summary box, which belongs in the top-level "taxes" list instead, not per item).
- Per-item CGST/SGST/IGST: if — and only if — THIS SPECIFIC row (or the block of
  text directly under this item) explicitly prints its own CGST/SGST/IGST split
  (e.g. "OUTPUT CGST 9%" and "OUTPUT SGST 9%" listed right under one item), fill
  igst_rate_percent/igst_amount, cgst_rate_percent/cgst_amount, and
  sgst_rate_percent/sgst_amount from those exact printed figures for that row.
  Leave whichever of the three types isn't printed as null (e.g. IGST fields stay
  null on a row that only shows CGST+SGST). If the row does NOT show its own
  CGST/SGST/IGST split and only has a single generic tax_rate_percent/tax_amount
  (or no per-row tax at all), leave all six of these fields null — do not copy the
  generic tax_rate_percent/tax_amount into them.
- If a discount is shown only ONCE for the whole invoice (e.g. "Discount 17%" in a
  totals/summary box, not repeated per row), that belongs in the top-level
  invoice_discount_percent / invoice_discount_amount instead — leave every item's
  discount_amount null in that case.
- Extract every row of the table, in the order shown, skip none. Do not add columns
  that don't exist on this particular invoice — leave those fields null.
- IMPORTANT — tax base per row: whatever "amount" you record for a row is the ONLY
  base that row's tax_rate_percent / tax_amount applies to. Some invoices (e.g.
  travel/booking invoices) print several distinct charge lines — such as a base
  fare/ticket line and a separate service-fee line — where tax (e.g. IGST/GST) is
  charged ONLY on the service-fee line and NOT on the fare line. Never assume tax
  applies uniformly across all rows; read each row's own printed tax figures only.

INVOICE-LEVEL TAXES: do not force values into fixed CGST/SGST/IGST/VAT slots. Instead,
list every distinct tax line actually printed in the invoice's totals/summary box as
its own entry in "taxes", using the label exactly as printed (e.g. "CGST", "SGST",
"IGST@18%", "VAT", "Service Tax", "GST") — one entry per line printed, in the order
shown, with rate_percent and/or amount filled from whatever that line shows. If the
invoice only prints one combined tax figure (no breakdown at all), that's a single
entry with whatever label is printed (or "Tax" if truly unlabeled). If a single
overall "Total Tax" figure is separately printed in addition to the breakdown, put
that in the top-level total_tax field — otherwise leave total_tax null rather than
computing it yourself by summing the taxes list.
DO NOT DUPLICATE: if the totals box only prints ONE combined tax figure and nothing
else — no distinct tax types broken out as their own separate lines — that single
figure belongs ONLY in total_tax. Do not also create a one-entry "taxes" list
repeating that same number under whatever label happened to be printed next to it
(e.g. "GST %", "Tax", "GST"). A "taxes" entry is for when the invoice shows genuinely
separate, distinctly-labeled tax lines side by side (e.g. CGST and SGST each printed
as their own row) — a single combined figure is not a breakdown, so leave "taxes"
empty in that case and rely on total_tax alone.

IMPORTANT — rate vs amount: judge by the actual NUMBER, not by the printed label text.
Some invoices carry a row literally labeled "GST %" that actually holds a currency
total (not a percentage) — a poorly designed template, not an error on your part. A
plausible tax rate is a small number, typically well under 100 (e.g. 5, 9, 12, 18). If
the number under a "...%"-labeled row is clearly not a plausible rate (e.g. it's in
the hundreds/thousands, or matches the sum of other amounts on the invoice), it is a
currency amount — put it in "amount", not "rate_percent", regardless of what the
printed label says.

HSN-WISE SUMMARY TABLE (optional, rare): some invoices carry a SECOND, separate table
— distinct from the main line-item table above — that groups quantities/amounts/tax
BY HSN CODE, typically with columns like HSN | QTY | BILL AMT | CGST | SGST, appearing
near the totals box. If such a table is present on this specific invoice, extract it
into "hsn_summary", one entry per HSN group/row, using exactly the columns that table
actually has (it may show IGST instead of CGST/SGST, or a single combined tax amount
instead — only fill what's printed). Most invoices do NOT have this second table —
in that case leave hsn_summary as an empty list. Do not construct one yourself by
grouping the line items — only extract it if it's a table that's actually printed.

FIELD RULES:
- All numeric fields: plain strings, no currency symbols, no thousands separators.
- customer_name: the human-readable name only — never include a GSTIN/PAN in it.
- vendor_pan: exactly 10 characters (5 letters + 4 digits + 1 letter), but ONLY if a
  PAN is explicitly printed on the invoice as its own field (e.g. "PAN No: ...",
  "PAN: ..."). Do NOT derive or guess a PAN by extracting characters out of the
  GSTIN, even though a GSTIN's middle 10 characters happen to match PAN format —
  that is fabricating a value that was never separately printed. If no distinct PAN
  field is printed, leave vendor_pan null.
- hsn_sac_code (top-level): a purely numeric 4-8 digit code explicitly labeled HSN
  or SAC — never an order/invoice number.
- Leave any field null if it is not present on the invoice. Do not invent values.
"""


def extract_invoice(file_bytes, mime_type="application/pdf"):
    import time

    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    EXTRACTION_PROMPT,
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=InvoiceData,
                ),
            )
            if response.parsed is not None:
                return response.parsed.model_dump(), None
            return json.loads(response.text), None
        except Exception as e:
            last_error = e
            transient = "UNAVAILABLE" in str(e) or "503" in str(e) or "overloaded" in str(e).lower()
            if transient and attempt < max_attempts:
                time.sleep(2 * attempt)
                continue
            break

    return {}, f"Gemini extraction error: {str(last_error)}"


def is_empty(val):
    return val in [None, "", "null", "NOT FOUND", "N/A", "n/a", [], {}]


def clean_customer_name(name):
    if not name:
        return name
    name = re.sub(GSTIN_PATTERN, '', name)
    name = re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '', name)
    return re.sub(r'\s{2,}', ' ', name).strip() or None


def _to_float(val):
    """Best-effort numeric parse; returns None for anything non-numeric/empty."""
    if is_empty(val):
        return None
    try:
        cleaned = str(val).replace(",", "").replace("₹", "").replace("Rs.", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Rule 1: Split whatever tax lines were extracted into CGST / SGST / IGST
# buckets (each with its own rate + amount), leaving unused buckets empty.
# Anything that isn't one of those three (VAT, Cess, Service Tax, plain
# "Tax", etc.) is kept as "other" so nothing gets silently dropped.
# ---------------------------------------------------------------------------
def classify_gst(taxes):
    buckets = {"CGST": None, "SGST": None, "IGST": None}
    others = []
    for t in taxes or []:
        if not isinstance(t, dict):
            continue
        label = (t.get("label") or "").upper()
        entry = {"rate": t.get("rate_percent"), "amount": t.get("amount")}
        if "CGST" in label and buckets["CGST"] is None:
            buckets["CGST"] = entry
        elif "SGST" in label and buckets["SGST"] is None:
            buckets["SGST"] = entry
        elif "IGST" in label and buckets["IGST"] is None:
            buckets["IGST"] = entry
        else:
            others.append(t)
    return buckets, others


# ---------------------------------------------------------------------------
# Rule 2: When an invoice has a separate HSN-wise summary table, match it
# against the line items (grouped by HSN code) and flag any HSN group whose
# summed line-item amount doesn't tally with what the summary table prints.
# ---------------------------------------------------------------------------
def match_items_to_hsn(items, hsn_summary):
    if not items or not hsn_summary:
        return []

    grouped = {}
    for it in items:
        code = it.get("hsn_sac_code")
        if not code:
            continue
        g = grouped.setdefault(code, {"amount": 0.0, "tax_amount": 0.0, "total_amount": 0.0, "count": 0})
        amt = _to_float(it.get("amount")) or 0.0
        tax_amt = _to_float(it.get("tax_amount")) or 0.0
        item_total = _to_float(it.get("total_amount"))
        g["amount"] += amt
        g["tax_amount"] += tax_amt
        g["total_amount"] += item_total if item_total is not None else (amt + tax_amt)
        g["count"] += 1

    results = []
    for row in hsn_summary:
        code = row.get("hsn_code")
        g = grouped.get(code)
        summary_bill = _to_float(row.get("bill_amount"))

        entry = {
            "hsn_code": code,
            "summary_bill_amount": row.get("bill_amount"),
            "summary_total_amount": row.get("total_amount"),
            "items_matched": g["count"] if g else 0,
            "items_amount_sum": round(g["amount"], 2) if g else None,
        }

        if not g:
            entry["status"] = "⚠️ No matching line items for this HSN code"
        elif summary_bill is None:
            entry["status"] = "Cannot verify — no bill amount printed in summary row"
        else:
            diff = round(g["amount"] - summary_bill, 2)
            entry["status"] = "✅ Matched" if abs(diff) <= TALLY_TOLERANCE else f"⚠️ Mismatch (Δ {diff:+.2f})"
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Rule 3: Recompute what the invoice total *should* be from its own printed
# parts, and compare against the printed total_amount. Handles the case
# where tax only applies to part of the invoice (e.g. service fee, not
# fare) by falling back to summing each line item's own total when a clean
# subtotal + tax path isn't available.
# ---------------------------------------------------------------------------
def reconcile_invoice_totals(final):
    total_amount = _to_float(final.get("total_amount"))
    if total_amount is None:
        return None

    subtotal = _to_float(final.get("subtotal"))
    total_tax = _to_float(final.get("total_tax"))

    taxes = final.get("taxes") or []
    taxes_sum = sum((_to_float(t.get("amount")) or 0.0) for t in taxes if isinstance(t, dict))
    effective_tax = total_tax if total_tax is not None else (taxes_sum if taxes else None)

    discount = _to_float(final.get("invoice_discount_amount"))
    if discount is None and final.get("invoice_discount_percent") and subtotal is not None:
        pct = _to_float(final.get("invoice_discount_percent"))
        discount = round(subtotal * pct / 100, 2) if pct is not None else None

    if subtotal is not None and effective_tax is not None:
        expected_total = round(subtotal - (discount or 0.0) + effective_tax, 2)
        basis = "subtotal − discount + tax"
    else:
        # Fall back to summing each line item's own total — this is what
        # correctly handles invoices where tax only applies to some rows
        # (e.g. a service fee) and not others (e.g. the fare), since each
        # item's own total already reflects whatever tax was on that row.
        items = final.get("items") or []
        items_total = 0.0
        any_item_data = False
        for it in items:
            amt = _to_float(it.get("amount"))
            tax_amt = _to_float(it.get("tax_amount"))
            item_total = _to_float(it.get("total_amount"))
            if item_total is not None:
                items_total += item_total
                any_item_data = True
            elif amt is not None:
                items_total += amt + (tax_amt or 0.0)
                any_item_data = True
        if not any_item_data:
            return None
        expected_total = round(items_total, 2)
        basis = "sum of each line item's own total (tax applied per-row)"

    diff = round(total_amount - expected_total, 2)
    return {
        "basis": basis,
        "expected_total": expected_total,
        "printed_total": total_amount,
        "difference": diff,
        "needs_review": abs(diff) > TALLY_TOLERANCE,
    }


# ---------------------------------------------------------------------------
# Single unified table: Item | HSN | Qty | Unit Price | Amount | IGST(%,Amt) |
# CGST(%,Amt) | SGST(%,Amt) | Total — one row per line item, one table per
# invoice.
#
# Three passes, most-specific evidence first:
#   1. Use whatever explicit CGST/SGST/IGST split is printed directly on
#      that row.
#   2. Use the row's own generic tax_rate_percent/tax_amount (or a tax
#      implied by that row's own printed total minus its own amount),
#      routed to whichever tax type the invoice actually uses.
#   3. For rows that still show NO tax signal at all, check whether the
#      invoice-level tax rate + amount arithmetically implies a taxable
#      base that matches exactly one of these otherwise-untaxed rows'
#      amount. If so, assign the tax there — this is deterministic
#      arithmetic on numbers already printed, not a guess. If no unique
#      match exists, the row is left untaxed and the totals reconciliation
#      will flag the invoice for review instead of silently misallocating.
# ---------------------------------------------------------------------------
def _find_unique_amount_match(candidates, target_amount, rate):
    if target_amount is None or not rate:
        return None
    implied_base = target_amount / (rate / 100)
    matches = [c for c in candidates
               if c["item_amt"] is not None and abs(c["item_amt"] - implied_base) <= TALLY_TOLERANCE]
    return matches[0] if len(matches) == 1 else None


def build_unified_table(final):
    items = final.get("items") or []
    gst = final.get("gst_breakdown") or {}
    cgst_b, sgst_b, igst_b = gst.get("CGST"), gst.get("SGST"), gst.get("IGST")

    parsed = []
    for it in items:
        igst_pct, igst_amt = it.get("igst_rate_percent"), it.get("igst_amount")
        cgst_pct, cgst_amt = it.get("cgst_rate_percent"), it.get("cgst_amount")
        sgst_pct, sgst_amt = it.get("sgst_rate_percent"), it.get("sgst_amount")
        gen_rate, gen_amt = it.get("tax_rate_percent"), it.get("tax_amount")
        item_amt = _to_float(it.get("amount"))
        item_total = _to_float(it.get("total_amount"))

        if is_empty(gen_amt) and is_empty(gen_rate) and item_total is not None and item_amt is not None:
            implied = round(item_total - item_amt, 2)
            if implied > TALLY_TOLERANCE:
                gen_amt = implied

        has_explicit_split = not (is_empty(igst_amt) and is_empty(cgst_amt) and is_empty(sgst_amt))
        has_row_signal = has_explicit_split or (not is_empty(gen_rate)) or (not is_empty(gen_amt))

        parsed.append({
            "it": it, "item_amt": item_amt,
            "igst_pct": igst_pct, "igst_amt": igst_amt,
            "cgst_pct": cgst_pct, "cgst_amt": cgst_amt,
            "sgst_pct": sgst_pct, "sgst_amt": sgst_amt,
            "gen_rate": gen_rate, "gen_amt": gen_amt,
            "has_explicit_split": has_explicit_split, "has_row_signal": has_row_signal,
        })

    # Pass 2 — route each row's own generic tax figure into the right columns.
    for p in parsed:
        if p["has_explicit_split"] or not p["has_row_signal"]:
            continue
        gen_rate, gen_amt, item_amt = p["gen_rate"], p["gen_amt"], p["item_amt"]
        if igst_b and not cgst_b and not sgst_b:
            p["igst_pct"] = gen_rate if not is_empty(gen_rate) else igst_b.get("rate")
            if not is_empty(gen_amt):
                p["igst_amt"] = gen_amt
            elif item_amt is not None and not is_empty(p["igst_pct"]):
                p["igst_amt"] = round(item_amt * _to_float(p["igst_pct"]) / 100, 2)
        elif cgst_b or sgst_b:
            half_rate = _to_float(gen_rate) / 2 if not is_empty(gen_rate) else None
            p["cgst_pct"] = half_rate if half_rate is not None else (cgst_b or {}).get("rate")
            p["sgst_pct"] = half_rate if half_rate is not None else (sgst_b or {}).get("rate")
            if not is_empty(gen_amt):
                p["cgst_amt"] = round(_to_float(gen_amt) / 2, 2)
                p["sgst_amt"] = round(_to_float(gen_amt) / 2, 2)
            elif item_amt is not None and not is_empty(p["cgst_pct"]):
                p["cgst_amt"] = round(item_amt * _to_float(p["cgst_pct"]) / 100, 2)
                p["sgst_amt"] = round(item_amt * _to_float(p["sgst_pct"]) / 100, 2)

    # Pass 3 — unique amount-match for rows with zero tax signal so far.
    undetermined = [p for p in parsed if not p["has_row_signal"]]
    if undetermined:
        if igst_b and not cgst_b and not sgst_b:
            target, rate = _to_float(igst_b.get("amount")), _to_float(igst_b.get("rate"))
            match = _find_unique_amount_match(undetermined, target, rate)
            if match:
                match["igst_pct"], match["igst_amt"] = igst_b.get("rate"), igst_b.get("amount")
        elif cgst_b or sgst_b:
            c_amt, s_amt = _to_float((cgst_b or {}).get("amount")) or 0, _to_float((sgst_b or {}).get("amount")) or 0
            c_rate, s_rate = _to_float((cgst_b or {}).get("rate")) or 0, _to_float((sgst_b or {}).get("rate")) or 0
            target = round(c_amt + s_amt, 2) if (cgst_b or sgst_b) else None
            rate = round(c_rate + s_rate, 2) if (c_rate or s_rate) else None
            match = _find_unique_amount_match(undetermined, target, rate)
            if match:
                match["cgst_pct"], match["cgst_amt"] = (cgst_b or {}).get("rate"), (cgst_b or {}).get("amount")
                match["sgst_pct"], match["sgst_amt"] = (sgst_b or {}).get("rate"), (sgst_b or {}).get("amount")

    rows = []
    for p in parsed:
        it = p["it"]
        row = {
            "Item": it.get("description"),
            "HSN": it.get("hsn_sac_code"),
            "Qty": it.get("quantity"),
            "Unit Price": it.get("unit_price"),
            "Amount": it.get("amount"),
            "IGST %": p["igst_pct"] if not is_empty(p["igst_pct"]) else "",
            "IGST Amt": p["igst_amt"] if not is_empty(p["igst_amt"]) else "",
            "CGST %": p["cgst_pct"] if not is_empty(p["cgst_pct"]) else "",
            "CGST Amt": p["cgst_amt"] if not is_empty(p["cgst_amt"]) else "",
            "SGST %": p["sgst_pct"] if not is_empty(p["sgst_pct"]) else "",
            "SGST Amt": p["sgst_amt"] if not is_empty(p["sgst_amt"]) else "",
        }
        total_amt = it.get("total_amount")
        if is_empty(total_amt):
            tax_sum = sum(v for v in [_to_float(p["igst_amt"]), _to_float(p["cgst_amt"]), _to_float(p["sgst_amt"])]
                          if v)
            total_amt = round(p["item_amt"] + tax_sum, 2) if p["item_amt"] is not None else ""
        row["Total Amt"] = total_amt
        rows.append(row)

    return rows


def postprocess(data):
    final = {k: v for k, v in data.items() if k != "items"}

    if final.get("vendor_gstin") and final.get("customer_gstin"):
        if final["vendor_gstin"] == final["customer_gstin"]:
            final["customer_gstin"] = None

    if final.get("customer_name"):
        final["customer_name"] = clean_customer_name(final["customer_name"])

    taxes = final.get("taxes") or []
    total_tax = final.get("total_tax")
    if len(taxes) == 1 and total_tax:
        only_amount = taxes[0].get("amount") if isinstance(taxes[0], dict) else None
        if only_amount and str(only_amount).strip() == str(total_tax).strip():
            final["taxes"] = []

    hsn = final.get("hsn_sac_code")
    if hsn and not re.match(r'^\d{4,8}$', str(hsn)):
        final["hsn_sac_code"] = None

    pan = final.get("vendor_pan")
    if pan and not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', str(pan)):
        final["vendor_pan"] = None

    items = data.get("items") or []
    cleaned_items = []
    for item in items:
        item_hsn = item.get("hsn_sac_code")
        if item_hsn and not re.match(r'^\d{4,8}$', str(item_hsn)):
            item["hsn_sac_code"] = None
        cleaned_items.append(item)
    final["items"] = cleaned_items

    if not final.get("invoice_number"):
        final["invoice_number"] = "NOT FOUND"

    # Derived, non-schema fields — computed once here so both the UI and the
    # downloaded JSON reflect the same reconciliation results.
    gst_buckets, other_taxes = classify_gst(final.get("taxes"))
    final["gst_breakdown"] = {"CGST": gst_buckets["CGST"], "SGST": gst_buckets["SGST"],
                               "IGST": gst_buckets["IGST"], "other_taxes": other_taxes}
    final["unified_table"] = build_unified_table(final)
    hsn_recon = match_items_to_hsn(cleaned_items, final.get("hsn_summary"))
    total_recon = reconcile_invoice_totals(final)
    final["total_reconciliation"] = total_recon
    final["hsn_reconciliation_detail"] = hsn_recon

    # One combined review flag covering both checks, so the UI only needs a
    # single status line instead of separate tables/boxes for each.
    review_reasons = []
    if total_recon and total_recon["needs_review"]:
        review_reasons.append(
            f"total amount off by {total_recon['difference']:+.2f} "
            f"(expected {total_recon['expected_total']} vs printed {total_recon['printed_total']})"
        )
    for row in hsn_recon:
        if "Mismatch" in row.get("status", "") or "No matching" in row.get("status", ""):
            review_reasons.append(f"HSN {row.get('hsn_code')}: {row.get('status')}")
    final["needs_human_review"] = bool(review_reasons)
    final["review_reasons"] = review_reasons

    return final


st.title("📄Invoice Extractor")
st.caption(
    "Powered by Gemini — reads digital PDFs, scanned/photographed invoices, "
    "and handwritten bills"
)

uploaded_files = st.file_uploader(
    "Upload Invoice PDF(s)", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"📎 {uploaded_file.name}")
        file_bytes = uploaded_file.read()

        with st.spinner(f"Extracting {uploaded_file.name} with Gemini..."):
            llm_data, error = extract_invoice(file_bytes)

        if error:
            st.error(error)
            continue

        final_result = postprocess(llm_data)

        new_entries = [e for e in audit_labels(final_result) if not e["known"]]
        if new_entries:
            list_by_category = {
                "Vendor": "KNOWN_VENDOR_LABELS",
                "Customer": "KNOWN_CUSTOMER_LABELS",
                "Tax": "KNOWN_TAX_LABELS",
            }
            lines = [f'{e["category"].lower()}: "{e["label"]}"  →  add to {list_by_category[e["category"]]}'
                     for e in new_entries]
            st.info("🆕 New label(s) found:\n\n" + "\n\n".join(lines))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🏢 Vendor Details**")
            for field in ["vendor_name", "vendor_gstin", "vendor_pan", "vendor_phone",
                          "vendor_email", "vendor_website", "vendor_bank_name",
                          "vendor_account_number", "vendor_ifsc"]:
                val = final_result.get(field)
                if not is_empty(val):
                    st.write(f"**{field.replace('vendor_', '').replace('_', ' ').title()}:** {val}")

        with col2:
            st.markdown("**👤 Customer Details**")
            for field in ["customer_name", "customer_gstin", "customer_phone", "customer_email"]:
                val = final_result.get(field)
                if not is_empty(val):
                    st.write(f"**{field.replace('customer_', '').replace('_', ' ').title()}:** {val}")

        st.markdown("**🧾 Invoice Details**")
        inv_cols = st.columns(3)
        for idx, field in enumerate(["invoice_number", "invoice_date", "due_date",
                                      "payment_terms", "place_of_supply", "currency",
                                      "hsn_sac_code"]):
            val = final_result.get(field)
            if not is_empty(val):
                inv_cols[idx % 3].write(f"**{field.replace('_', ' ').title()}:** {val}")

        # ---------------- Single unified item + tax table ----------------
        unified_rows = final_result.get("unified_table") or []
        if unified_rows:
            import pandas as pd
            st.markdown("**📦 Items & Tax**")
            col_order = ["Item", "HSN", "Qty", "Unit Price", "Amount",
                         "IGST %", "IGST Amt", "CGST %", "CGST Amt", "SGST %", "SGST Amt",
                         "Total Amt"]
            df = pd.DataFrame(unified_rows)
            df = df[[c for c in col_order if c in df.columns]]
            df = df.fillna("").replace("null", "").replace("None", "")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Total items: **{len(unified_rows)}**")

        st.markdown("**💰 Amounts**")
        amt_cols = st.columns(3)
        static_fields = ["subtotal", "invoice_discount_percent",
                          "invoice_discount_amount", "total_tax", "total_amount",
                          "amount_in_words"]
        for idx, field in enumerate(static_fields):
            val = final_result.get(field)
            if not is_empty(val):
                amt_cols[idx % 3].write(f"**{field.replace('_', ' ').title()}:** {val}")

        # ---------------- Combined reconciliation status ----------------
        recon = final_result.get("total_reconciliation")
        if recon:
            st.write(
                f"Total check ({recon['basis']}): expected **{recon['expected_total']}** "
                f"vs printed **{recon['printed_total']}**"
            )

        if final_result.get("needs_human_review"):
            st.error("🚨 NEEDS HUMAN REVIEW\n\n" + "\n\n".join(f"- {r}" for r in final_result["review_reasons"]))
        elif recon:
            st.success("Totals tally correctly.")

        with st.expander("📋 Full JSON Output"):
            st.json(final_result)

        json_str = json.dumps(final_result, indent=4, ensure_ascii=False)
        st.download_button(
            label=f"📥 Download JSON — {uploaded_file.name}",
            data=json_str,
            file_name=f"invoice_{final_result.get('invoice_number', 'unknown')}.json",
            mime="application/json",
            key=f"dl_{uploaded_file.name}_{uploaded_file.size}",
        )
