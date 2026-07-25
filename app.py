import streamlit as st
import json
import os
import re
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from google import genai
from google.genai import types

# ─────────────────────────────────────────
# 🔑 API CLIENT (Gemini)
# ─────────────────────────────────────────
_api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
client = genai.Client(api_key=_api_key)


GEMINI_MODEL = "gemini-3.5-flash"


OUTPUT_DIR = "extracted_invoices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def _normalize_label(label):
    return re.sub(r'[^a-z ]', '', label.lower()).strip()


def _strip_rate_suffix(label):
    return re.sub(r'[@(]?\s*[\d.]+\s*%\)?', '', label).strip()
    
    


def audit_labels(final):
    """
    Returns every label the model reported (vendor, customer, each tax line),
    tagged as known or new against the glossaries above. Always returns
    entries for whatever was actually detected — not just the unmatched ones —
    so you can see the full picture of wording used on each invoice, not only
    the rare miss.
    """
    entries = []

    v_label = final.get("vendor_label_used")
    if v_label:
        known = _normalize_label(v_label) in KNOWN_VENDOR_LABELS
        entries.append({"category": "Vendor", "label": v_label, "known": known})

    c_label = final.get("customer_label_used")
    if c_label:
        known = _normalize_label(c_label) in KNOWN_CUSTOMER_LABELS
        entries.append({"category": "Customer", "label": c_label, "known": known})

    for tax_line in final.get("taxes") or []:
        label = tax_line.get("label") if isinstance(tax_line, dict) else None
        if not label:
            continue
        known = _normalize_label(_strip_rate_suffix(label)) in KNOWN_TAX_LABELS
        entries.append({"category": "Tax", "label": label, "known": known})

    return entries




class TaxLine(BaseModel):
    label: Optional[str] = None          # verbatim as printed, e.g. "CGST@9%", "VAT", "Service Tax"
    rate_percent: Optional[str] = None   # if a % is printed for this line
    amount: Optional[str] = None         # if a currency amount is printed for this line


class HsnSummaryLine(BaseModel):
    """
    Some invoices carry a SECOND table, separate from the main line-item
    table, that groups quantities/amounts/tax by HSN code (e.g. columns like
    HSN | QTY | BILL AMT | CGST | SGST). This is optional — most invoices
    don't have one — so this list just stays empty when it's absent.
    """
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
- "amount" = whatever the row's own total/net column shows (often quantity ×
  unit_price) — use exactly what's printed under that column, whatever it's labeled
  (Amount, Net Amount, Total, etc). Do not invent a separate "taxable amount" field
  if the invoice only has one such column.
- Tax per row: if the row shows a tax RATE (e.g. "GST% = 5.00"), put that number in
  tax_rate_percent. If the row instead shows a tax AMOUNT in currency, put that in
  tax_amount. Only fill whichever one is actually printed for that row — most
  invoices show only one, not both, and most do NOT break tax into separate lines
  per row (that split, if present at all, is usually only in an invoice-level
  summary box, which belongs in the top-level "taxes" list instead, not per item).
- If a discount is shown only ONCE for the whole invoice (e.g. "Discount 17%" in a
  totals/summary box, not repeated per row), that belongs in the top-level
  invoice_discount_percent / invoice_discount_amount instead — leave every item's
  discount_amount null in that case.
- Extract every row of the table, in the order shown, skip none. Do not add columns
  that don't exist on this particular invoice — leave those fields null.

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


# ─────────────────────────────────────────
# 🤖 EXTRACTION
# ─────────────────────────────────────────

def extract_invoice(file_bytes, mime_type="application/pdf"):
    """
    Sends the file directly to Gemini — no OCR, no PDF-to-image rendering, no
    poppler/pytesseract dependency. Gemini reads digital-text and
    scanned/image PDFs natively in the same call.
    """
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
            # Fallback: schema-valid JSON text even if .parsed didn't populate
            return json.loads(response.text), None
        except Exception as e:
            last_error = e
            # 503/UNAVAILABLE means Google's servers are temporarily
            # overloaded — not a bug on our end. Worth a couple of short
            # retries before giving up, since it usually clears within
            # seconds. Anything else (bad key, invalid file, etc.) fails fast.
            transient = "UNAVAILABLE" in str(e) or "503" in str(e) or "overloaded" in str(e).lower()
            if transient and attempt < max_attempts:
                time.sleep(2 * attempt)  # 2s, then 4s
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

    return final


def save_json(final_result, filename_hint):
    safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', str(final_result.get("invoice_number") or filename_hint))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"{safe_name}_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    return path


# ─────────────────────────────────────────
# 🎨 STREAMLIT UI
# ─────────────────────────────────────────

st.title("📄 Invoice Extractor")
st.caption(
    "Powered by Gemini — reads digital PDFs, scanned/photographed invoices, "
    "and handwritten bills 
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
        saved_path = save_json(final_result, uploaded_file.name.rsplit(".", 1)[0])
        st.success(f"✅ Extracted and saved to `{saved_path}`")

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

        if final_result.get("items"):
            st.markdown("**📦 Line Items**")
            import pandas as pd
            items = final_result["items"]
            preferred_cols = ["sl_no", "description", "hsn_sac_code", "quantity",
                               "unit_price", "discount_amount", "amount",
                               "tax_rate_percent", "tax_amount"]
            df = pd.DataFrame(items)
            ordered = [c for c in preferred_cols if c in df.columns]
            df = df[ordered + [c for c in df.columns if c not in ordered]]
            df = df.fillna("").replace("null", "").replace("None", "")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Total items extracted: **{len(items)}**")

        st.markdown("**💰 Amounts**")
        amt_cols = st.columns(3)
        static_fields = ["subtotal", "invoice_discount_percent",
                          "invoice_discount_amount", "total_tax", "total_amount",
                          "amount_in_words"]
        for idx, field in enumerate(static_fields):
            val = final_result.get(field)
            if not is_empty(val):
                amt_cols[idx % 3].write(f"**{field.replace('_', ' ').title()}:** {val}")

        taxes = final_result.get("taxes") or []
        if taxes:
            st.markdown("**Tax Breakdown** *(as printed on the invoice)*")
            tax_cols = st.columns(3)
            for idx, tax_line in enumerate(taxes):
                label = tax_line.get("label") or "Tax"
                rate = tax_line.get("rate_percent")
                amount = tax_line.get("amount")
                parts = []
                if not is_empty(rate):
                    parts.append(f"{rate}%")
                if not is_empty(amount):
                    # If the printed label itself contains a "%" (e.g. this
                    # invoice's own row literally says "GST %") but only an
                    # amount was found — not an actual rate — make it clear
                    # this number is money, not a percentage, so it doesn't
                    # read as an invalid rate like "525%".
                    if "%" in label and is_empty(rate):
                        parts.append(f"{amount} (amount, not a rate — invoice's own label says \"%\")")
                    else:
                        parts.append(str(amount))
                if parts:
                    tax_cols[idx % 3].write(f"**{label}:** {' / '.join(parts)}")

        hsn_summary = final_result.get("hsn_summary") or []
        if hsn_summary:
            st.markdown("**📊 HSN-wise Summary** *(separate grouping table found on this invoice)*")
            import pandas as pd
            hsn_df = pd.DataFrame(hsn_summary)
            hsn_df = hsn_df.fillna("").replace("null", "").replace("None", "")
            st.dataframe(hsn_df, use_container_width=True, hide_index=True)

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
