# ingest_sam.py
from __future__ import annotations
import os
import time
import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("SAM_API_KEY")
BASE_URL = "https://api.sam.gov/opportunities/v2/search"

OUT_FILE         = Path("rfp.csv")
MAX_RECORDS_HARD_CAP = 1000
MAX_RETRIES      = 3

# Notice types with richest narrative descriptions.
# ptype codes: o=Solicitation, p=Presolicitation, r=Sources Sought,
#              k=Combined Synopsis/Solicitation, s=Special Notice,
#              a=Award Notice (no scope text — skip by default)
DEFAULT_PTYPES = ["o", "p", "r", "k", "s"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_url(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    try:
        u = urlparse(text.strip())
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def mmddyyyy(iso_date: str) -> str:
    """Convert 'YYYY-MM-DD' -> 'MM/DD/YYYY' for SAM."""
    y, m, d = iso_date.split("-")
    return f"{m}/{d}/{y}"


def _get_with_retry(url: str, params: dict, timeout: int = 60) -> requests.Response | None:
    """
    GET with exponential backoff on 429, 5xx, or network error.
    Returns Response on success (2xx), None if all retries exhausted.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            wait = 2 ** attempt
            if attempt < MAX_RETRIES:
                print(f"  Network error (attempt {attempt+1}/{MAX_RETRIES+1}): {exc} — retrying in {wait}s")
                time.sleep(wait)
                continue
            else:
                print(f"  Network error (attempt {attempt+1}/{MAX_RETRIES+1}): {exc} — giving up")
                return None

        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2 ** attempt
            if attempt < MAX_RETRIES:
                print(f"  HTTP {resp.status_code} (attempt {attempt+1}/{MAX_RETRIES+1}) — retrying in {wait}s")
                time.sleep(wait)
                continue
            else:
                print(f"  HTTP {resp.status_code} (attempt {attempt+1}/{MAX_RETRIES+1}) — giving up")
                return None

        return resp  # 2xx or non-retryable 4xx

    return None


def _save_rows(rows: list[dict]) -> None:
    """Deduplicate and write rows to OUT_FILE."""
    if not rows:
        print("No rows to save.")
        return
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["id"])
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUT_FILE.resolve()}")


def _org_from_parent_path(parent_path: str) -> str:
    """
    Extract the leaf agency name from fullParentPathName.
    'DEPT OF DEFENSE.DLA.DLA AVIATION' -> 'DLA AVIATION'
    """
    if not parent_path:
        return ""
    segments = [s.strip() for s in parent_path.split(".") if s.strip()]
    return segments[-1] if segments else ""


def _clean_html(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split()).strip()


# ── build_params ──────────────────────────────────────────────────────────────

def build_params(posted_from_iso: str,
                 posted_to_iso: str,
                 limit: int,
                 offset: int,
                 naics: str | None,
                 state: str | None,
                 ptypes: list[str] | None = None) -> dict:
    params = {
        "api_key":    API_KEY,
        "postedFrom": mmddyyyy(posted_from_iso),
        "postedTo":   mmddyyyy(posted_to_iso),
        "limit":      min(max(1, limit), 1000),
        "offset":     max(0, offset),
        "sort":       "-postedDate",
    }
    if naics:
        params["ncode"] = naics
    if state:
        params["state"] = state
    if ptypes:
        params["ptype"] = ",".join(ptypes)
    return params


# ── flatten_notice ─────────────────────────────────────────────────────────────
# IMPORTANT: this function does NOT make any HTTP calls.
# If the `description` field is a URL, we store it in `description_url`
# and leave `description_text` as whatever inline text exists.
# Run `--fetch-descriptions` separately to fill in description_text.

def flatten_notice(n: dict) -> dict:
    """Map a single SAM notice JSON into a flat CSV row. No HTTP calls made here."""
    notice_id  = n.get("noticeId") or n.get("id") or ""
    title      = (n.get("title") or "").strip()
    summary_raw = (n.get("summary") or n.get("shortDescription") or "").strip()

    raw_desc = (n.get("description") or "").strip()
    addl     = (n.get("additionalInfoLink") or "").strip()

    # Store the description fetch URL so we can retrieve it later without
    # burning quota on this pass.
    if _is_url(raw_desc):
        description_url  = raw_desc
        description_text = summary_raw if not _is_url(summary_raw) else ""
    else:
        description_url  = addl if _is_url(addl) else ""
        # raw_desc is inline text — use it directly
        inline = raw_desc if raw_desc and not _is_url(raw_desc) else ""
        description_text = " ".join(filter(None, [
            summary_raw if not _is_url(summary_raw) else "",
            inline,
        ])).strip()

    # organization
    org_obj  = n.get("organization") or {}
    if isinstance(org_obj, dict):
        org_name = (org_obj.get("name") or "").strip()
        org_code = (org_obj.get("code") or "").strip()
    else:
        org_name = (n.get("organizationName") or "").strip()
        org_code = (n.get("organizationCode") or "").strip()

    parent_path = (n.get("fullParentPathName") or "").strip()
    if not org_name:
        org_name = _org_from_parent_path(parent_path)

    # type
    t = n.get("type") or {}
    if isinstance(t, dict):
        base_type = (t.get("baseType") or "").strip()
        type_name = (t.get("name")     or "").strip()
    else:
        base_type = (n.get("baseType") or "").strip()
        type_name = (n.get("type")     or "").strip()

    posted   = (n.get("postedDate") or n.get("publishDate") or "").strip()
    response = (
        n.get("responseDeadLine")
        or n.get("reponseDeadLine")   # SAM typo in field name
        or n.get("responseDate")
        or n.get("closeDate")
        or ""
    )
    response = str(response).strip()

    # place of performance / state
    state_val = ""
    pop = n.get("placeOfPerformance") or {}
    if isinstance(pop, dict):
        st = pop.get("state") or pop.get("stateCode") or ""
        if isinstance(st, dict):
            state_val = (st.get("code") or st.get("name") or "").strip()
        else:
            state_val = str(st).strip()

    # NAICS
    naics_list: list[str] = []
    nc = n.get("naicsCode")
    if nc:
        naics_list.append(str(nc).strip())
    for x in (n.get("naics") or []):
        if isinstance(x, str):
            naics_list.append(x.strip())
        elif isinstance(x, dict):
            naics_list.append((x.get("code") or x.get("value") or "").strip())

    # PSC
    psc_list: list[str] = []
    cc = n.get("classificationCode")
    if isinstance(cc, str):
        psc_list.append(cc.strip())
    elif isinstance(cc, list):
        psc_list.extend(str(x).strip() for x in cc if x)

    # SAM.gov v2 API uses typeOfSetAside (code) and typeOfSetAsideDescription
    # (label). The older field name "setAside" is also checked as a fallback.
    # We prefer the short code for filtering; store description as a bonus.
    set_aside = (
        n.get("typeOfSetAside")
        or n.get("typeOfSetAsideDescription")
        or (n.get("solicitation") or {}).get("setAside")
        or (n.get("solicitation") or {}).get("typeOfSetAside")
        or n.get("setAside")
        or ""
    ).strip()
    source_url = (
        n.get("publicLink") or n.get("permalink")
        or n.get("uiLink")  or n.get("additionalInfoLink")
        or ""
    ).strip()
    ui_link = (n.get("uiLink") or "").strip()

    return {
        "id":                  notice_id,
        "source":              "sam",
        "source_url":          source_url,
        "ui_link":             ui_link,
        "title":               title,
        "summary":             summary_raw if not _is_url(summary_raw) else "",
        "solicitation_number": (n.get("solicitationNumber") or "").strip(),
        "organization_name":   org_name,
        "organization_code":   org_code,
        "full_parent_path_name": parent_path,
        "base_type":           base_type,
        "type":                type_name,
        "posted_date":         posted,
        "response_date":       response,
        "jurisdiction":        (n.get("jurisdiction") or "").strip(),
        "state":               state_val,
        "naics":               ";".join(x for x in naics_list if x),
        "psc":                 ";".join(x for x in psc_list if x),
        "set_aside":           set_aside,
        "place_of_performance": state_val,
        "budget_low":          "",
        "budget_high":         str((n.get("award") or {}).get("amount") or "").strip(),
        "additional_info_link": addl,
        "description_url":     description_url,   # ← new: URL to fetch later
        "resource_links":      "",
        "url_pdf":             "",
        "description_text":    description_text,
    }


# ── main pull ─────────────────────────────────────────────────────────────────

def _debug_notice(n: dict) -> None:
    """Print all top-level keys and set-aside-related values from a raw notice."""
    print("  Top-level keys:", sorted(n.keys()))
    for key in ("setAside", "typeOfSetAside", "typeOfSetAsideDescription", "solicitation"):
        val = n.get(key)
        if val is not None:
            print(f"  {key}: {repr(str(val)[:120])}")
    sol = n.get("solicitation") or {}
    if isinstance(sol, dict):
        for key in ("setAside", "typeOfSetAside", "typeOfSetAsideDescription"):
            val = sol.get(key)
            if val is not None:
                print(f"  solicitation.{key}: {repr(str(val)[:120])}")
    print()


def fetch_to_csv(posted_from: str,
                 posted_to: str,
                 limit: int = 100,
                 max_records: int = MAX_RECORDS_HARD_CAP,
                 naics: str | None = None,
                 state: str | None = None,
                 ptypes: list[str] | None = None,
                 sleep: float = 0.3,
                 debug_fields: bool = False) -> None:
    """
    Pull notice metadata from SAM.gov and write to rfp.csv.
    Does NOT fetch description text — run --fetch-descriptions separately.
    Each page of 100 records costs exactly 1 API call.
    """
    if not API_KEY:
        raise SystemExit("ERROR: SAM_API_KEY not set. Put it in .env or environment.")

    if ptypes is None:
        ptypes = DEFAULT_PTYPES

    print(f"Notice type filter (ptype): {ptypes}")
    print("Description fetching: DEFERRED — run --fetch-descriptions after this completes.")

    rows:         list[dict] = []
    offset        = 0
    total_records = None

    while True:
        params = build_params(posted_from, posted_to, limit, offset, naics, state, ptypes)
        print(f"Fetching offset={offset} ...")

        resp = _get_with_retry(BASE_URL, params)

        if resp is None:
            print(f"All retries failed at offset={offset}. Saving {len(rows)} rows collected so far.")
            _save_rows(rows)
            return

        if resp.status_code != 200:
            print(f"WARN: status {resp.status_code} at offset={offset}: {resp.text[:400]}")
            _save_rows(rows)
            return

        data = resp.json()
        if total_records is None:
            total_records = data.get("totalRecords") or 0
            print(f"totalRecords reported by API: {total_records}")

        items = data.get("opportunitiesData") or data.get("data") or []
        if not items:
            print("No items on this page — done.")
            break

        for n in items:
            if debug_fields and len(rows) < 3:
                print(f"  [debug notice {len(rows)}]")
                _debug_notice(n)
            rows.append(flatten_notice(n))
            if len(rows) >= max_records:
                break

        print(f"  Got {len(items)} items this page; accumulated {len(rows)}/{max_records}")

        if len(rows) >= max_records:
            break

        offset += limit
        if total_records is not None and offset >= total_records:
            break

        time.sleep(sleep)

    _save_rows(rows)


# ── description fetch pass ────────────────────────────────────────────────────

def fetch_descriptions(sleep: float = 1.0, max_fetch: int = 200) -> None:
    """
    Second-pass: read rfp.csv, fetch description_text for rows that have a
    description_url but empty description_text. Saves progress after each fetch.

    Run this on a separate day from the main pull to avoid burning quota on both.
    Each description fetch costs 1 API call. With sleep=1.0, 100 fetches ≈ 2 min.

    Args:
        sleep:     seconds to wait between API calls (default 1.0 — be conservative)
        max_fetch: max descriptions to fetch in this run (default 200)
    """
    if not API_KEY:
        raise SystemExit("ERROR: SAM_API_KEY not set.")

    if not OUT_FILE.exists():
        raise SystemExit(f"rfp.csv not found. Run main pull first.")

    df = pd.read_csv(OUT_FILE, dtype=str).fillna("")

    # Ensure the description_url column exists (older pulls won't have it)
    if "description_url" not in df.columns:
        raise SystemExit(
            "description_url column not found in rfp.csv.\n"
            "This column was added in the latest version of ingest_sam.py.\n"
            "Re-run the main pull to get the URL column, then run --fetch-descriptions."
        )

    # Rows that have a URL to fetch but no description yet
    needs_fetch = df[
        df["description_url"].str.strip().str.len().gt(0) &
        df["description_text"].str.strip().str.len().eq(0)
    ].copy()

    total_needing = len(needs_fetch)
    to_fetch      = needs_fetch.head(max_fetch)
    print(f"Rows needing description fetch: {total_needing}")
    print(f"Will fetch up to {max_fetch} this run.")
    print(f"Sleeping {sleep}s between calls — {len(to_fetch)} fetches ≈ {len(to_fetch)*sleep:.0f}s")
    print()

    fetched = 0
    failed  = 0

    for idx, row in to_fetch.iterrows():
        desc_url = row["description_url"].strip()
        params   = {"api_key": API_KEY}

        try:
            resp = requests.get(desc_url, params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            print(f"  [{idx}] network error: {exc}")
            failed += 1
            time.sleep(sleep)
            continue

        if resp.status_code == 429:
            print(f"  [{idx}] 429 rate-limited — stopping. Saving progress.")
            break
        if resp.status_code != 200:
            print(f"  [{idx}] HTTP {resp.status_code} — skipping.")
            failed += 1
            time.sleep(sleep)
            continue

        raw = resp.text or ""
        desc = raw
        try:
            data = resp.json()
            if isinstance(data, dict) and "description" in data:
                desc = data.get("description", "") or ""
        except Exception:
            pass

        desc = _clean_html(desc)

        if desc:
            # Prepend any existing inline text (summary etc.)
            existing = df.at[idx, "description_text"].strip()
            df.at[idx, "description_text"] = (
                (existing + " " + desc).strip() if existing else desc
            )
            fetched += 1
            print(f"  [{idx}] OK  {len(desc):5d} chars  {row['title'][:60]}")
        else:
            print(f"  [{idx}] empty response for {row['title'][:60]}")
            failed += 1

        # Save after every fetch so a crash/quota-hit doesn't lose progress
        df.to_csv(OUT_FILE, index=False)
        time.sleep(sleep)

    print()
    print(f"Done. Fetched: {fetched}  Failed/empty: {failed}")
    remaining = total_needing - fetched
    if remaining > 0:
        print(f"Still needing fetch: {remaining} — run --fetch-descriptions again tomorrow.")
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved to {OUT_FILE.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch RFPs from SAM.gov into rfp.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow (designed for tight daily API quotas):

  Step 1 — pull metadata (1 API call per 100 records, no description fetches):
    python ingest_sam.py --posted-from 2025-06-01 --posted-to 2026-03-24 \\
        --limit 100 --max-records 1000 --ptypes o,p,r

  Step 2 — fetch descriptions on a subsequent day (1 API call per row):
    python ingest_sam.py --fetch-descriptions --max-fetch 200

  Repeat Step 2 daily until all descriptions are filled.
""",
    )
    p.add_argument("--posted-from",  help="YYYY-MM-DD  (required unless --fetch-descriptions)")
    p.add_argument("--posted-to",    help="YYYY-MM-DD  (required unless --fetch-descriptions)")
    p.add_argument("--limit",        type=int, default=100, help="records per page (<=1000)")
    p.add_argument("--max-records",  type=int, default=MAX_RECORDS_HARD_CAP,
                   help=f"max total records to fetch (default {MAX_RECORDS_HARD_CAP})")
    p.add_argument("--naics",        type=str, default=None, help="e.g. 541620")
    p.add_argument("--state",        type=str, default=None, help="two-letter state code")
    p.add_argument(
        "--ptypes", type=str, default=",".join(DEFAULT_PTYPES),
        help=(
            "Comma-separated notice type codes. "
            "o=Solicitation, p=Presolicitation, r=Sources Sought, "
            "k=Combined Synopsis/Solicitation, s=Special Notice. "
            f"Default: {','.join(DEFAULT_PTYPES)}"
        ),
    )
    p.add_argument("--fetch-descriptions", action="store_true",
                   help="Second-pass: fill in description_text from stored description_url values.")
    p.add_argument("--max-fetch", type=int, default=200,
                   help="Max descriptions to fetch per run (default 200).")
    p.add_argument("--desc-sleep", type=float, default=1.0,
                   help="Seconds to sleep between description fetches (default 1.0).")
    p.add_argument("--debug-fields", action="store_true",
                   help="Print raw JSON keys for first 3 notices — use to diagnose missing fields.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.fetch_descriptions:
        fetch_descriptions(sleep=args.desc_sleep, max_fetch=args.max_fetch)
    else:
        if not args.posted_from or not args.posted_to:
            raise SystemExit("ERROR: --posted-from and --posted-to are required for the main pull.")
        fetch_to_csv(
            posted_from=args.posted_from,
            posted_to=args.posted_to,
            limit=args.limit,
            max_records=args.max_records,
            naics=args.naics,
            state=args.state,
            ptypes=[t.strip() for t in args.ptypes.split(",") if t.strip()],
            debug_fields=args.debug_fields,
        )
