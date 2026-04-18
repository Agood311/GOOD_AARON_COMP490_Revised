# main.py
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer

DATA_FILE = Path("rfp.csv")
TEMPLATES_DIR = Path("templates")

app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# globals set during startup
df = None
texts = None
tfidf_vectorizer = None
tfidf_matrix = None
embed_model = None
embeddings = None


def init_indexes():
    global df, texts, tfidf_vectorizer, tfidf_matrix, embed_model, embeddings

    if not DATA_FILE.exists():
        raise RuntimeError("rfp.csv not found. Run ingest_sam.py first.")

    df_loaded = pd.read_csv(DATA_FILE, dtype=str).fillna("")

    if "id" not in df_loaded.columns:
        raise RuntimeError("rfp.csv must contain an 'id' column")

    df_loaded = df_loaded.drop_duplicates(subset=["id"]).reset_index(drop=True)

    for col in [
        "title",
        "description_text",
        "organization_name",
        "full_parent_path_name",
        "response_date",
        "ui_link",
        "source_url",
        "additional_info_link",
        "naics",
        "psc",
        "state",
        "place_of_performance",
    ]:
        if col not in df_loaded.columns:
            df_loaded[col] = ""

    # build combined text field from available structured fields
    combined_texts = []
    for _, row in df_loaded.iterrows():
        pieces = []

        title = row.get("title", "").strip()
        if title:
            pieces.append(title)

        org = row.get("organization_name", "").strip()
        if org:
            pieces.append(org)

        parent = row.get("full_parent_path_name", "").strip()
        if parent:
            pieces.append(parent)

        naics = row.get("naics", "").strip()
        if naics:
            pieces.append(f"NAICS {naics}")

        psc = row.get("psc", "").strip()
        if psc:
            pieces.append(f"PSC {psc}")

        desc = row.get("description_text", "").strip()
        if desc:
            pieces.append(desc)

        combined_texts.append(" ".join(pieces))

    df_loaded["combined_text"] = combined_texts

    mask = df_loaded["combined_text"].str.len() > 0
    df_loaded = df_loaded[mask].reset_index(drop=True)

    if df_loaded.empty:
        raise RuntimeError("No usable rows with text found in rfp.csv")

    combined = df_loaded["combined_text"].tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(combined)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(
        combined,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    df = df_loaded
    texts = combined
    tfidf_vectorizer = vectorizer
    tfidf_matrix = X
    embed_model = model
    embeddings = emb

    print(f"Initialized indexes on {len(df)} RFPs")


def normalize_scores(scores):
    """Min-max normalize to [0, 1]."""
    if len(scores) == 0:
        return scores
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def run_tfidf(query, top_k=100):
    q_vec = tfidf_vectorizer.transform([query])
    sims = linear_kernel(q_vec, tfidf_matrix).ravel()
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_scores = sims[top_idx]
    mask = top_scores > 0
    return top_idx[mask], normalize_scores(top_scores[mask])


def run_semantic(query, top_k=100):
    q_vec = embed_model.encode([query], normalize_embeddings=True)[0]
    sims = embeddings @ q_vec
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_scores = sims[top_idx]
    mask = top_scores > 0
    return top_idx[mask], normalize_scores(top_scores[mask])


def run_hybrid(query, alpha=0.5, top_k=100):
    """
    Hybrid search: normalize TF-IDF and semantic scores across the full corpus
    before combining. This preserves each method's confidence distribution
    and makes the fusion weight meaningful.
    """
    q_tfidf = tfidf_vectorizer.transform([query])
    tfidf_raw = linear_kernel(q_tfidf, tfidf_matrix).ravel()

    q_sem = embed_model.encode([query], normalize_embeddings=True)[0]
    sem_raw = embeddings @ q_sem

    # zero out low-confidence matches before normalizing
    tfidf_raw = np.where(tfidf_raw > 0.0, tfidf_raw, 0.0)
    sem_raw   = np.where(sem_raw > 0.25, sem_raw, 0.0)

    tfidf_norm = normalize_scores(tfidf_raw)
    sem_norm   = normalize_scores(sem_raw)

    hybrid_raw = alpha * tfidf_norm + (1.0 - alpha) * sem_norm

    top_idx = np.argsort(hybrid_raw)[::-1][:top_k]

    out = df.iloc[top_idx].copy()
    out["score"]          = hybrid_raw[top_idx]
    out["tfidf_score"]    = tfidf_norm[top_idx]
    out["semantic_score"] = sem_norm[top_idx]

    return out


def run_search(query, mode, top_k=50, alpha=0.5):
    if mode == "tfidf":
        idx, scores = run_tfidf(query, top_k)
        out = df.iloc[idx].copy()
        out["score"] = scores
        return out
    elif mode == "semantic":
        idx, scores = run_semantic(query, top_k)
        out = df.iloc[idx].copy()
        out["score"] = scores
        return out
    else:
        return run_hybrid(query, alpha=alpha, top_k=top_k)


def parse_date(date_str):
    if not date_str:
        return None
    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%Y%m%d"]:
        try:
            return datetime.strptime(date_str.split("T")[0].split(" ")[0], fmt)
        except ValueError:
            continue
    return None


# NAICS code labels for the dropdown
NAICS_LABELS = {
    "236210": "Industrial Building Construction",
    "236220": "Commercial & Institutional Building Construction",
    "237110": "Water & Sewer Line Construction",
    "237310": "Highway, Street & Bridge Construction",
    "238210": "Electrical Contractors",
    "238220": "Plumbing, Heating & AC Contractors",
    "238910": "Site Preparation Contractors",
    "311999": "Food Manufacturing",
    "332510": "Hardware Manufacturing",
    "332911": "Industrial Valve Manufacturing",
    "332912": "Fluid Power Valve Manufacturing",
    "332991": "Ball & Roller Bearing Manufacturing",
    "332996": "Fabricated Pipe & Fitting Manufacturing",
    "332999": "Misc. Fabricated Metal Products",
    "333415": "AC, Refrigeration & Heating Equipment",
    "333612": "Speed Changer & Drive Manufacturing",
    "333914": "Measuring & Dispensing Pump Manufacturing",
    "334220": "Radio & TV Broadcasting Equipment",
    "334290": "Other Communications Equipment",
    "334412": "Printed Circuit Assembly Manufacturing",
    "334512": "Automatic Environmental Control Manufacturing",
    "334519": "Other Measuring & Controlling Instruments",
    "335220": "Major Household Appliance Manufacturing",
    "335312": "Motor & Generator Manufacturing",
    "335313": "Switchgear & Switchboard Apparatus",
    "335314": "Relay & Industrial Control Manufacturing",
    "335999": "Misc. Electrical Equipment Manufacturing",
    "336212": "Truck Trailer Manufacturing",
    "336320": "Motor Vehicle Electrical & Electronic Equipment",
    "336330": "Motor Vehicle Steering & Suspension",
    "336413": "Other Aircraft Parts & Equipment",
    "336415": "Guided Missile & Space Vehicle Propulsion",
    "336611": "Ship Building & Repairing",
    "336612": "Boat Building",
    "336992": "Military Armored Vehicle Manufacturing",
    "339991": "Gasket, Packing & Sealing Device Manufacturing",
    "484110": "General Freight Trucking, Local",
    "484121": "General Freight Trucking, Long-Distance",
    "511210": "Software Publishers",
    "517110": "Wired Telecommunications Carriers",
    "517312": "Wireless Telecommunications Carriers",
    "518210": "Computing Infrastructure & Data Processing",
    "519130": "Internet Publishing & Web Search Portals",
    "541110": "Offices of Lawyers",
    "541199": "Other Legal Services",
    "541211": "Offices of Certified Public Accountants",
    "541310": "Architectural Services",
    "541330": "Engineering Services",
    "541340": "Drafting Services",
    "541380": "Testing Laboratories",
    "541511": "Custom Computer Programming Services",
    "541512": "Computer Systems Design Services",
    "541513": "Computer Facilities Management Services",
    "541519": "Other Computer Related Services",
    "541611": "Admin Management & General Mgmt Consulting",
    "541612": "Human Resources Consulting",
    "541613": "Marketing Consulting Services",
    "541614": "Process & Logistics Consulting",
    "541618": "Other Management Consulting Services",
    "541620": "Environmental Consulting Services",
    "541690": "Other Scientific & Technical Consulting",
    "541715": "R&D in Physical, Engineering & Life Sciences",
    "541720": "R&D in Social Sciences & Humanities",
    "541810": "Advertising Agencies",
    "541990": "Other Professional & Technical Services",
    "561110": "Office Administrative Services",
    "561210": "Facilities Support Services",
    "561311": "Employment Placement Agencies",
    "561320": "Temporary Help Services",
    "561410": "Document Preparation Services",
    "561421": "Telephone Answering Services",
    "561499": "Other Business Support Services",
    "561510": "Travel Agencies",
    "561612": "Security Guards & Patrol Services",
    "561621": "Security Systems Services",
    "561720": "Janitorial Services",
    "561730": "Landscaping Services",
    "561740": "Carpet & Upholstery Cleaning Services",
    "561790": "Other Services to Buildings & Dwellings",
    "561910": "Packaging & Labeling Services",
    "561990": "Other Support Services",
    "562111": "Solid Waste Collection",
    "562910": "Remediation Services",
    "611430": "Professional & Management Development Training",
    "611710": "Educational Support Services",
    "621111": "Offices of Physicians",
    "621330": "Offices of Mental Health Practitioners",
    "621399": "Other Outpatient Care Centers",
    "621999": "Other Ambulatory Health Care Services",
    "622110": "General Medical & Surgical Hospitals",
    "711519": "Independent Artists, Writers & Performers",
    "721110": "Hotels & Motels",
    "811212": "Computer & Office Machine Repair",
    "811310": "Commercial & Industrial Machinery Repair",
    "923120": "Public Finance Activities",
    "928110": "National Security",
}

SET_ASIDE_LABELS = {
    "SBA":     "Small Business",
    "8A":      "8(a) Program",
    "8AN":     "8(a) Sole Source",
    "HZC":     "HUBZone",
    "HZCS":    "HUBZone Sole Source",
    "SDVOSBC": "Service-Disabled Veteran-Owned Small Business",
    "SDVOSBS": "SDVOSB Sole Source",
    "WOSB":    "Women-Owned Small Business",
    "WOSBSS":  "WOSB Sole Source",
    "EDWOSB":  "Economically Disadvantaged WOSB",
    "EDWOSBSS":"EDWOSB Sole Source",
    "VSA":     "Veteran-Owned Small Business",
    "VSS":     "VSB Sole Source",
    "BICiv":   "Buy Indian Act — Civilian",
    "IEE":     "Indian Economic Enterprise",
    "ISBEE":   "Indian Small Business Economic Enterprise",
    "LAS":     "Local Area Set-Aside",
    "RSB":     "Emerging Small Business",
    "NONE":    "No Set-Aside",
}


def apply_filters(df_res, state=None, date_from=None, date_to=None, naics=None, set_aside=None):
    if df_res.empty:
        return df_res

    if state:
        state = state.upper().strip()
        df_res = df_res[df_res["state"].str.upper().str.strip() == state]

    # NAICS prefix match so "5416" catches 541620, 541611, etc.
    if naics:
        naics = naics.strip()
        df_res = df_res[df_res["naics"].str.split(";").apply(
            lambda codes: any(c.strip().startswith(naics) for c in codes)
        )]

    if set_aside:
        set_aside = set_aside.strip().upper()
        df_res = df_res[df_res["set_aside"].str.strip().str.upper() == set_aside]

    if date_from or date_to:
        date_from_dt = parse_date(date_from) if date_from else None
        date_to_dt   = parse_date(date_to)   if date_to   else None

        def in_date_range(row_date):
            dt = parse_date(row_date)
            if dt is None:
                return True
            if date_from_dt and dt < date_from_dt:
                return False
            if date_to_dt and dt > date_to_dt:
                return False
            return True

        mask = df_res["response_date"].apply(in_date_range)
        df_res = df_res[mask]

    return df_res


def get_available_states():
    if df is None:
        return []
    states = df["state"].str.upper().str.strip().unique()
    return sorted(s for s in states if s)


def get_naics_options(top_n=20):
    if df is None:
        return []
    counts = {}
    for cell in df["naics"]:
        for code in cell.split(";"):
            code = code.strip()
            if code:
                counts[code] = counts.get(code, 0) + 1
    top = sorted(counts, key=lambda c: counts[c], reverse=True)[:top_n]
    options = []
    for code in sorted(top):
        name = NAICS_LABELS.get(code)
        label = f"{code} — {name}" if name else code
        options.append({"code": code, "label": label, "count": counts[code]})
    return options


def get_set_aside_options():
    if df is None:
        return []
    codes = df["set_aside"].str.strip().str.upper().unique()
    options = []
    for code in sorted(c for c in codes if c):
        name = SET_ASIDE_LABELS.get(code, code)
        options.append({"code": code, "label": f"{name} ({code})"})
    return options


def format_date(date_str):
    if not date_str:
        return ""
    dt = parse_date(date_str)
    if dt:
        return dt.strftime("%b %d, %Y")
    return date_str


def format_results(df_res):
    results = []
    for _, row in df_res.iterrows():
        title = (row.get("title") or "").strip() or "(no title)"

        agency = (row.get("organization_name") or "").strip()
        if not agency:
            agency = (row.get("full_parent_path_name") or "").strip()

        deadline = format_date((row.get("response_date") or "").strip())

        url = ""
        for col in ("source_url", "ui_link", "additional_info_link"):
            val = row.get(col)
            if isinstance(val, str) and val.startswith("http"):
                url = val
                break

        text = (
            row.get("description_text")
            or row.get("combined_text")
            or ""
        )
        text = str(text).replace("\n", " ")
        snippet = text[:400]

        score_val = float(row.get("score", 0.0) or 0.0)

        result_dict = {
            "id":        row.get("id"),
            "title":     title,
            "agency":    agency,
            "deadline":  deadline,
            "url":       url,
            "snippet":   snippet,
            "score":     f"{score_val:.4f}",
            "state":     (row.get("state") or "").strip(),
            "naics":     (row.get("naics") or "").strip(),
            "set_aside": (row.get("set_aside") or "").strip(),
        }

        if "tfidf_score" in row:
            result_dict["tfidf_score"] = f"{row['tfidf_score']:.4f}"
            result_dict["semantic_score"] = f"{row['semantic_score']:.4f}"

        results.append(result_dict)
    return results


@app.on_event("startup")
def on_startup():
    init_indexes()


@app.get("/", response_class=HTMLResponse)
async def search_page(
    request: Request,
    q: str = "",
    mode: str = "hybrid",
    alpha: float = 0.7,
    page: int = 1,
    per_page: int = 10,
    state: str = "",
    date_from: str = "",
    date_to: str = "",
    naics: str = "",
    set_aside: str = "",
):
    q         = (q or "").strip()
    mode      = (mode or "hybrid").lower()
    naics     = (naics or "").strip()
    set_aside = (set_aside or "").strip().upper()
    if mode not in ("tfidf", "semantic", "hybrid"):
        mode = "hybrid"
    alpha    = max(0.0, min(1.0, alpha))
    page     = max(page, 1)
    per_page = max(per_page, 1)

    results = []
    total = 0
    total_pages = 0
    available_states  = get_available_states()
    naics_options     = get_naics_options()
    set_aside_options = get_set_aside_options()

    if q:
        df_res = run_search(q, mode, top_k=100, alpha=alpha)
        df_res = apply_filters(
            df_res,
            state=state,
            date_from=date_from,
            date_to=date_to,
            naics=naics,
            set_aside=set_aside,
        )

        all_results = format_results(df_res)
        total = len(all_results)
        total_pages = (total + per_page - 1) // per_page

        start = (page - 1) * per_page
        end   = start + per_page
        results = all_results[start:end]

    return templates.TemplateResponse(
        "search.html",
        {
            "request":           request,
            "query":             q,
            "mode":              mode,
            "results":           results,
            "page":              page,
            "per_page":          per_page,
            "total":             total,
            "total_pages":       total_pages,
            "state":             state,
            "date_from":         date_from,
            "date_to":           date_to,
            "naics":             naics,
            "set_aside":         set_aside,
            "available_states":  available_states,
            "naics_options":     naics_options,
            "set_aside_options": set_aside_options,
        },
    )


@app.post("/refresh-local", response_class=HTMLResponse)
async def refresh_local():
    """Reload indexes from rfp.csv."""
    init_indexes()
    return RedirectResponse("/", status_code=303)
