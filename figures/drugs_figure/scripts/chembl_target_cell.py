#!/usr/bin/env python
# coding: utf-8

# # Drug targets in the cell
# 
# Takes a list of drugs (a plate-map CSV, a text list, or names/ChEMBL IDs), finds each drug in **ChEMBL**, pulls its mechanism-of-action targets, looks those proteins up in **UniProt**, and plots where in the cell they act.

# ## Setup
# Requires `requests`, `matplotlib` and `numpy` (`pip install requests matplotlib numpy`). API calls retry automatically on rate limits and server errors.

# In[1]:


import csv
import math
import os
import pathlib
import re
import sys
import textwrap
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from IPython.display import Image, display
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from notebook_init_utils.notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT = "https://rest.uniprot.org/uniprotkb"
S = requests.Session()
S.headers["Accept"] = "application/json"
S.mount(
    "https://",
    requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
        )
    ),
)


def get(url, **params):
    r = S.get(url, params=params, timeout=90)
    r.raise_for_status()
    return r.json()


# In[2]:


# Input: a plate-map CSV (drug names in COLUMN), a text file (one drug per line),
# (a string or pathlib.Path), or a Python list of names / ChEMBL IDs, e.g. ["selumetinib", "imatinib", "CHEMBL25"]
INPUT = pathlib.Path(f"{root_dir}/config/platemaps/platemap2.csv")
COLUMN = "Treatment"  # column holding drug names when INPUT is a plate-map CSV
ALL_EVIDENCE = False  # False = only experimentally supported GO terms (recommended)
WORKERS = 6  # parallel API requests

# output file paths
cell_structures_targets_image_out_file_path = pathlib.Path(
    f"{root_dir}/figures/drugs_figure/figures/"
    "drug_targets_by_cellular_compartment_from_uniprot_go_annotations.png"
)
cell_protein_targets_results_out_file_path = pathlib.Path(
    f"{root_dir}/figures/drugs_figure/results/"
    "drug_target_proteins_subcellular_location_and_process_results.csv"
)
cell_protein_targets_image_out_file_path = pathlib.Path(
    f"{root_dir}/figures/drugs_figure/figures/"
    "drug_target_proteins_by_primary_subcellular_location_and_process.png"
)
cell_structures_targets_image_out_file_path.parent.mkdir(parents=True, exist_ok=True)
cell_protein_targets_results_out_file_path.parent.mkdir(parents=True, exist_ok=True)
cell_protein_targets_image_out_file_path.parent.mkdir(parents=True, exist_ok=True)


# ## Drug name → ChEMBL ID
# Tries an exact preferred name, then an exact synonym, then free-text search. Prefers the parent compound (not a salt) and the most advanced clinical phase. ChEMBL IDs are accepted directly.


# ## Drug name → ChEMBL ID
# Tries an exact preferred name, then an exact synonym, then free-text search. Prefers the parent compound (not a salt) and the most advanced clinical phase. ChEMBL IDs are accepted directly.

# In[3]:


def resolve_drug(query):
    """Return (chembl_id, pref_name) for a drug name/synonym/ChEMBL ID, or (None, None)."""
    q = query.strip()
    if q.upper().startswith("CHEMBL"):
        m = get(f"{CHEMBL}/molecule/{q.upper()}.json")
        return m["molecule_chembl_id"], m.get("pref_name") or q
    # 1) exact preferred name, 2) exact synonym, 3) free-text search
    for params in (
        {"pref_name__iexact": q},
        {"molecule_synonyms__molecule_synonym__iexact": q},
    ):
        hits = get(f"{CHEMBL}/molecule.json", limit=20, **params)["molecules"]
        if hits:
            break
    else:
        hits = get(f"{CHEMBL}/molecule/search.json", q=q, limit=10)["molecules"]
    if not hits:
        return None, None

    # prefer the parent compound and the most advanced clinical phase
    def score(m):
        is_parent = (m.get("molecule_hierarchy") or {}).get("parent_chembl_id") == m[
            "molecule_chembl_id"
        ]
        return (is_parent, float(m.get("max_phase") or 0))

    best = max(hits, key=score)
    parent = (best.get("molecule_hierarchy") or {}).get("parent_chembl_id") or best[
        "molecule_chembl_id"
    ]
    return parent, best.get("pref_name") or q


# ## ChEMBL mechanisms and targets
# Mechanisms are often filed under a salt form, so they are queried by parent **and** by ID. A ChEMBL target can be a single protein, a complex or a family; each component's UniProt accession is returned.


# ## ChEMBL mechanisms and targets
# Mechanisms are often filed under a salt form, so they are queried by parent **and** by ID. A ChEMBL target can be a single protein, a complex or a family; each component's UniProt accession is returned.

# In[4]:


def get_mechanisms(chembl_id):
    """Mechanisms are often stored on a salt form, so query by parent AND by id."""
    seen, mechs = set(), []
    for key in ("parent_molecule_chembl_id", "molecule_chembl_id"):
        for m in get(f"{CHEMBL}/mechanism.json", **{key: chembl_id, "limit": 1000})[
            "mechanisms"
        ]:
            if m["mec_id"] not in seen and m.get("target_chembl_id"):
                seen.add(m["mec_id"])
                mechs.append(m)
    return mechs


@lru_cache(maxsize=None)
def target_accessions(target_chembl_id):
    """A ChEMBL target can be a single protein, a complex, or a family."""
    t = get(f"{CHEMBL}/target/{target_chembl_id}.json")
    accs = tuple(
        c["accession"] for c in t.get("target_components", []) if c.get("accession")
    )
    return t.get("pref_name"), t.get("target_type"), t.get("organism"), accs


# ## UniProt entries and GO Cellular Component terms


# ## UniProt entries and GO Cellular Component terms

# In[5]:


@lru_cache(maxsize=None)
def uniprot_entry(acc):
    return get(f"{UNIPROT}/{acc}.json")


@lru_cache(maxsize=None)
def uniprot_info(acc):
    d = uniprot_entry(acc)
    gene = (d.get("genes") or [{}])[0].get("geneName", {}).get("value", acc)
    go_cc = []
    for x in d.get("uniProtKBCrossReferences", []):
        if x["database"] == "GO":
            props = {p["key"]: p["value"] for p in x.get("properties", [])}
            term = props.get("GoTerm", "")
            if term.startswith("C:"):  # C = cellular component
                go_cc.append((x["id"], term[2:], props.get("GoEvidenceType", "")))
    return gene, tuple(go_cc)


# ## GO term → cell compartment (GO plot)
# Keyword matching on GO term names; first match wins. `EXPERIMENTAL` lists the GO evidence codes kept when `ALL_EVIDENCE = False`.


# ## GO term → cell compartment (GO plot)
# Keyword matching on GO term names; first match wins. `EXPERIMENTAL` lists the GO evidence codes kept when `ALL_EVIDENCE = False`.

# In[6]:


# Keyword matching on GO CC term names. Order matters (first match wins).
COMPARTMENTS = [
    ("Extracellular", ["extracellular", "secreted"]),
    (
        "Plasma membrane",
        [
            "plasma membrane",
            "cell surface",
            "focal adhesion",
            "cell junction",
            "adherens",
            "ruffle",
            "lamellipodium",
            "filopodium",
            "caveola",
            "synap",
            "receptor complex",
            "axon",
            "dendrite",
        ],
    ),
    ("Nucleus", ["nucle", "chromatin", "chromosome", "spliceosom"]),
    ("Mitochondrion", ["mitochondri"]),
    ("Endoplasmic reticulum", ["endoplasmic reticulum"]),
    ("Golgi", ["golgi"]),
    ("Endosome / Lysosome", ["endosom", "lysosom", "vacuol", "vesicle"]),
    (
        "Cytoskeleton",
        ["cytoskelet", "microtubule", "actin", "centrosome", "spindle", "cilium"],
    ),
    ("Cytoplasm", ["cytoplasm", "cytosol", "perinuclear", "intracellular"]),
]
COMP_NAMES = [c for c, _ in COMPARTMENTS]

# GO evidence codes from experiments vs. computational inference (IEA, ISS, ...)
EXPERIMENTAL = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HDA", "HTP", "HMP"}


def compartment_of(term_name):
    low = term_name.lower()
    for comp, keys in COMPARTMENTS:
        if any(k in low for k in keys):
            return comp
    return None


# ## Primary location and process (protein map)
# Each protein gets **one** location from UniProt's curated subcellular location: a transmembrane location wins; otherwise the head term of the first listed location; otherwise the most common GO compartment.
#
# Process (card colour) comes from gene-prefix rules, then UniProt keywords.
#
# **✏️ Edit here:** `LOCATION_OVERRIDES`, `PROCESS_GENE_RULES`, `PROCESS_KEYWORD_RULES`.


# ## Primary location and process (protein map)
# Each protein gets **one** location from UniProt's curated subcellular location: a transmembrane location wins; otherwise the head term of the first listed location; otherwise the most common GO compartment.
# 
# Process (card colour) comes from gene-prefix rules, then UniProt keywords.
# 
# **✏️ Edit here:** `LOCATION_OVERRIDES`, `PROCESS_GENE_RULES`, `PROCESS_KEYWORD_RULES`.

# In[7]:


# UniProt's curated "Subcellular location" lists locations with the main one first.
UNIPROT_LOC_RULES = [
    ("Extracellular", ["secreted", "extracellular"]),
    (
        "Plasma membrane",
        [
            "cell membrane",
            "plasma membrane",
            "cell surface",
            "cell projection",
            "cell junction",
            "synapse",
            "apical",
            "basolateral",
            "sarcolemma",
        ],
    ),
    ("Nucleus", ["nucle", "chromosome"]),
    ("Mitochondrion", ["mitochond"]),
    (
        "Endoplasmic reticulum",
        ["endoplasmic reticulum", "sarcoplasmic reticulum", "microsome"],
    ),
    ("Golgi", ["golgi"]),
    ("Endosome / Lysosome", ["lysosom", "endosom", "vacuol", "vesicle", "melanosome"]),
    ("Cytoskeleton", ["cytoskeleton", "centrosome", "microtubule"]),
    ("Cytoplasm", ["cytoplasm", "cytosol"]),
]
# Gene-prefix overrides where the head-term rule misleads (edit freely)
LOCATION_OVERRIDES = {"TUBA": "Cytoskeleton", "TUBB": "Cytoskeleton"}


def _uniprot_comp(text):
    low = text.lower()
    for comp, keys in UNIPROT_LOC_RULES:
        if any(k in low for k in keys):
            return comp
    return None


def primary_location(acc, gene, go_terms):
    """(compartment, UniProt text, is_transmembrane). Rules, in order:
    1) a transmembrane location (single/multi-pass, GPI) wins - that's where it sits;
    2) otherwise the head term of the first listed location ("Cytoplasm, cytosol" -> Cytoplasm);
    3) otherwise the most common compartment among the GO terms."""
    for pre, comp in LOCATION_OVERRIDES.items():
        if gene.startswith(pre):
            return comp, "override", False
    locs = []
    for c in uniprot_entry(acc).get("comments", []):
        if c.get("commentType") == "SUBCELLULAR LOCATION":
            for l in c.get("subcellularLocations", []):
                locs.append(
                    (l["location"]["value"], (l.get("topology") or {}).get("value", ""))
                )
    for val, topo in locs:
        if "pass" in topo.lower() or "gpi" in topo.lower():
            # a bare "Membrane" on a transmembrane protein almost always means the cell membrane
            comp = _uniprot_comp(val) or (
                "Plasma membrane" if val.lower() == "membrane" else None
            )
            if comp:
                return comp, f"{val} [{topo}]", True
    for val, topo in locs:
        head = val.split(",")[0].strip()
        if head.lower() == "membrane":  # too generic, try the next one
            continue
        comp = _uniprot_comp(head)
        if comp:
            return comp, val + (f" [{topo}]" if topo else ""), False
    if go_terms:
        counts = defaultdict(int)
        for comp, *_ in go_terms:
            counts[comp] += 1
        return max(counts, key=counts.get), "GO (most frequent)", False
    return "Cytoplasm", "unknown", False


# Process/pathway assignment: gene-prefix rules first, then UniProt keywords. Edit freely.
PROCESS_GENE_RULES = [
    ("MAPK cascade", ["MAP2K", "MAPK", "BRAF", "RAF1", "ARAF", "KRAS", "NRAS", "HRAS"]),
    ("PI3K–AKT–mTOR", ["PIK3", "AKT", "MTOR", "FKBP1A", "RPTOR", "RICTOR"]),
    (
        "Chromatin & epigenetics",
        ["HDAC", "BRD2", "BRD3", "BRD4", "BRDT", "EZH", "DNMT", "KDM", "SIRT"],
    ),
    ("Protein folding (HSP90)", ["HSP90", "HSPA"]),
    ("Ubiquitin–proteasome", ["CRBN", "VHL", "PSM", "MDM2", "CUL"]),
    ("PKC / Ca²⁺ signalling", ["PRKC"]),
    ("Ion pumps & channels", ["ATP1", "FXYD", "SCN", "KCN", "CACN"]),
    (
        "Non-receptor tyrosine kinases",
        ["ABL", "BCR", "SRC", "JAK", "BTK", "LCK", "FYN"],
    ),
]
PROCESS_KEYWORD_RULES = [
    ("GPCR signalling", ["G protein-coupled receptor"]),
    ("Receptor tyrosine kinases", ["Receptor+Tyrosine-protein kinase"]),
    ("Non-receptor tyrosine kinases", ["Tyrosine-protein kinase"]),
    ("Ion pumps & channels", ["Ion transport", "Ion channel"]),
    ("Chromatin & epigenetics", ["Chromatin regulator"]),
    ("Ubiquitin–proteasome", ["Ubl conjugation pathway"]),
]


def process_of(acc, gene):
    for proc, prefixes in PROCESS_GENE_RULES:
        if any(gene.startswith(p) for p in prefixes):
            return proc
    kws = {k["name"] for k in uniprot_entry(acc).get("keywords", [])}
    for proc, rules in PROCESS_KEYWORD_RULES:
        for rule in rules:
            if all(r in kws for r in rule.split("+")):
                return proc
    return "Other"


# ## Protein display names
# Cards and legends show the protein name with the gene symbol in italics (MEK1 / *MAP2K1*). Genes not listed show their gene symbol.
#
# **✏️ Edit here:** `PROTEIN_NAMES`.


# ## Protein display names
# Cards and legends show the protein name with the gene symbol in italics (MEK1 / *MAP2K1*). Genes not listed show their gene symbol.
# 
# **✏️ Edit here:** `PROTEIN_NAMES`.

# In[8]:


# Common protein names for display (gene symbol -> protein). Unlisted genes show the gene
# symbol (e.g. HDAC1, BRD4 - where protein and gene names coincide). Edit freely.
PROTEIN_NAMES = {
    "MAP2K1": "MEK1",
    "MAP2K2": "MEK2",
    "MAPK1": "ERK2",
    "MAPK3": "ERK1",
    "BRAF": "B-RAF",
    "MTOR": "mTOR",
    "FKBP1A": "FKBP12",
    "PIK3CA": "PI3Kα",
    "PIK3CB": "PI3Kβ",
    "PIK3CD": "PI3Kδ",
    "PIK3CG": "PI3Kγ",
    "HSP90AA1": "HSP90α",
    "HSP90AB1": "HSP90β",
    "CRBN": "Cereblon",
    "PRKCA": "PKCα",
    "PRKCB": "PKCβ",
    "PRKCG": "PKCγ",
    "PRKCD": "PKCδ",
    "PRKCE": "PKCε",
    "PRKCH": "PKCη",
    "PRKCQ": "PKCθ",
    "PRKCZ": "PKCζ",
    "PRKCI": "PKCι",
    "ATP1A1": "NKA α1",
    "ATP1A2": "NKA α2",
    "ATP1A3": "NKA α3",
    "ATP1A4": "NKA α4",
    "ATP1B1": "NKA β1",
    "ATP1B2": "NKA β2",
    "ATP1B3": "NKA β3",
    "FXYD2": "NKA γ",
    "ABL1": "c-ABL",
    "PDGFRA": "PDGFRα",
    "PDGFRB": "PDGFRβ",
    "KIT": "c-KIT",
    "KDR": "VEGFR2",
    "FLT1": "VEGFR1",
    "FLT4": "VEGFR3",
    "MET": "c-MET",
    "IGF1R": "IGF-1R",
    "INSR": "Insulin R",
    "HRH1": "Histamine H1R",
    "EGFR": "EGFR",
}


def protein_label(acc, gene):
    return PROTEIN_NAMES.get(gene, gene)


# ## Manual targets
# `MANUAL_TARGETS` supplies targets for drugs ChEMBL has no mechanism for (research compounds such as ARV-825 and staurosporine). `MANUAL_ADD` adds targets on top of ChEMBL's, e.g. mTOR for rapalogs (ChEMBL records FKBP12, their direct binding partner).
#
# **✏️ Edit here:** `MANUAL_TARGETS`, `MANUAL_ADD`.


# ## Manual targets
# `MANUAL_TARGETS` supplies targets for drugs ChEMBL has no mechanism for (research compounds such as ARV-825 and staurosporine). `MANUAL_ADD` adds targets on top of ChEMBL's, e.g. mTOR for rapalogs (ChEMBL records FKBP12, their direct binding partner).
# 
# **✏️ Edit here:** `MANUAL_TARGETS`, `MANUAL_ADD`.

# In[9]:


# Drugs whose mechanism ChEMBL doesn't annotate. Gene symbols (human). Edit freely.
MANUAL_TARGETS = {
    "arv-825": ["BRD4", "CRBN"],  # BET PROTAC: binds BRD4, recruits cereblon E3 ligase
    # Pan-kinase inhibitor, classically used as a PKC inhibitor. Conventional + novel PKC
    # isoforms listed; atypical PKCs (PRKCZ, PRKCI) are much less sensitive.
    "staurosporine": ["PRKCA", "PRKCB", "PRKCG", "PRKCD", "PRKCE", "PRKCH", "PRKCQ"],
}
# Targets ADDED on top of ChEMBL's mechanisms. Rapalogs bind FKBP12 (ChEMBL's target),
# and the drug-FKBP12 complex then inhibits mTOR (mTORC1), the functional target.
MANUAL_ADD = {
    "everolimus": ["MTOR"],
    "rapamycin": ["MTOR"],
    "sirolimus": ["MTOR"],
    "temsirolimus": ["MTOR"],
}


@lru_cache(maxsize=None)
def accession_for_gene(gene):
    d = get(
        f"{UNIPROT}/search",
        query=f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
        fields="accession",
        size=1,
    )
    res = d.get("results") or []
    return res[0]["primaryAccession"] if res else None


# ## Per-drug processing
# Resolves the drug, collects its targets (deduplicated by UniProt accession), and attaches GO compartments, primary location, protein name and process to each target.


# ## Per-drug processing
# Resolves the drug, collects its targets (deduplicated by UniProt accession), and attaches GO compartments, primary location, protein name and process to each target.

# In[10]:


def process_drug(query, experimental_only):
    out = dict(query=query, chembl_id=None, name=query, targets=[], note="")
    try:
        cid, name = resolve_drug(query)
        if not cid:
            out["note"] = "not found in ChEMBL"
            return out
        out.update(chembl_id=cid, name=name)
        mechs = get_mechanisms(cid)
        manual = MANUAL_TARGETS.get(query.lower()) or MANUAL_TARGETS.get(name.lower())
        if not mechs and manual:
            mechs = [
                dict(
                    target_chembl_id=None,
                    mechanism_of_action="manual target list",
                    action_type="MANUAL",
                    genes=manual,
                )
            ]
            out["note"] = "targets from MANUAL_TARGETS (not in ChEMBL)"
        add = MANUAL_ADD.get(query.lower()) or MANUAL_ADD.get(name.lower())
        if add:
            mechs = mechs + [
                dict(
                    target_chembl_id=None,
                    genes=add,
                    action_type="MANUAL",
                    mechanism_of_action="functional target (MANUAL_ADD)",
                )
            ]
        if not mechs:
            out["note"] = "no mechanism annotated"
            return out
        seen_acc = set()
        for m in mechs:
            if m.get("genes"):
                tname, ttype, org = "manual", "MANUAL", "Homo sapiens"
                accs = tuple(a for a in map(accession_for_gene, m["genes"]) if a)
            else:
                tname, ttype, org, accs = target_accessions(m["target_chembl_id"])
            for acc in accs:
                if acc in seen_acc:  # same protein via another mechanism record
                    continue
                seen_acc.add(acc)
                gene, go_cc = uniprot_info(acc)
                terms = []
                for go_id, term, ev in go_cc:
                    code = ev.split(":")[0]
                    if experimental_only and code not in EXPERIMENTAL:
                        continue
                    comp = compartment_of(term)
                    if comp:
                        terms.append((comp, go_id, term, code))
                prim, prim_txt, tm = primary_location(acc, gene, terms)
                out["targets"].append(
                    dict(
                        gene=gene,
                        protein=protein_label(acc, gene),
                        accession=acc,
                        organism=org,
                        target_chembl_id=m["target_chembl_id"],
                        target_type=ttype,
                        moa=m["mechanism_of_action"],
                        action=m["action_type"],
                        terms=terms,
                        compartments=sorted({t[0] for t in terms}),
                        primary=prim,
                        primary_text=prim_txt,
                        transmembrane=tm,
                        process=process_of(acc, gene),
                    )
                )
        if not out["targets"]:
            out["note"] = "mechanism target is not a protein (e.g. DNA)"
        elif not any(t["compartments"] for t in out["targets"]):
            out["note"] = "targets have no matching GO CC terms"
    except requests.RequestException as e:
        out["note"] = f"API error: {e}"
    return out


# ## Drawing: cell body
# Cell geometry, organelle colours and positions (`MITOS` holds the perinuclear mitochondria), and the shared cell illustration used by both plots.


# ## Drawing: shared style and cell body
# **✏️ Edit here:** `FONT` sets every text size (points) and `DOT_R` / `LEGEND_DOT_R` the drug-dot sizes; `SCALE` is inches per data unit (the cell is ~15 units across). Both figures are cropped to their content.
# 
# Also here: cell geometry, organelle colours and positions (`MITOS` holds the perinuclear mitochondria), the cell illustration used by both plots, and the shared drug dots and two-column drug legend (drugs 1–10 left, 11–20 right).

# In[11]:


# ---- Figure style: change these to resize text and dots in both figures ----
SCALE = 0.5  # inches per data unit; the cell is ~15 units across
FONT = {
    "card": 14,  # protein names on cards (protein map)
    "gene": 10.5,  # gene symbols on cards (protein map)
    "dot": 11,  # numbers inside the drug dots
    "label": 15,  # compartment labels (GO plot)
    "legend": 19,  # drug names in the legend
    "legend_sub": 12.5,  # targets under each drug name in the legend
}
DOT_R = 0.27  # drug-dot radius on the cell (data units)
LEGEND_DOT_R = 0.38  # drug-dot radius in the legend
PT = 72 * SCALE  # points per data unit
BG = "#fbfaf7"

# Cell geometry (data units). The cell is an ellipse centred at (CX, CY).
CX, CY, RX, RY = 11.5, 9.0, 7.4, 7.1
NUC = (11.0, 9.5, 2.95)  # nucleus centre x, y, radius
CENTROSOME = (12.9, 5.4)
LYSO = (6.2, 6.1, 0.62)  # lysosome used as anchor on the protein map
# Mitochondria sit in a perinuclear ring, long axis tangent to the nucleus: (x, y, w, h, angle).
# The protein map draws only the first two.
MITOS = [
    (13.93, 12.43, 2.1, 0.9, 135),
    (11.56, 13.51, 1.9, 0.85, 172),
    (14.55, 7.45, 2.0, 0.85, 60),
    (10.45, 5.59, 1.8, 0.8, 172),
]

ORG_COLORS = {  # (fill, edge) per compartment
    "Extracellular": ("#d7ecec", "#3f8f8f"),
    "Plasma membrane": ("#f6e3bf", "#b9782a"),
    "Nucleus": ("#dde7f6", "#3f63a0"),
    "Mitochondrion": ("#f9d4cf", "#b7473c"),
    "Endoplasmic reticulum": ("#cfe8c3", "#4b8a37"),
    "Golgi": ("#ecd2f1", "#8a4a99"),
    "Endosome / Lysosome": ("#f8e1a8", "#b07d1a"),
    "Cytoskeleton": ("#e3e7ec", "#68737f"),
    "Cytoplasm": ("#fbf1dc", "#9a7a45"),
}


def drug_palette(n):
    base = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    return [base[i % len(base)] for i in range(n)]


def _ell(t, f=1.0):
    """Point on the cell ellipse at angle t (deg), scaled by f."""
    a = math.radians(t)
    return CX + f * RX * math.cos(a), CY + f * RY * math.sin(a)


def _band_arc(ax, c, r, t1, t2, fill, edge, w=7, z=3):
    """A thick membrane-like arc: dark outline with a lighter core."""
    for lw, col in ((w + 2.4, edge), (w, fill)):
        arc = Arc(c, 2 * r, 2 * r, theta1=t1, theta2=t2, lw=lw, color=col, zorder=z)
        arc.set_capstyle("round")
        ax.add_patch(arc)


def _mito(ax, x, y, w, h, ang, rng):
    fill, edge = ORG_COLORS["Mitochondrion"]
    outer = Ellipse((x, y), w, h, angle=ang, fc=fill, ec=edge, lw=1.8, zorder=3)
    inner = Ellipse(
        (x, y), w - 0.16, h - 0.16, angle=ang, fc="none", ec=edge, lw=0.8, zorder=3
    )
    ax.add_patch(outer)
    ax.add_patch(inner)
    # cristae: folds of the inner membrane, alternating from each side
    a = math.radians(ang)
    ux, uy, vx, vy = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
    n = int(w / 0.28)
    for k in range(1, n):
        s = -w / 2 + k * w / n
        side = 1 if k % 2 else -1
        depth = h * (0.55 + 0.15 * rng.random())
        p0 = (x + s * ux + side * h / 2 * vx, y + s * uy + side * h / 2 * vy)
        p1 = (p0[0] - side * depth * vx, p0[1] - side * depth * vy)
        (ln,) = ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=edge,
            lw=1.6,
            solid_capstyle="round",
            zorder=3,
        )
        ln.set_clip_path(inner)


def _draw_cell_body(ax, rng, full=True):
    """full=False draws a sparser cell (fewer decorative organelles) for the protein map."""
    # --- extracellular matter: matrix fibres and a few ligands
    fill, edge = ORG_COLORS["Extracellular"]
    for _ in range(26):
        t = rng.uniform(0, 360)
        f = rng.uniform(1.07, 1.22)
        x, y = _ell(t, f)
        if 0.2 < y < 17.9:
            ax.add_patch(
                Circle(
                    (x, y),
                    rng.uniform(0.05, 0.11),
                    fc=edge,
                    ec="none",
                    alpha=0.55,
                    zorder=1,
                )
            )
    for t0 in (60, 120, 250, 300):
        pts = [
            _ell(t0 + d, 1.12 + 0.025 * math.sin(d * 0.45))
            for d in np.arange(-14, 14.5, 0.5)
        ]
        ax.plot(*zip(*pts), color=edge, lw=1.1, alpha=0.45, zorder=1)

    # --- cytoplasm: radial gradient clipped to the inside of the membrane
    inner = Ellipse((CX, CY), 2 * (RX - 0.3), 2 * (RY - 0.3), fc="none", ec="none")
    ax.add_patch(inner)
    yy, xx = np.mgrid[-1:1:300j, -1:1:300j]
    rr = np.clip(np.sqrt(xx**2 + yy**2), 0, 1)
    cmap = LinearSegmentedColormap.from_list("cyto", ["#fffaf0", "#fbefd6", "#f4dfb6"])
    im = ax.imshow(
        rr,
        cmap=cmap,
        extent=(CX - RX, CX + RX, CY - RY, CY + RY),
        origin="lower",
        zorder=0.5,
        interpolation="bicubic",
    )
    im.set_clip_path(inner)

    # --- cytoskeleton: microtubules from the centrosome, actin cortex at the edge
    cfill, cedge = ORG_COLORS["Cytoskeleton"]
    cx0, cy0 = CENTROSOME
    for t in np.linspace(0, 360, 22, endpoint=False):
        x1, y1 = _ell(t + rng.uniform(-6, 6), 0.9)
        mx, my = (
            (cx0 + x1) / 2 + rng.uniform(-0.5, 0.5),
            (cy0 + y1) / 2 + rng.uniform(-0.5, 0.5),
        )
        tt = np.linspace(0, 1, 30)
        bx = (1 - tt) ** 2 * cx0 + 2 * (1 - tt) * tt * mx + tt**2 * x1
        by = (1 - tt) ** 2 * cy0 + 2 * (1 - tt) * tt * my + tt**2 * y1
        ax.plot(bx, by, color=cedge, lw=0.9, alpha=0.45, zorder=1)
    for dx, dy, ang in ((0, 0, 0), (0.12, 0.05, 90)):
        ax.add_patch(
            FancyBboxPatch(
                (cx0 + dx - 0.22, cy0 + dy - 0.07),
                0.44,
                0.14,
                boxstyle="round,pad=0.02",
                fc=cedge,
                ec="none",
                zorder=2,
                transform=Affine2D().rotate_deg_around(cx0 + dx, cy0 + dy, ang)
                + ax.transData,
            )
        )
    for t in np.arange(0, 360, 3.2):
        f = rng.uniform(0.9, 0.94)
        x0, y0 = _ell(t, f)
        x1, y1 = _ell(t + rng.uniform(2, 5), f + rng.uniform(-0.02, 0.02))
        ax.plot([x0, x1], [y0, y1], color="#c49a9a", lw=0.8, alpha=0.7, zorder=1.5)

    # --- plasma membrane: lipid bilayer with head groups, plus receptors
    pfill, pedge = ORG_COLORS["Plasma membrane"]
    ax.add_patch(Ellipse((CX, CY), 2 * RX, 2 * RY, fc=pfill, ec="none", zorder=0.3))
    for f, n in ((1.0, 260), (0.958, 250)):
        for t in np.linspace(0, 360, n, endpoint=False):
            ax.add_patch(Circle(_ell(t, f), 0.065, fc=pedge, ec="none", zorder=4))
    for t in (28, 72, 148, 205, 236, 330) if full else ():
        a = math.radians(t)
        nx, ny = RY * math.cos(a), RX * math.sin(a)
        nn = math.hypot(nx, ny)
        nx, ny = nx / nn, ny / nn
        bx, by = _ell(t, 0.94)
        tx, ty = bx + 0.75 * nx, by + 0.75 * ny
        ax.plot(
            [bx, tx], [by, ty], color="#4f6fb0", lw=6, solid_capstyle="round", zorder=5
        )
        ax.add_patch(
            Circle(
                (tx + 0.12 * nx, ty + 0.12 * ny),
                0.17,
                fc="#8fb0e8",
                ec="#4f6fb0",
                lw=1.2,
                zorder=5,
            )
        )

    # --- rough ER: stacked cisternae wrapped around the nucleus, studded with ribosomes
    efill, eedge = ORG_COLORS["Endoplasmic reticulum"]
    nx0, ny0, nr = NUC
    for k, r in enumerate((nr + 0.5, nr + 0.95, nr + 1.4)):
        t1, t2 = 108 + 6 * k, 222 - 5 * k
        _band_arc(ax, (nx0, ny0), r, t1, t2, efill, eedge, w=6.5)
        for t in np.arange(t1 + 2, t2 - 1, 4.5):
            a = math.radians(t)
            ax.add_patch(
                Circle(
                    (nx0 + (r + 0.13) * math.cos(a), ny0 + (r + 0.13) * math.sin(a)),
                    0.045,
                    fc="#2f5a22",
                    ec="none",
                    zorder=3.5,
                )
            )

    # --- nucleus: double envelope with pores, chromatin, nucleolus
    nfill, nedge = ORG_COLORS["Nucleus"]
    nuc = Circle((nx0, ny0), nr, fc=nfill, ec=nedge, lw=2.2, zorder=3)
    ax.add_patch(nuc)
    ax.add_patch(Circle((nx0, ny0), nr - 0.13, fc="none", ec=nedge, lw=1.1, zorder=3))
    k = np.ones(9) / 9
    for _ in range(16):  # chromatin: smoothed random walks
        steps = rng.normal(0, 0.09, (80, 2))
        steps = np.column_stack([np.convolve(steps[:, j], k, "same") for j in (0, 1)])
        pts = (
            np.array([nx0, ny0])
            + rng.uniform(-1.5, 1.5, 2)
            + np.cumsum(steps, axis=0) * 2.2
        )
        (ln,) = ax.plot(
            *pts.T,
            color="#a9bde0",
            lw=1.6,
            alpha=0.75,
            zorder=3,
            solid_capstyle="round",
        )
        ln.set_clip_path(nuc)
    ax.add_patch(
        Circle(
            (nx0 + 0.55, ny0 + 0.45), 0.72, fc="#9db3d9", ec="#6683b5", lw=1, zorder=3.2
        )
    )
    for t in np.arange(0, 360, 22.5):
        a = math.radians(t)
        ax.add_patch(
            Rectangle(
                (
                    nx0 + (nr - 0.07) * math.cos(a) - 0.11,
                    ny0 + (nr - 0.07) * math.sin(a) - 0.06,
                ),
                0.22,
                0.12,
                fc=nfill,
                ec=nedge,
                lw=0.9,
                zorder=3.3,
                transform=Affine2D().rotate_deg_around(
                    nx0 + (nr - 0.07) * math.cos(a),
                    ny0 + (nr - 0.07) * math.sin(a),
                    t + 90,
                )
                + ax.transData,
            )
        )

    # --- Golgi: curved stack of cisternae with budding vesicles
    gfill, gedge = ORG_COLORS["Golgi"]
    gc = (19.6, 8.2)
    for k, r in enumerate((2.55, 2.9, 3.25, 3.6)):
        half = 24 - 3 * k
        _band_arc(ax, gc, r, 180 - half, 180 + half, gfill, gedge, w=5.5)
    for dx, dy in (
        (-3.2, 1.9),
        (-3.55, -1.75),
        (-2.3, 1.55),
        (-4.15, 0.6),
        (-4.2, -0.9),
    ):
        ax.add_patch(
            Circle((gc[0] + dx, gc[1] + dy), 0.14, fc=gfill, ec=gedge, lw=1.1, zorder=3)
        )

    # --- mitochondria
    for x, y, w, h, ang in MITOS if full else MITOS[:2]:
        _mito(ax, x, y, w, h, ang, rng)

    # --- endosomes (with intraluminal vesicles) and lysosomes (with granules)
    lfill, ledge = ORG_COLORS["Endosome / Lysosome"]
    vesicles = [
        (6.2, 6.1, 0.5, "endo"),
        (5.6, 8.8, 0.38, "lyso"),
        (7.5, 5.0, 0.36, "lyso"),
        (13.9, 3.6, 0.42, "endo"),
    ]
    for x, y, r, kind in vesicles if full else [LYSO + ("lyso",)]:
        ax.add_patch(Circle((x, y), r, fc=lfill, ec=ledge, lw=1.6, zorder=3))
        for _ in range(6 if kind == "endo" else 9):
            ang, rad = rng.uniform(0, 2 * math.pi), rng.uniform(0, r * 0.6)
            ax.add_patch(
                Circle(
                    (x + rad * math.cos(ang), y + rad * math.sin(ang)),
                    0.07 if kind == "endo" else 0.04,
                    fc="white" if kind == "endo" else ledge,
                    ec=ledge,
                    lw=0.6,
                    zorder=3.2,
                )
            )


# ---- text measuring, drug dots, legend, figure finishing (shared by both plots) ----
@lru_cache(maxsize=None)
def _text_w(s, size, weight="normal", style="normal"):
    """Rendered width of `s` in data units."""
    path = TextPath(
        (0, 0), s, size=size, prop=FontProperties(weight=weight, style=style)
    )
    return path.get_extents().width / PT


def _fit(s, size, max_w, weight="normal"):
    """Shorten `s` with an ellipsis until it fits in max_w."""
    if _text_w(s, size, weight) <= max_w:
        return s
    while s and _text_w(s + "…", size, weight) > max_w:
        s = s[:-1]
    return s.rstrip(", ") + "…"


def _wrap_items(items, size, max_w, weight="normal", style="normal"):
    """Greedy wrap of comma-separated items into lines no wider than max_w."""
    lines, cur = [], ""
    for k, item in enumerate(items):
        piece = item + ("," if k < len(items) - 1 else "")
        trial = f"{cur} {piece}".strip()
        if cur and _text_w(trial, size, weight, style) > max_w:
            lines.append(cur)
            cur = piece
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _compact_names(items, min_items=3):
    """PKCα, PKCβ, PKCγ -> PKCα, β, γ when every item shares a name prefix
    (3+ items). Keeps big grouped cards small enough to fit inside the cell."""
    if len(items) < min_items:
        return list(items)
    prefix = os.path.commonprefix(list(items))
    while prefix and prefix[-1].isdigit():  # HDAC1/HDAC11 share "HDAC", not "HDAC1"
        prefix = prefix[:-1]
    if len(prefix.strip()) < 2 or any(len(i) == len(prefix) for i in items):
        return list(items)
    return [items[0]] + [i[len(prefix) :] for i in items[1:]]


def _natural_key(s):
    """Sort HDAC2 before HDAC11."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _ink(rgb):
    """Black or white text, whichever reads better on this fill."""
    r, g, b = rgb[:3]
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "#111"


def _drug_dot(ax, x, y, i, colors, r=DOT_R, size=None, z=9):
    ax.add_patch(Circle((x, y), r, fc=colors[i], ec="#222", lw=0.9, zorder=z))
    ax.text(
        x,
        y,
        str(i + 1),
        ha="center",
        va="center",
        fontsize=size or FONT["dot"],
        fontweight="bold",
        color=_ink(colors[i]),
        zorder=z + 1,
    )


def _drug_legend(ax, plotted, colors, x0, y_mid):
    """Two columns: drugs 1..ceil(n/2) on the left, the rest on the right.
    Each row: numbered dot, drug name, and its target proteins underneath.
    Returns the legend's bounding box (x0, y0, x1, y1)."""
    n = len(plotted)
    per_col = max(1, math.ceil(n / 2))
    row_h = 1.45
    gap = LEGEND_DOT_R * 2 + 0.3
    name_w = max(_text_w(d["query"], FONT["legend"], "bold") for d in plotted)
    text_w = max(name_w, 4.2)
    col_w = gap + text_w + 0.9
    top = y_mid + per_col * row_h / 2
    for i, d in enumerate(plotted):
        col, row = divmod(i, per_col)
        cx = x0 + col * col_w + LEGEND_DOT_R
        cy = top - (row + 0.5) * row_h
        _drug_dot(ax, cx, cy, i, colors, r=LEGEND_DOT_R, size=FONT["dot"] + 2)
        tx = cx + LEGEND_DOT_R + 0.3
        ax.text(
            tx,
            cy + 0.2,
            d["query"],
            ha="left",
            va="center",
            fontsize=FONT["legend"],
            fontweight="bold",
            color="#1e1e1e",
        )
        targets = ", ".join(dict.fromkeys(t["protein"] for t in d["targets"]))
        ax.text(
            tx,
            cy - 0.36,
            _fit(targets, FONT["legend_sub"], text_w),
            ha="left",
            va="center",
            fontsize=FONT["legend_sub"],
            color="#555",
        )
    return (x0, top - per_col * row_h, x0 + 2 * col_w - 0.9, top)


def _finish(fig, ax, bounds, out_png, pad=0.3):
    """Crop the canvas to the drawn content, size the figure so 1 unit = SCALE
    inches (keeps FONT sizes meaningful), and save."""
    x0 = min(b[0] for b in bounds) - pad
    y0 = min(b[1] for b in bounds) - pad
    x1 = max(b[2] for b in bounds) + pad
    y1 = max(b[3] for b in bounds) + pad
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.set_size_inches((x1 - x0) * SCALE, (y1 - y0) * SCALE)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_png, dpi=200, facecolor=BG)
    plt.close(fig)


# ## Plot 1: GO plot
# Each drug appears as a numbered, coloured dot on every compartment its targets' GO terms place it in. `GO_CLUSTERS` sets where each compartment's dots and label sit on the cell. The drug legend sits to the right.

# In[12]:


# Where each compartment's drug dots sit on the cell, and where its label goes.
#   grid:     rows of `cols` dots centred on `at`
#   ring:     dots evenly around a circle of radius `r` centred on `at`
#   nuc_arc:  dots along arcs around the nucleus (radii `rs`, angles t1..t2 in degrees)
#   cell_arc: dots along the cell outline at scale `f`, angles t1..t2
# label = (x, y, horizontal alignment, vertical alignment); text may contain "\n".
GO_CLUSTERS = {
    "Nucleus": dict(
        kind="grid",
        at=(NUC[0], NUC[1] - 0.25),
        cols=4,
        label=(NUC[0], NUC[1] + 1.75, "center", "center"),
    ),
    "Plasma membrane": dict(
        kind="cell_arc",
        f=0.985,
        t1=222,
        t2=322,
        label=(CX, CY - RY - 0.45, "center", "top"),
    ),
    "Extracellular": dict(
        kind="cell_arc",
        f=1.1,
        t1=98,
        t2=128,
        label=(*_ell(113, 1.22), "center", "bottom"),
    ),
    "Endoplasmic reticulum": dict(
        kind="nuc_arc",
        rs=(NUC[2] + 0.72, NUC[2] + 1.4),
        t1=122,
        t2=202,
        label=(7.35, 13.65, "center", "center"),
        text="Endoplasmic\nreticulum",
    ),
    "Golgi": dict(
        kind="grid",
        at=(16.55, 8.2),
        cols=2,
        label=(16.55, 10.45, "center", "bottom"),
    ),
    "Mitochondrion": dict(
        kind="grid",
        at=(14.75, 13.35),
        cols=3,
        label=(14.75, 12.55, "center", "top"),
        text="Mitochondria",
    ),
    "Cytoskeleton": dict(
        kind="ring",
        at=CENTROSOME,
        r=0.95,
        label=(14.45, 5.4, "left", "center"),
    ),
    "Endosome / Lysosome": dict(
        kind="ring",
        at=(6.55, 5.95),
        r=0.85,
        label=(6.3, 7.12, "center", "bottom"),
        text="Endosome /\nlysosome",
    ),
    "Cytoplasm": dict(
        kind="grid",
        at=(10.0, 3.78),
        cols=6,
        label=(9.15, 4.72, "center", "bottom"),
    ),
}


def _cluster_points(spec, n):
    step = 2 * DOT_R + 0.08
    kind = spec["kind"]
    if kind == "grid":
        cols = min(spec["cols"], n)
        rows = math.ceil(n / cols)
        pts = []
        for k in range(n):
            r, c = divmod(k, cols)
            in_row = min(cols, n - r * cols)
            pts.append(
                (
                    spec["at"][0] + (c - (in_row - 1) / 2) * step,
                    spec["at"][1] + ((rows - 1) / 2 - r) * step,
                )
            )
        return pts
    if kind == "ring":
        x0, y0 = spec["at"]
        return [
            (
                x0 + spec["r"] * math.cos(math.radians(90 + 360 * k / n)),
                y0 + spec["r"] * math.sin(math.radians(90 + 360 * k / n)),
            )
            for k in range(n)
        ]
    if kind == "cell_arc":
        ts = (
            np.linspace(spec["t1"], spec["t2"], n)
            if n > 1
            else [(spec["t1"] + spec["t2"]) / 2]
        )
        return [_ell(t, spec["f"]) for t in ts]
    if kind == "nuc_arc":
        rs = spec["rs"]
        counts = [
            int(n * r / sum(rs)) for r in rs
        ]  # longer (outer) arcs take more dots
        counts[-1] += n - sum(counts)
        pts = []
        for r, m in zip(rs, counts):
            if m == 0:
                continue
            half = (spec["t2"] - spec["t1"]) / 2
            mid = (spec["t1"] + spec["t2"]) / 2
            span = min(half, math.degrees(step * (m - 1) / 2 / r) * 1.35)
            ts = np.linspace(mid - span, mid + span, m) if m > 1 else [mid]
            pts += [
                (
                    NUC[0] + r * math.cos(math.radians(t)),
                    NUC[1] + r * math.sin(math.radians(t)),
                )
                for t in ts
            ]
        return pts
    raise ValueError(kind)


def _label_box(x, y, text, size, ha, va, weight="bold", style="italic"):
    lines = text.split("\n")
    w = max(_text_w(l, size, weight, style) for l in lines)
    h = len(lines) * size * 1.25 / PT
    x0 = {"left": x, "center": x - w / 2, "right": x - w}[ha]
    y0 = {"bottom": y, "center": y - h / 2, "top": y - h}[va]
    return (x0, y0, x0 + w, y0 + h)


def draw_cell(drugs, out_png, evidence_label):
    plotted = [d for d in drugs if d["targets"]]
    colors = drug_palette(len(plotted))
    rng = np.random.default_rng(7)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    _draw_cell_body(ax, rng)

    by_comp = defaultdict(list)
    for i, d in enumerate(plotted):
        for c in sorted({c for t in d["targets"] for c in t["compartments"]}):
            by_comp[c].append(i)

    bounds = [(CX - RX * 1.03, CY - RY * 1.03, CX + RX * 1.03, CY + RY * 1.03)]
    for comp, spec in GO_CLUSTERS.items():
        idxs = by_comp.get(comp, [])
        if not idxs:
            continue
        for (x, y), i in zip(_cluster_points(spec, len(idxs)), idxs):
            _drug_dot(ax, x, y, i, colors)
            bounds.append((x - DOT_R, y - DOT_R, x + DOT_R, y + DOT_R))
        lx, ly, ha, va = spec["label"]
        text = spec.get("text", comp)
        ax.text(
            lx,
            ly,
            text,
            ha=ha,
            va=va,
            fontsize=FONT["label"],
            fontweight="bold",
            style="italic",
            color=ORG_COLORS[comp][1],
            linespacing=1.05,
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8),
        )
        bounds.append(_label_box(lx, ly, text, FONT["label"], ha, va))

    x_leg = max(b[2] for b in bounds) + 0.9
    bounds.append(_drug_legend(ax, plotted, colors, x_leg, CY))
    _finish(fig, ax, bounds, out_png)


# ## Plot 2: protein map
# Proteins in the same compartment, with the same process and the identical set of drugs, share one card. Each card shows the protein names, gene symbols in italics, and a numbered dot for every drug that hits it. Membrane proteins keep their own bar in the bilayer, grouped and joined to the shared card. Cards are packed into free space without overlaps; the drug legend sits to the right.

# In[13]:


PROCESS_COLORS = {
    "MAPK cascade": "#e07b39",
    "PI3K–AKT–mTOR": "#2f8f6a",
    "Chromatin & epigenetics": "#5161b5",
    "Protein folding (HSP90)": "#c2457a",
    "Ubiquitin–proteasome": "#8b5bb5",
    "PKC / Ca²⁺ signalling": "#c9a227",
    "Ion pumps & channels": "#2b8cbf",
    "Receptor tyrosine kinases": "#c0392b",
    "Non-receptor tyrosine kinases": "#9c5b2e",
    "GPCR signalling": "#3aa39b",
    "Other": "#7a7a7a",
}
# Points on organelles that organelle-bound protein cards are tethered to
ORGANELLE_ANCHOR = {
    "Endosome / Lysosome": LYSO[:2],
    "Endoplasmic reticulum": (NUC[0] - 2.95, NUC[1] + 2.15),
    "Golgi": (16.6, 8.2),
    "Mitochondrion": MITOS[0][:2],
    "Cytoskeleton": CENTROSOME,
    "Cytoplasm": (12.3, 4.6),
}


def _tint(hex_color, amount=0.82):
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _in_cytoplasm(px, py):
    """True if a point is free cytoplasm (inside the cell, clear of drawn organelles)."""
    nx0, ny0, nr = NUC
    if ((px - CX) / (RX * 0.89)) ** 2 + ((py - CY) / (RY * 0.89)) ** 2 > 1:
        return False
    d = math.hypot(px - nx0, py - ny0)
    ang = math.degrees(math.atan2(py - ny0, px - nx0)) % 360
    if d < nr + 0.25:  # nucleus
        return False
    if nr + 0.2 < d < nr + 1.8 and 102 < ang < 230:  # rough ER
        return False
    if px > 15.2 and abs(py - 8.2) < 2.3:  # Golgi
        return False
    for mx, my, mw, mh, _ in MITOS[:2]:  # mitochondria
        if math.hypot(px - mx, py - my) < mw / 2 + 0.2:
            return False
    if math.hypot(px - LYSO[0], py - LYSO[1]) < LYSO[2] + 0.2:
        return False
    return True


def _in_nucleus(px, py):
    nx0, ny0, nr = NUC
    return math.hypot(px - nx0, py - ny0) < nr - 0.12


def _in_cell_not_nucleus(px, py):
    """Fallback for cytoplasm/organelle cards: may overlap ER, Golgi or
    mitochondria, but never the nucleus (that would read as nuclear)."""
    nx0, ny0, nr = NUC
    return _in_cell(px, py) and math.hypot(px - nx0, py - ny0) > nr + 0.15


def _in_cell(px, py):
    """Relaxed fallback region: anywhere inside the cell outline, organelles
    included. Used only when a card has no free, organelle-clear spot left --
    bigger cards (bigger fonts) can run out of clear cytoplasm before they
    run out of room entirely, and a card overlapping an organelle reads
    better than one silently missing from the figure."""
    return ((px - CX) / (RX * 0.93)) ** 2 + ((py - CY) / (RY * 0.93)) ** 2 <= 1


class _Packer:
    """Places variable-size cards at the free spot nearest an anchor, without overlaps."""

    def __init__(self):
        self.rects = []
        xs = np.arange(CX - RX, CX + RX, 0.1)
        ys = np.arange(CY - RY, CY + RY, 0.1)
        self.cands = [(float(x), float(y)) for x in xs for y in ys]

    def _clear(self, x, y, w, h, region):
        m = 0.1
        for a0, b0, a1, b1 in self.rects:
            if (
                x - w / 2 - m < a1
                and x + w / 2 + m > a0
                and y - h / 2 - m < b1
                and y + h / 2 + m > b0
            ):
                return False
        nxs = max(2, int(w / 0.25) + 1)
        nys = max(2, int(h / 0.25) + 1)
        for px in np.linspace(x - w / 2 - 0.05, x + w / 2 + 0.05, nxs):
            for py in np.linspace(y - h / 2 - 0.05, y + h / 2 + 0.05, nys):
                if not region(px, py):
                    return False
        return True

    def place(self, w, h, near, region=_in_cytoplasm):
        for x, y in sorted(
            self.cands, key=lambda c: (c[0] - near[0]) ** 2 + (c[1] - near[1]) ** 2
        ):
            if self._clear(x, y, w, h, region):
                self.rects.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
                return x, y
        return None


CARD_TEXT_W = 4.0  # widest a card's text may get before wrapping (data units)
CARD_PAD = 0.17
DOTS_PER_ROW = 5
DOT_STEP = 2 * DOT_R + 0.07
LH_CARD = 1.22 * FONT["card"] / PT  # line heights in data units
LH_GENE = 1.22 * FONT["gene"] / PT


def _card_layout(proteins, genes, n_drugs):
    """Wrap the comma-separated names; return (name_lines, gene_lines, w, h).
    The card also holds a row (or rows) of numbered drug dots under the text."""
    show_genes = any(g != p for g, p in zip(genes, proteins))
    name_lines = _wrap_items(
        _compact_names(proteins), FONT["card"], CARD_TEXT_W, "bold"
    )
    gene_lines = (
        _wrap_items(genes, FONT["gene"], CARD_TEXT_W, style="italic")
        if show_genes
        else []
    )
    dot_rows = math.ceil(n_drugs / DOTS_PER_ROW)
    w = 2 * CARD_PAD + max(
        [_text_w(l, FONT["card"], "bold") for l in name_lines]
        + [_text_w(l, FONT["gene"], style="italic") for l in gene_lines]
        + [min(n_drugs, DOTS_PER_ROW) * DOT_STEP]
    )
    h = (
        CARD_PAD
        + len(name_lines) * LH_CARD
        + len(gene_lines) * LH_GENE
        + dot_rows * DOT_STEP
        + CARD_PAD * 0.6
    )
    return name_lines, gene_lines, w, h


def _card(ax, x, y, group, colors):
    col = PROCESS_COLORS.get(group["process"], PROCESS_COLORS["Other"])
    w, h = group["w"], group["h"]
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            fc=_tint(col),
            ec=col,
            lw=1.8,
            zorder=8,
        )
    )
    ty = y + h / 2 - CARD_PAD - LH_CARD / 2
    for line in group["name_lines"]:
        ax.text(
            x,
            ty,
            line,
            ha="center",
            va="center",
            fontsize=FONT["card"],
            fontweight="bold",
            color="#1e1e1e",
            zorder=9,
        )
        ty -= LH_CARD
    ty += (LH_CARD - LH_GENE) / 2
    for line in group["gene_lines"]:
        ax.text(
            x,
            ty,
            line,
            ha="center",
            va="center",
            fontsize=FONT["gene"],
            style="italic",
            color="#555",
            zorder=9,
        )
        ty -= LH_GENE
    ty -= (DOT_STEP - LH_GENE) / 2 if group["gene_lines"] else (DOT_STEP - LH_CARD) / 2
    drugs_ = group["drugs"]
    for r0 in range(0, len(drugs_), DOTS_PER_ROW):
        row = drugs_[r0 : r0 + DOTS_PER_ROW]
        x0 = x - DOT_STEP * (len(row) - 1) / 2
        for k, i in enumerate(row):
            _drug_dot(ax, x0 + k * DOT_STEP, ty, i, colors, z=9)
        ty -= DOT_STEP


def _normal(t):
    a = math.radians(t)
    nx_, ny_ = RY * math.cos(a), RX * math.sin(a)
    nn = math.hypot(nx_, ny_)
    return nx_ / nn, ny_ / nn


def draw_protein_map(drugs, out_png):
    plotted = [d for d in drugs if d["targets"]]
    colors = drug_palette(len(plotted))
    rng = np.random.default_rng(7)

    # one entry per protein: where it sits, its process, and which drugs hit it
    prots = {}
    for i, d in enumerate(plotted):
        for t in d["targets"]:
            p = prots.setdefault(
                t["accession"],
                dict(
                    gene=t["gene"],
                    protein=t["protein"],
                    comp=t["primary"],
                    tm=t["transmembrane"],
                    process=t["process"],
                    drugs=[],
                ),
            )
            if i not in p["drugs"]:
                p["drugs"].append(i)

    # group proteins in the same place, same process, hit by exactly the same drugs
    groups = {}
    for p in prots.values():
        kind = (
            ("tm" if p["tm"] else "peripheral")
            if p["comp"] == "Plasma membrane"
            else p["comp"]
        )
        k = (p["comp"], kind, p["process"], tuple(sorted(p["drugs"])))
        g = groups.setdefault(
            k,
            dict(
                comp=p["comp"],
                kind=kind,
                process=p["process"],
                drugs=sorted(p["drugs"]),
                members=[],
            ),
        )
        g["members"].append(p)
    order = list(PROCESS_COLORS)
    for g in groups.values():
        g["members"].sort(key=lambda p: _natural_key(p["protein"]))
        g["name_lines"], g["gene_lines"], g["w"], g["h"] = _card_layout(
            [p["protein"] for p in g["members"]],
            [p["gene"] for p in g["members"]],
            len(g["drugs"]),
        )
    by_comp = defaultdict(list)
    for g in groups.values():
        by_comp[g["comp"]].append(g)
    for v in by_comp.values():  # process keeps colours together; big cards first
        v.sort(
            key=lambda g: (
                order.index(g["process"]) if g["process"] in order else 99,
                -g["w"] * g["h"],
            )
        )

    fig, ax = plt.subplots()
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    _draw_cell_body(ax, rng, full=False)
    packer = _Packer()
    bounds = [(CX - RX * 1.03, CY - RY * 1.03, CX + RX * 1.03, CY + RY * 1.03)]

    def place_card(g, x, y):
        g["pos"] = (x, y)
        _card(ax, x, y, g, colors)
        bounds.append((x - g["w"] / 2, y - g["h"] / 2, x + g["w"] / 2, y + g["h"] / 2))

    # --- plasma membrane & extracellular: one card per group, members side by side in the bilayer
    ring = by_comp.get("Plasma membrane", []) + by_comp.get("Extracellular", [])
    weights = [max(g["w"], g["h"]) + 0.9 for g in ring]
    total = sum(weights) or 1
    t_acc = 90.0
    for g, wt in zip(ring, weights):
        span = 360 * wt / total
        tg = t_acc + span / 2
        t_acc += span
        col = PROCESS_COLORS.get(g["process"], "#777")
        nx_, ny_ = _normal(tg)
        bx0, by0 = _ell(tg, 1.0)
        ext = abs(nx_) * g["w"] / 2 + abs(ny_) * g["h"] / 2
        cx, cy = bx0 + (0.85 + ext) * nx_, by0 + (0.85 + ext) * ny_
        m = len(g["members"])
        spread = min(2.4, span * 0.8 / max(m, 1))
        for j, p in enumerate(g["members"]):
            t = tg + (j - (m - 1) / 2) * spread
            ux, uy = _normal(t)
            bx, by = _ell(t, 0.95)
            if g["kind"] == "tm":  # spans the bilayer
                ax.plot(
                    [bx, bx + 0.5 * ux],
                    [by, by + 0.5 * uy],
                    color=col,
                    lw=8,
                    solid_capstyle="round",
                    zorder=6,
                )
                tip = (bx + 0.5 * ux, by + 0.5 * uy)
            elif g["kind"] == "peripheral":  # inner leaflet
                ax.add_patch(
                    Circle(
                        (bx - 0.08 * ux, by - 0.08 * uy),
                        0.2,
                        fc=col,
                        ec="#222",
                        lw=0.8,
                        zorder=6,
                    )
                )
                tip = (bx, by)
            else:  # secreted / extracellular
                tip = _ell(t, 1.06)
                ax.add_patch(Circle(tip, 0.2, fc=col, ec="#222", lw=0.8, zorder=6))
            ax.plot([tip[0], cx], [tip[1], cy], color=col, lw=1.2, zorder=7)
        place_card(g, cx, cy)

    # --- nucleus: cards inside the nucleus, overflow tethered just outside.
    # The biggest card is packed towards the bottom so the others fit above it.
    nuc_anchor = (NUC[0], NUC[1] - NUC[2] + 0.2)
    for g in sorted(by_comp.get("Nucleus", []), key=lambda g: -g["w"] * g["h"]):
        spot = packer.place(g["w"], g["h"], nuc_anchor, region=_in_nucleus)
        if spot is None:
            spot = packer.place(g["w"], g["h"], NUC[:2])
            if spot is None:
                print(f"  warning: no room for {g['name_lines']}")
                continue
            ax.plot(
                [spot[0], NUC[0]], [spot[1], NUC[1]], color="#3f63a0", lw=1.2, zorder=7
            )
        place_card(g, *spot)

    # --- organelle-bound proteins: nearest free spot, tethered to the organelle
    for comp in (
        "Endosome / Lysosome",
        "Endoplasmic reticulum",
        "Golgi",
        "Mitochondrion",
        "Cytoskeleton",
    ):
        anchor = ORGANELLE_ANCHOR[comp]
        edge = ORG_COLORS[comp][1]
        for g in by_comp.get(comp, []):
            spot = packer.place(g["w"], g["h"], anchor)
            if spot is None:
                spot = packer.place(g["w"], g["h"], anchor, region=_in_cell_not_nucleus)
            if spot is None:
                print(f"  warning: no room for {g['name_lines']}")
                continue
            ax.plot(
                [spot[0], anchor[0]], [spot[1], anchor[1]], color=edge, lw=1.5, zorder=7
            )
            ax.add_patch(Circle(anchor, 0.12, fc="white", ec=edge, lw=1.8, zorder=7.5))
            place_card(g, *spot)

    # --- cytoplasm: biggest cards first, filling outward from the cytoplasm centre
    for g in sorted(by_comp.get("Cytoplasm", []), key=lambda g: -g["w"] * g["h"]):
        spot = packer.place(g["w"], g["h"], ORGANELLE_ANCHOR["Cytoplasm"])
        if spot is None:
            spot = packer.place(
                g["w"],
                g["h"],
                ORGANELLE_ANCHOR["Cytoplasm"],
                region=_in_cell_not_nucleus,
            )
        if spot is None:
            print(f"  warning: no room for {g['name_lines']}")
            continue
        place_card(g, *spot)

    x_leg = max(b[2] for b in bounds) + 0.9
    bounds.append(_drug_legend(ax, plotted, colors, x_leg, CY))
    _finish(fig, ax, bounds, out_png)


# ## Heatmap and CSV

# In[14]:


def write_csv(drugs, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "query",
                "drug",
                "chembl_id",
                "target_chembl_id",
                "target_type",
                "protein",
                "gene",
                "uniprot",
                "organism",
                "mechanism",
                "action",
                "primary_location",
                "uniprot_location",
                "process",
                "compartment",
                "go_id",
                "go_term",
                "evidence",
                "note",
            ]
        )
        for d in drugs:
            if not d["targets"]:
                w.writerow(
                    [d["query"], d["name"], d["chembl_id"]] + [""] * 15 + [d["note"]]
                )
            for t in d["targets"]:
                base = [
                    d["query"],
                    d["name"],
                    d["chembl_id"],
                    t["target_chembl_id"],
                    t["target_type"],
                    t["protein"],
                    t["gene"],
                    t["accession"],
                    t["organism"],
                    t["moa"],
                    t["action"],
                    t["primary"],
                    t["primary_text"],
                    t["process"],
                ]
                if not t["terms"]:
                    w.writerow(base + ["", "", "", "", "no matching GO CC terms"])
                for comp, go_id, term, ev in t["terms"]:
                    w.writerow(base + [comp, go_id, term, ev, ""])
    print(f"Saved {out_csv}")


# In[15]:


def load_drugs_from_csv(in_csv):
    """Reverse of write_csv: rebuild the per-drug/per-target nested structure
    draw_cell/draw_protein_map expect from the flat results CSV."""
    df = pd.read_csv(in_csv, keep_default_na=False, dtype=str)
    drugs = []
    group_cols = ["query", "drug", "chembl_id"]
    for (query, name, chembl_id), g in df.groupby(group_cols, sort=False, dropna=False):
        note = next((n for n in g["note"] if n), "")
        targets = []
        seen_acc = set()
        target_cols = [
            "target_chembl_id",
            "target_type",
            "protein",
            "gene",
            "uniprot",
            "organism",
            "mechanism",
            "action",
            "primary_location",
            "uniprot_location",
            "process",
        ]
        for key, tg in g.groupby(target_cols, sort=False, dropna=False):
            (
                target_chembl_id,
                target_type,
                protein,
                gene,
                acc,
                organism,
                moa,
                action,
                primary,
                primary_text,
                process,
            ) = key
            if not acc or acc in seen_acc:
                continue
            seen_acc.add(acc)
            terms = [
                (r.compartment, r.go_id, r.go_term, r.evidence)
                for r in tg.itertuples()
                if r.go_id
            ]
            targets.append(
                dict(
                    gene=gene,
                    protein=protein,
                    accession=acc,
                    organism=organism,
                    target_chembl_id=target_chembl_id or None,
                    target_type=target_type,
                    moa=moa,
                    action=action,
                    terms=terms,
                    compartments=sorted({c for c, *_ in terms}),
                    primary=primary,
                    primary_text=primary_text,
                    transmembrane="pass" in primary_text.lower()
                    or "gpi" in primary_text.lower(),
                    process=process,
                )
            )
        drugs.append(
            dict(
                query=query,
                chembl_id=chembl_id or None,
                name=name,
                targets=targets,
                note=note,
            )
        )
    return drugs


# ## Reading the input
# Accepts a plate-map CSV, a text list or a list of names. Vehicle controls (DMSO, PBS, media, …) are dropped.


# ## Reading the input
# Accepts a plate-map CSV, a text list or a list of names. Vehicle controls (DMSO, PBS, media, …) are dropped.

# In[16]:


CONTROLS = {
    "dmso",
    "pbs",
    "media",
    "medium",
    "untreated",
    "vehicle",
    "blank",
    "empty",
    "none",
    "water",
    "h2o",
    "ethanol",
    "etoh",
    "control",
    "ctrl",
    "nan",
    "",
}


def read_inputs(items, column="Treatment"):
    """Accept a plate-map CSV (uses `column`), a plain text list, or names on the CLI.
    File paths may be strings or pathlib.Path objects."""
    items = [os.fspath(i) if isinstance(i, os.PathLike) else str(i) for i in items]
    if len(items) == 1 and os.path.isfile(items[0]):
        path = items[0]
        with open(path, newline="", encoding="utf-8-sig") as f:
            head = f.readline()
            f.seek(0)
            if path.lower().endswith(".csv") and column.lower() in head.lower():
                rows = csv.DictReader(f)
                col = next(
                    c for c in rows.fieldnames if c.strip().lower() == column.lower()
                )
                names = [(r.get(col) or "").strip() for r in rows]
            else:
                names = [ln.split("#")[0].strip() for ln in f]
    else:
        names = items
    # unique, original order, vehicle/controls dropped
    return [n for n in dict.fromkeys(names) if n.lower() not in CONTROLS]


# ## Run
# Looks up every drug (in parallel), prints a summary, and writes the CSV and the three plots.


# ## Run
# Looks up every drug (in parallel), prints a summary, and writes the CSV and the three plots.

# In[17]:


items = INPUT if isinstance(INPUT, (list, tuple)) else [INPUT]
queries = read_inputs(list(items), COLUMN)
experimental_only = not ALL_EVIDENCE
label = (
    "using only experimentally confirmed locations (GO evidence codes, not "
    "computational predictions)"
    if experimental_only
    else "using all location evidence, including computational predictions"
)
print(
    f"Processing {len(queries)} drugs "
    f"({'experimental' if experimental_only else 'all'} GO evidence)..."
)

if not cell_protein_targets_results_out_file_path.exists():
    with ThreadPoolExecutor(WORKERS) as ex:
        drugs = list(ex.map(lambda q: process_drug(q, experimental_only), queries))

    for d in drugs:
        comps = sorted({c for t in d["targets"] for c in t["compartments"]})
        genes = ", ".join(
            dict.fromkeys(f"{t['protein']} ({t['gene']})" for t in d["targets"])
        )
        print(f"  {d['query']:<16} {d['chembl_id'] or '-':<14} {genes or '-'}")

    skipped = [d for d in drugs if not d["targets"]]
    if skipped:
        print(
            "Not plotted: " + "; ".join(f"{d['query']} ({d['note']})" for d in skipped)
        )

    write_csv(drugs, cell_protein_targets_results_out_file_path)
else:
    drugs = load_drugs_from_csv(cell_protein_targets_results_out_file_path)

draw_cell(drugs, cell_structures_targets_image_out_file_path, label)
draw_protein_map(drugs, cell_protein_targets_image_out_file_path)


# ## Results


# ## Results

# In[18]:


display(Image(f"{cell_structures_targets_image_out_file_path}", width=1100))
display(Image(f"{cell_protein_targets_image_out_file_path}", width=1100))

