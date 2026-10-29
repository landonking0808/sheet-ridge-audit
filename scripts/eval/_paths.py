"""Shared paths for the evaluation scripts (docs/eval/).

All downloads and intermediate outputs go under one work directory, set by the
environment variable SRA_EVAL_DATA (default: ./sra-eval-data in the current
directory). Small inputs shipped with the repo (the published affine, the
review labels) are read from the repo itself. Nothing large is committed: the
CC BY-NC data are fetched from the Vesuvius Challenge open-data bucket.
"""
import os
import sys
import urllib.request

EVAL = os.path.dirname(os.path.abspath(__file__))           # scripts/eval
REPO = os.path.dirname(os.path.dirname(EVAL))
DOCS = os.path.join(REPO, "docs", "eval")
DATA = os.path.abspath(os.environ.get("SRA_EVAL_DATA", "sra-eval-data"))
if EVAL not in sys.path:
    sys.path.insert(0, EVAL)                                 # lasagna_backend, s3ls

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"

# Segments used by the evaluations: local name -> tifxyz folder in the bucket.
SEGMENTS = {
    # PHerc0139 w025, native on the Lasagna scan (baseline study, v0.3 dev plants)
    "20250108000000-on-20260102150214-2.399um.tifxyz":
        "PHerc0139/segments/20250108000000-w025_2025010863/mesh/20250108000000-on-20260102150214-2.399um.tifxyz/",
    # PHerc1667 w011 on the 2.399 um Lasagna scan and on the 7.91 um scan (affine check)
    "20260108140509-on-20251217075048-2.399um.tifxyz":
        "PHerc1667/segments/20260108140509-w011_20260108140509268_flatboi/mesh/20260108140509-on-20251217075048-2.399um.tifxyz/",
    "20260108140509-on-20231117161658-7.91um.tifxyz":
        "PHerc1667/segments/20260108140509-w011_20260108140509268_flatboi/mesh/20260108140509-on-20231117161658-7.91um.tifxyz/",
}


def data(*parts):
    """A path under the work directory; its parent folder is created."""
    p = os.path.join(DATA, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def segment(name):
    """Local path of a segment in SEGMENTS, downloaded on first use (x/y/z.tif + meta.json)."""
    d = data("segments", name)
    os.makedirs(d, exist_ok=True)
    for f in ("meta.json", "x.tif", "y.tif", "z.tif"):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(BUCKET + SEGMENTS[name] + f, p + ".part")
            os.replace(p + ".part", p)
    return d


W025 = "20250108000000-on-20260102150214-2.399um.tifxyz"
