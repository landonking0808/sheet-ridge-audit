"""VC3D point-collection JSON writer.

Format as read/written by VC3D's ``PointCollections`` (ScrollPrize/villa,
``volume-cartographer/core/src/PointCollections.cpp``,
``VC_POINTCOLLECTIONS_JSON_VERSION "1"``) and by villa's Python loader
(``spiral-fitting/point_collection.py``)::

    {"vc_pointcollections_json_version": "1",
     "collections": {"<id>": {"uuid": "...", "name": "...",
                              "points": {"<id>": {"p": [x, y, z], "creation_time": ms,
                                                  "wind_a": null}},
                              "metadata": {"winding_is_absolute": false},
                              "color": [r, g, b], "tags": {"k": "v"}}}}

``p`` is in volume voxels, **x, y, z** order (villa reads ``zyx = p[::-1]``).
"""
from __future__ import annotations

import json
import time
import uuid as _uuid


def suspects_collection(suspects, segment_id: str, color=(1.0, 0.25, 0.25), extra_tags=None):
    now = int(time.time() * 1000)
    pts = {}
    for s in suspects:
        z, y, x = s.zyx
        pts[str(s.id)] = {"p": [float(x), float(y), float(z)], "creation_time": now, "wind_a": None}
    tags = {"tool": "sheet-ridge-audit", "segment": str(segment_id),
            "point_ids": "suspect rank (1 = most severe); see suspects.csv"}
    tags.update({k: str(v) for k, v in (extra_tags or {}).items()})
    return {
        "uuid": str(_uuid.uuid4()),
        "name": f"sra {segment_id} suspects",
        "points": pts,
        "metadata": {"winding_is_absolute": False},
        "color": [float(c) for c in color],
        "tags": tags,
    }


def write_point_collections(path, collections):
    doc = {"vc_pointcollections_json_version": "1",
           "collections": {str(i + 1): c for i, c in enumerate(collections)}}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return doc
