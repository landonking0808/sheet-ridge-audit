"""sheet-ridge-audit: find where tifxyz segments leave the papyrus sheet or switch sheets.

Stable public API::

    from sheet_ridge_audit import audit, open_field, Params, Field
    res = audit("SEG.tifxyz", open_field("lasagna", "PHerc0139"), stride=2)
    res.scores, res.suspects, res.metrics, res.labels, res.write("out/")

* :func:`audit` (``sheet_ridge_audit.audit.audit``) -> :class:`AuditResult`
* :func:`open_field` -> a :class:`Field` (``sample(points_zyx) -> values``)
* :class:`Params` (detection thresholds), :mod:`sheet_ridge_audit.plant` (planted switches)
"""
__version__ = "0.2.0"
from .fields import ArrayField, Field, ZarrField, as_field  # noqa: F401,E402
from .tifxyz import Segment, grid_normals, read_tifxyz, write_tifxyz  # noqa: F401,E402
from .analysis import Params, Suspect, Metrics  # noqa: F401,E402
from .audit import AuditResult, audit, patch_score, run_audit  # noqa: F401,E402
from .backends import open_field  # noqa: F401,E402
