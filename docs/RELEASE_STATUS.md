# Release status — private research preview

**GitHub integration:** the validated production scripts, HPC launchers, selected derived CSVs, and 12 publication figures (PNG + SVG) have been synchronized from the uploaded Swan review archive. This is a private research preview, not a peer-reviewed or submission-ready public release.

## Included

- Swan production B1, fixed B4 and inflight B4/B8/B16 launchers and corresponding source code
- 500-structure deterministic manifest, 10-structure numerical-equivalence diagnostic
- Production performance summary and B1/inflight matched comparison CSV
- Geometry comparison CSV and outlier tables, but **not** the underlying final coordinate files
- 12 publication figures in PNG and SVG; draft manuscript and scientific limitations

## Still required before public release

- Independently test reproduction from a clean environment and pin the full dependency lockfile
- Verify OC20-Dense and UMA checkpoint redistribution and attribution terms
- Resolve production exit-state vs geometry-capture post-hoc force-threshold bookkeeping
- Audit manuscript references, figures, author list and scientific claims
- Decide whether and how to share final structure coordinates; without them, some figures cannot be regenerated from the public repo alone

**No unpublished PFAS calculations or raw proprietary data are included.**
