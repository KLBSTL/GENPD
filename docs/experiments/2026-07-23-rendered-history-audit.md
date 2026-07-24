# Historical Rendered Result Audit

`paper-20260723-rendered` remains a read-only historical result root. It was
produced before the NCG line-search and invalid-state fixes, so it is not an R2
evidence source.

Run `scripts/audit_historical_rendered_results.ps1` to derive a separate audit
manifest and validity table. The script preserves every legacy CSV and writes
the following classifications:

- `qualified`: a legacy candidate passed the old position-error gate and had no
  invalid frame.
- `E0: frame-0 explosion`: the first profile row has `exploded=1`; it is an
  engineering failure, not an `NQ` quality result.
- `quality_gate_not_met`: no legacy candidate qualified without an E0 failure.

The four hanging-cloth `386^2` gather-derived cases are `E0`. They are excluded
from performance rankings, ablations, and all R2 figure inputs. New paper
figures may show them only in a clearly labelled historical validity panel.
