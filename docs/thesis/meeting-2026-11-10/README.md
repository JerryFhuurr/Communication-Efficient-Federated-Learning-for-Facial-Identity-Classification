# Supervisor meeting — 10 November 2026

Updated 5 November with completed WSL development, budget sensitivity and frozen confirmation evidence. The earlier Windows-pilot deck remains historical and should not be used as the current meeting presentation.

Current package: `outputs/meeting-20261110-20261105/`. Its README contains the run settings, original evidence locations, reproduction commands and source identity. The editable PowerPoint contains seven main slides for five minutes and two appendix slides. It includes the literature comparison, budget rationale, real saved-update example and per-seed confirmation results.

`presentation-notes.md` contains the timed script and Q&A. `confirmation-protocol.md` preserves the frozen setup and pass/fail results. `reproduction.md` and `saved-update-walkthrough.md` point to the current package.

Main finding: adaptive tau=5.9 saves about 92% of serialized training uploads, but fails the predeclared 1.0 percentage-point per-seed accuracy gate on seed 44. Do not tune on those confirmation results or describe the policy as consistently preserving accuracy.

The literature review is available at https://www.overleaf.com/project/6aeb378d04c7d5dba6f6b8f9 . Its original experimental snapshot precedes the November 3 sweep/confirmation. The meeting slides provide the newer evidence.

The package is prepared for user review and sharing by 9 November evening. No email or external sharing has been performed.
