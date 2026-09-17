# Project working guidelines

- Read only files relevant to the requested change.
- Complete the agreed experiment batch, audit its results once, and report once.
- Do not add experiments beyond that batch without a new user request.
- Reuse existing datasets, feature caches, and valid completed results.
- Run affected tests after code changes; do not rerun the full suite for
  documentation or hyperparameter-only changes.
- While jobs run, complete independent work already within scope.
- Prefer completion signals; avoid redundant polling and repeated log reads.
- Save generated results under outputs/ and keep them excluded from Git.
