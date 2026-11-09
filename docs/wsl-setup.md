# Running this project in WSL2

Migration target: the `Ubuntu` distribution (Ubuntu 26.04.1), user `jerrycao`.
The separate `Ubuntu-24.04` distribution is not the target. The Linux project
directory is `/home/jerrycao/projects/demo-new`. The duplicate Windows working files were subsequently moved to the Recycle Bin
at the user's request. The old Windows folder retains only OPEN_WSL.cmd, a
short README and a protected .git metadata directory. Use the Linux copy for
all project and Git work.

## Open and activate

From PowerShell:

```powershell
wsl -d Ubuntu --cd /home/jerrycao/projects/demo-new
```

Then, inside Ubuntu:

```bash
source .venv/bin/activate
python --version
python -m pip check
```

To open the Linux directory in VS Code, run `code .` from that activated Ubuntu
terminal. In Windows Explorer the project is accessible at
`\\wsl.localhost\Ubuntu\home\jerrycao\projects\demo-new`.

Use the Linux `.venv/bin/python`, not the old Windows `.venv/Scripts/python.exe`.
The project uses Python 3.11.7; Ubuntu's system Python 3.14 is outside the
supported project range. A separate managed interpreter avoids replacing it.

## Environment and data

The setup retains Flower 1.36.0, Ray 2.55.1, CPU PyTorch 2.6.0 and TorchVision
0.21.0. Development, plotting and PDF dependencies are installed as well.
The current machine has Intel Arc graphics and the existing project runs on
CPU. Installing the NVIDIA CUDA package stack is unnecessary for this setup.

The Windows virtual environment and disposable runtime caches are excluded
from the copy. Git history, code, both image datasets, prepared subsets,
feature caches, pretrained weights, initial head and saved outputs are copied.
The cached features are reused rather than regenerated.

Active `data/**/manifest.json` image roots are relocated to Linux paths.
Original manifest bytes and before/after hashes are preserved under
`outputs/wsl-migration-20261028/`. Cached tensors, weights and historical source
hashes remain unchanged. The relocated manifest has a new checksum, so fresh
Linux experiments must record it as a new environment/input instance.

Historical `outputs/` are retained as records, including embedded Windows
paths. Do not rewrite them to impersonate Linux runs or try to resume a frozen
Windows batch under different platform/manifest checksums. Historical tools that resolve those paths directly need explicit path
relocation or restoration of their Windows inputs before use. Fresh WSL runs write their own immutable output directories.

## Commands

The existing CLI options are unchanged. For example, a deliberately small
synthetic adaptive runtime check is:

```bash
python -m flower_face.check_adaptive_llz --output-dir outputs/wsl-validation
```

That check contains exactly three two-round/four-client runs and is environment
validation, not DigiFace accuracy evidence or calibration of a new thesis budget.
Migration performs it once; do not repeat it unless resolving an environment
issue. Source filenames remain unchanged, and Bash uses `/` path separators.

To inspect a command's options without starting training:

```bash
python -m flower_face.run --help
python -m flower_face.compression_study --help
```

Keep the latest supervisor priorities: justify the adaptive distortion budget,
prepare a cited PDF with graphs/tables before meetings, and show the evidence
chain step by step. Environment migration does not select a new budget or
authorize an additional scientific experiment batch.

## Reproducibility records

The migration report, installed package list, manifest-relocation mapping,
bounded test log and runtime validation live under
`outputs/wsl-migration-20261028/` in the Linux project. Generated data/results
remain ignored by Git. Migration initially preserved the Windows copy. The
subsequent requested cleanup recycled the duplicate working files, retaining
protected metadata. See outputs/wsl-migration-20261028/windows-cleanup.json.

## GitHub account

WSL Git is configured for the existing GitHub account `JerryFhuurr`, with
commit name `Jerry Cao` and email
`79422097+JerryFhuurr@users.noreply.github.com`. These defaults are also set
for future repositories in this Ubuntu user account.

HTTPS authentication reuses the existing Windows GitHub CLI sign-in via the
Git credential helper. No token is stored in this project. This replaces the
Windows Git Credential Manager bridge, which failed in this WSL environment.
The repository URL is the current GitHub location and push authentication
was verified with a dry run. No commits were uploaded during account setup.

Inside the project in Ubuntu, these commands show the configured identity and
verify the current account/access:

```bash
git config user.name
gh.exe api user --jq .login
git push --dry-run origin HEAD:main
```

`gh.exe` uses the installed Windows GitHub CLI. Ordinary `git` commands use
native Linux Git. Account/setup verification is recorded in
`outputs/wsl-migration-20261028/github-setup.json`.

Upstream references: [Microsoft WSL file-system guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems),
[uv managed Python](https://docs.astral.sh/uv/guides/install-python/),
[official PyTorch 2.6 installation commands](https://pytorch.org/get-started/previous-versions/),
and [GitHub CLI credential-helper configuration](https://cli.github.com/manual/gh_auth_setup-git).
