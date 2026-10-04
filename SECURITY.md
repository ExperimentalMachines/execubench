# Security

## Reporting

Report a vulnerability or a data leak privately through GitHub's "Report a vulnerability"
(Security tab) on this repository, or by email to the maintainer listed on the GitHub profile.
Do not open a public issue for it. Expect an acknowledgement within a week.

Leaks that count: credentials or tokens, an AWS account number, a device serial, IMEI, MAC
address, SIM identity or any other per-unit identifier in `data/`, and anything in a pulled
artifact that identifies a person.

## What the code does to prevent leaks

- Credentials come only from the environment (AWS's credential chain, Hugging Face's own token
  store or `HF_TOKEN`); nothing in the repository reads or writes them.
- `execubench devicefarm pull` redacts per-unit identifiers in the probe's dumps and in Device
  Farm's own run and job records, masks the account number, and **refuses** a pull if an
  identifier appears in any other file (`execubench/devicefarm.py`). Tests scan every
  committed dump for identifier-shaped values.
- Downloads are size-bounded, archives are checked for path traversal, symlinks and expansion.
- Dependencies are hash-pinned (`requirements/*.lock`) and CI installs with `--require-hashes`;
  GitHub Actions are pinned to commit SHAs and run with read-only permissions.

## Data retention

See `docs/DATA-POLICY.md`.
