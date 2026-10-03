# Probe runs

Raw device dumps from the probe script, pulled with `execubench devicefarm pull`. At pull time
the unit serial becomes `unit-<hash>`, other per-device identifiers (IMEIs, SIM ICCIDs, AP,
DDR and chip serials) become `<redacted>`, the AWS account number becomes `<account>`, and
adb's per-percent push progress lines are dropped (`execubench/devicefarm.py`, `scrub`);
nothing else is changed. Each run folder holds the exact `probe.sh` and `testspec.yml` it ran
(each probe script was checked by sha256 against the package Device Farm kept) and Device
Farm's own run record, `devicefarm-run.json`.

| Folder | Device Farm run id | Script | Devices | Jobs passed | Metered minutes |
|---|---|---|---|---|---|
| `2026-10-04/` | `86b8ddf5-d1b1-4b54-a199-f6ebbab3fe18` | v1 (`probe.sh` sha256 `9e93c1e5...`) | 19 candidates | 19 of 19 | 8.41 |
| `2026-10-04-v2/` | `7af87807-ce27-4e13-872a-6ef48ed066fe` | v2 (`5b390d6e...`): adds GPU via Vulkan, NPU runtimes, `dumpsys meminfo`, simpleperf | the same 19 | 19 of 19 | 9.24 |
| `2026-10-04-v3-samsung/` | `b55d5a66-a5e8-4481-bcfb-dd151f9915ec` | v3 (`d74a59a8...`): no on-phone pipelines, verified push, required dumps fail the job | 16 Samsung A-series and Note models | 16 of 16 | 8.86 |
| `2026-10-04-v3-catalogue/` | `8a979433-95de-46e0-9971-b56c02b21c36` | v3 (`d74a59a8...`) | every other arm64 Android model (48) | 43 of 48 | 25.33 |
| `2026-10-04-v3.1-check/` | `89c2924e-83d1-4e35-b932-834f0c848ead` | v3.1 (`f70f2cf7...`, the committed script): per-core coverage, exit codes, thermal service optional below Android 10 | Galaxy S25 Ultra and Pixel 2 XL (Android 8.1) | 2 of 2 | 1.11 |

The five failed jobs in the catalogue run are Android 8.1 and 9 phones without a thermal
service, which v3 treated as a required dump; their other dumps are complete, and v3.1 (the
committed `devicefarm/probe/probe.sh`) treats it as optional below Android 10. Run-level
metered minutes are Device Farm's own totals; job-level sums differ by rounding (8.43 and
9.25 for the first two). The first two runs used Device Farm's default video capture and the
execuserve APK as carrier; the v3 runs disabled video capture and used the manifest-only
carrier in `devicefarm/carrier/`.

Each device folder holds `devicefarm-job.json` (Device Farm's job record, verbatim but for the
account number), `unit.txt` (the unit hash), `artifacts/probe/` (raw dumps) and
`profile.json` (parsed by `execubench/devices.py`; regenerate after any parser change, the
tests check it is current).
