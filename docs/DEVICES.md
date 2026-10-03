# Devices: the standard, and what the phones say about themselves

On 2026-10-04 every arm64 Android model in AWS Device Farm's catalogue was probed: 83 models
in four runs (`data/devices/probe/README.md` lists them). The catalogue itself is snapshotted in
`data/devices/catalogue/2026-10-04.json` (181 devices, 106 of them Android entries; one, the
32-bit Galaxy Tab A 10.1, was skipped). Everything in the "reported" columns comes from the
phones; parsed records are in `data/devices/devices.json`, which validates against
`schemas/device.schema.json`. Published values come only from chip and phone makers, listed
with URLs in `data/devices/specs.yaml`, and only for the phones considered for the standard.
Unit serials are replaced by a hash and other identifiers are redacted before anything is
written (`execubench/devicefarm.py`).

## How the device fields are derived

| Field | Source | Rule |
|---|---|---|
| SoC (reported) | `ro.soc.manufacturer`, `ro.soc.model`, `ro.board.platform` | Verbatim. Phones before Android 12 often leave `ro.soc.*` empty; the platform code remains |
| SoC marketing name | `specs.yaml` | The phone maker's name for the chip in that phone where it gives one ("for Galaxy" variants), else the chip maker's |
| CPU count | `/sys/devices/system/cpu/cpu*` | Cores present |
| Core design | `MIDR_EL1` per core, named by the Linux kernel's `arch/arm64/include/asm/cputype.h` (pinned in `data/reference/`) | The kernel macro name, e.g. `CORTEX_X4`; `VENDOR:0xPART` where the kernel does not list the part |
| Core class | Arm's and Qualcomm's own classification of each design (`execubench/devices.py`) | efficiency, performance or unknown; a design in neither list is unknown, never assumed big |
| Clock | `cpuinfo_max_freq` per core | Never Device Farm's catalogue `cpu.clock`, which is the first cluster's clock |
| Topology and cluster count | cpufreq policies, cross-checked against each core's `related_cpus` | A cluster is a clock domain: cores that must share one frequency. If the two sources disagree the topology is left unknown |
| Tiers | Cores grouped by design and top clock | Kept beside topology, because the two differ on some chips |
| All big cores | Core classes | False if any core is efficiency-class; true only if every core is known performance-class; otherwise unknown |
| Architecture label | The two fields above | e.g. "all-big-core, tri-cluster", "big.LITTLE, quad-cluster" |
| GPU | Vulkan `deviceName` from `cmd gpu vkjson` | `reported`; the GLES renderer string is kept too |
| NPU name and TOPS | `specs.yaml` | `published` or `unknown`; TOPS only with a stated precision |
| NPU runtimes present | Vendor libraries in `/vendor/lib64` and friends | Which vendor NPU stacks a phone carries (reported) |
| arm64 userspace | `ro.product.cpu.abilist` | ExecuTorch's Android builds are arm64-v8a only |
| RAM, usable | `MemTotal` | GiB, `reported`. Smaller than marketed RAM, because firmware carve-outs are excluded |
| RAM, marketed | `specs.yaml` | GB as the phone maker states it, per model code |
| Memory type and fitted data rate | Bootloader property `ro.boot.hardware.ddr` where present (Pixels), else the phone maker | The chip's supported types and maximum rate are not the fitted DRAM |
| Memory bandwidth | Not found in maker sources reviewed | See below |
| BF16 TFLOPS | Not found in maker sources reviewed | See below |
| Device id | Model id, Android version and the first 8 hex digits of sha256(build fingerprint) | A new firmware build is a new device id; results are never pooled across ids |

### Why Device Farm's catalogue is not a spec source

`aws devicefarm list-devices` gives every device a `memory` and a `cpu.clock`. In the snapshot,
the Android `memory` values range from 16 to 512 GB: that is storage. `cpu.clock` is the first
cluster's maximum: the Pixel 10's Tensor G5 is listed at 2,246 MHz, which is its Cortex-A520
pair; the X4 core runs at 3,782 MHz. The catalogue's `cpu.architecture` can also matter: the
Galaxy A13 5G is listed as `armeabi-v7a` and indeed runs a 32-bit-only userspace on Cortex-A76
cores, so arm64 ExecuTorch cannot run there. Names can differ from the maker's: model
23090RA98G is "Xiaomi Redmi Note 13+" on Device Farm and the Redmi Note 13 Pro+ 5G at Xiaomi.

### Why memory bandwidth and BF16 TFLOPS are `unknown`

An independent review (Codex, gpt-6.1-sol, with web search) found, and our own checks of the
Qualcomm, MediaTek, Samsung and Google pages agree, that in the maker sources reviewed for the
12 chips of the candidate set:

- **No BF16 or FP16 TFLOPS figure is published** for the CPU, GPU or NPU. Where an NPU figure
  is published (Exynos 1580: "up to 14.7 TOPS"), its precision is not stated, so it is not an
  INT8 figure and cannot be converted to FLOPS.
- **No DRAM bus width is published**, so theoretical bandwidth cannot be derived. Makers state
  the controller's maximum supported rate (Qualcomm in "MHz" without saying MT/s, MediaTek in
  Mbps), which is not the speed of the DRAM a given phone has fitted. One phone maker states the
  fitted rate (Xiaomi 13: LPDDR5X at 8,533 Mbps).

So the `bf16_tflops` and `bandwidth_gbps` columns stay in the schema, as asked, with `unknown`
and a reason. What would fill them honestly, in order of preference:

1. **Measured** (planned for v2): a microbenchmark on the device, such as a BF16 GEMM through
   XNNPACK and a streaming-memory kernel, with clocks and thermal state recorded. Labelled
   "achieved", per engine.
2. **Derived CPU peak**, only where every factor is sourced: cores x clock x BF16 instructions
   per cycle (from the core's Software Optimization Guide) x FLOPs per instruction (128-bit
   BFMMLA is 32 FLOPs, BFDOT 16). Not available for Oryon, whose throughput tables are not
   public. SME needs care: Cortex-X925 does not implement it, and C1's SME2 is a shared unit, so
   a per-core sum would overcount.

## What the probes found

The table is generated from `devices.json` by `execubench devices table`; a test fails if it
drifts. "SoC (published name)" is filled only for the candidates in `specs.yaml`.

<!-- devices-table:begin (generated by `execubench devices table`; do not edit) -->

| Device (model id) | Android | SoC reported | SoC (published name) | Topology | Tiers | Architecture | Top MHz | GPU (reported) | MemTotal GiB | Marketed RAM GB | arm64 |
|---|---|---|---|---|---|---|---:|---|---:|---|---|
| Google Pixel 10 (GLBW0) | 16 | Google Tensor G5 | Google Tensor G5 | 1+3+2+2 | 1+5+2 | big.LITTLE, quad-cluster | 3782.0 | PowerVR D-Series DXT-48-1536 MC1 | 11.27 | 12 | yes |
| Google Pixel 10 Pro (G4QUR) | 16 | Google Tensor G5 | Google Tensor G5 | 1+3+2+2 | 1+5+2 | big.LITTLE, quad-cluster | 3782.0 | PowerVR D-Series DXT-48-1536 MC1 | 15.18 | 16 | yes |
| Google Pixel 10 Pro XL (GUL82) | 16 | Google Tensor G5 | Google Tensor G5 | 1+3+2+2 | 1+5+2 | big.LITTLE, quad-cluster | 3782.0 | PowerVR D-Series DXT-48-1536 MC1 | 15.18 | 16 | yes |
| Google Pixel 11 (GUJ0N) | 17 | Google Tensor G6 | Google Tensor G6 | 1+4+2 | 1+4+2 | all-big-core, tri-cluster | 4109.0 | PowerVR C-Series CXTP-48-1536 MC1 | 11.37 | 12 | yes |
| Google Pixel 11 Pro (G7SWN) | 17 | Google Tensor G6 | Google Tensor G6 | 1+4+2 | 1+4+2 | all-big-core, tri-cluster | 4109.0 | PowerVR C-Series CXTP-48-1536 MC1 | 11.37 | 12 (256 GB storage); 16 (512 GB, 1 TB) | yes |
| Google Pixel 11 Pro XL (G4HCD) | 17 | Google Tensor G6 | Google Tensor G6 | 1+4+2 | 1+4+2 | all-big-core, tri-cluster | 4109.0 | PowerVR C-Series CXTP-48-1536 MC1 | 11.37 | 12 (256 GB storage); 16 (512 GB, 1 TB) | yes |
| Google Pixel 2 XL (Google Pixel 2 XL) | 8.1.0 | msm8998 | unknown | 4+4 | 4+4 | core classes unknown, dual-cluster | 2457.6 | Adreno (TM) 540 | 3.56 | unknown | yes |
| Google Pixel 3 (Pixel 3) | 10 | sdm845 | unknown | 4+4 | 4+4 | big.LITTLE, dual-cluster | 2803.2 | Adreno (TM) 630 | 3.50 | unknown | yes |
| Google Pixel 3 XL (Pixel 3 XL) | 10 | sdm845 | unknown | 4+4 | 4+4 | big.LITTLE, dual-cluster | 2803.2 | Adreno (TM) 630 | 3.46 | unknown | yes |
| Google Pixel 3a (G020G) | 10 | sdm710 | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 1996.8 | Adreno (TM) 615 | 3.51 | unknown | yes |
| Google Pixel 3a XL (G020C) | 12 | Qualcomm SDM670 | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 1996.8 | Adreno (TM) 615 | 3.51 | unknown | yes |
| Google Pixel 4 (Unlocked) (GA01188-US) | 11 | msmnile | unknown | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 2841.6 | Adreno (TM) 640 | 5.34 | unknown | yes |
| Google Pixel 4a (Pixel 4a) | 11 | sm6150 | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2208.0 | Adreno (TM) 618 | 5.46 | unknown | yes |
| Google Pixel 5 (Unlocked) (Pixel 5) | 12 | Qualcomm SM7250 | unknown | 1+1+6 | 1+1+6 | big.LITTLE, tri-cluster | 2400.0 | Adreno (TM) 620 | 7.29 | unknown | yes |
| Google Pixel 5a 5G (GA02618US) | 12 | Qualcomm SM7250 | unknown | 1+1+6 | 1+1+6 | big.LITTLE, tri-cluster | 2400.0 | Adreno (TM) 620 | 5.33 | unknown | yes |
| Google Pixel 6 (Unlocked) (Pixel 6) | 13 | Google Tensor | unknown | 2+2+4 | 2+2+4 | big.LITTLE, tri-cluster | 2802.0 | Mali-G78 | 7.41 | unknown | yes |
| Google Pixel 6 Pro (Unlocked) (Pixel 6 Pro) | 17 | Google Tensor | unknown | 2+2+4 | 2+2+4 | big.LITTLE, tri-cluster | 2802.0 | Mali-G78 | 11.28 | unknown | yes |
| Google Pixel 7 (GQML3) | 14 | Google GS201 | unknown | 2+2+4 | 2+2+4 | big.LITTLE, tri-cluster | 2850.0 | Mali-G710 | 7.30 | unknown | yes |
| Google Pixel 7 Pro (GE2AE) | 13 | Google GS201 | unknown | 2+2+4 | 2+2+4 | big.LITTLE, tri-cluster | 2850.0 | Mali-G710 | 11.15 | unknown | yes |
| Google Pixel 7a (GWKK3) | 13 | Google GS201 | unknown | 2+2+4 | 2+2+4 | big.LITTLE, tri-cluster | 2850.0 | Mali-G710 | 7.25 | unknown | yes |
| Google Pixel 8 (G9BQD) | 15 | Google Tensor G3 | unknown | 1+4+4 | 1+4+4 | big.LITTLE, tri-cluster | 2914.0 | Mali-G715 | 7.42 | unknown | yes |
| Google Pixel 8 Pro (G1MNW) | 15 | Google Tensor G3 | unknown | 1+4+4 | 1+4+4 | big.LITTLE, tri-cluster | 2914.0 | Mali-G715 | 11.28 | unknown | yes |
| Google Pixel 8a (GKV4X) | 17 | Google Tensor G3 | unknown | 1+4+4 | 1+4+4 | big.LITTLE, tri-cluster | 2914.0 | Mali-G715 | 7.40 | unknown | yes |
| Google Pixel 9 (GUR25) | 16 | Google Tensor G4 | Google Tensor G4 | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 3105.0 | Mali-G715 | 11.28 | 12 | yes |
| Google Pixel 9 Pro (GEC77) | 15 | Google Tensor G4 | Google Tensor G4 | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 3105.0 | Mali-G715 | 15.19 | unknown | yes |
| Google Pixel 9 Pro XL (GZC4K) | 17 | Google Tensor G4 | Google Tensor G4 | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 3105.0 | Mali-G715 | 15.22 | unknown | yes |
| Google Pixel 9a (GXQ96) | 15 | Google Tensor G4 | Google Tensor G4 | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 3105.0 | Mali-G715 | 7.38 | unknown | yes |
| Google Pixel Tablet (GTU8P) | 17 | Google GS201 | unknown | 2+2+4 | 2+2+4 | big.LITTLE, tri-cluster | 2850.0 | Mali-G710 | 7.27 | unknown | yes |
| LG Stylo 5 (LM-Q720QM) | 9 | msm8953 | unknown | 8 | 8 | big.LITTLE, single-cluster | 1804.8 | Adreno (TM) 506 | 2.75 | unknown | yes |
| LG Stylo 6 (LMQ730) | 10 | mt6765 | unknown | 4+4 | 4+4 | big.LITTLE, dual-cluster | 2301.0 | PowerVR Rogue GE8320 | 2.73 | unknown | yes |
| Samsung A51 (SM-A515F,SM-A515U1) | 13 | Samsung Exynos 9611 | unknown | 4+4 | 4+4 | big.LITTLE, dual-cluster | 2314.0 | Mali-G72 | 3.49 | unknown | yes |
| Samsung Galaxy A13 5G (SM-A136U1) | 11 | mt6833 | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2203.0 | Mali-G57 MC2 | 3.54 | unknown | no |
| Samsung Galaxy A14 5G (SM-A146U1) | 13 | Mediatek MT6833V/NZA | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2203.0 | Mali-G57 MC2 | 3.49 | unknown | yes |
| Samsung Galaxy A15 (SM-A156U1) | 14 | Mediatek MT6835 | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2200.0 | Mali-G57 MC2 | 3.50 | unknown | yes |
| Samsung Galaxy A16 (SM-A165F) | 14 | Mediatek MT6789V/CD | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2200.0 | Mali-G57 MC2 | 3.54 | unknown | yes |
| Samsung Galaxy A17 (SM-A176U) | 16 | Samsung s5e8535 | Exynos 1330 | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2400.0 | Mali-G68 | 3.37 | 4 | yes |
| Samsung Galaxy A24 4G (SM-A245M) | 14 | Mediatek MT6789V/CD | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2200.0 | Mali-G57 MC2 | 3.57 | unknown | yes |
| Samsung Galaxy A25 (SM-A256E) | 14 | Samsung s5e8825 | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2400.0 | Mali-G68 | 5.26 | unknown | yes |
| Samsung Galaxy A26 (SM-A266U1) | 15 | Samsung s5e8835 | unknown | 4+4 | 4+4 | big.LITTLE, dual-cluster | 2400.0 | Mali-G68 | 5.30 | unknown | yes |
| Samsung Galaxy A34 (SM-A346B) | 13 | Mediatek MT6877V/TTZA | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2600.0 | Mali-G68 MC4 | 7.32 | unknown | yes |
| Samsung Galaxy A35 (SM-A356U1) | 14 | Samsung s5e8835 | unknown | 4+4 | 4+4 | big.LITTLE, dual-cluster | 2400.0 | Mali-G68 | 5.30 | unknown | yes |
| Samsung Galaxy A36 (SM-A366U1) | 16 | QTI SM6475 | Snapdragon 6 Gen 3 | 4+4 | 4+4 | big.LITTLE, dual-cluster | 2400.0 | Adreno (TM) 710 | 5.26 | 6 | yes |
| Samsung Galaxy A53 5G (SM-A536U1) | 12 | Samsung s5e8825 | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2400.0 | Mali-G68 | 5.26 | unknown | yes |
| Samsung Galaxy A54 (SM-A546U1) | 13 | Samsung s5e8835 | unknown | 4+4 | 4+4 | big.LITTLE, dual-cluster | 2400.0 | Mali-G68 | 5.29 | unknown | yes |
| Samsung Galaxy A55 (SM-A556U1) | 14 | Samsung s5e8845 | unknown | 4+4 | 4+4 | big.LITTLE, dual-cluster | 2750.0 | Samsung Xclipse 530 | 7.26 | unknown | yes |
| Samsung Galaxy A56 (SM-A566U1) | 15 | Samsung s5e8855 | Exynos 1580 | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 2910.0 | Samsung Xclipse 540 | 7.25 | 8 | yes |
| Samsung Galaxy A71 (SM-A715F/DS) | 11 | sm6150 | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2208.0 | Adreno (TM) 618 | 5.40 | unknown | yes |
| Samsung Galaxy A73 5G (SM-A736B) | 12 | QTI SM7325 | unknown | 3+1+4 | 4+4 | big.LITTLE, tri-cluster | 2400.0 | Adreno (TM) 642L | 7.13 | unknown | yes |
| Samsung Galaxy Note20 (SM-N980F) | 11 | universal990 | unknown | 2+2+4 | 2+2+4 | big.LITTLE, tri-cluster | 2730.0 | Mali-G77 | 7.27 | unknown | yes |
| Samsung Galaxy S10 (SM-G973U1) | 9 | msmnile | unknown | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 2841.6 | Adreno (TM) 640 | 7.30 | unknown | yes |
| Samsung Galaxy S20 (Unlocked) (SM-G981U1) | 13 | QTI SM8250 | unknown | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 2841.6 | Adreno (TM) 650 | 10.35 | unknown | yes |
| Samsung Galaxy S21 (SM-G991U1) | 12 | QTI SM8350 | unknown | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 2841.6 | Adreno (TM) 660 | 7.19 | unknown | yes |
| Samsung Galaxy S21 Ultra (SM-G998U1) | 12 | QTI SM8350 | unknown | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 2841.6 | Adreno (TM) 660 | 10.07 | unknown | yes |
| Samsung Galaxy S22 5G (SM-S901U1) | 13 | QTI SM8450 | unknown | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 2995.2 | Adreno (TM) 730 | 7.06 | unknown | yes |
| Samsung Galaxy S22 Ultra 5G (SM-S908U1) | 12 | QTI SM8450 | unknown | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 2995.2 | Adreno (TM) 730 | 7.05 | unknown | yes |
| Samsung Galaxy S22+ 5G (SM-S906U1) | 12 | QTI SM8450 | unknown | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 2995.2 | Adreno (TM) 730 | 7.06 | unknown | yes |
| Samsung Galaxy S23 (SM-S911U1) | 14 | QTI SM8550 | Snapdragon 8 Gen 2 | 1+4+3 | 1+2+2+3 | big.LITTLE, tri-cluster | 3360.0 | Adreno (TM) 740 | 6.89 | unknown | yes |
| Samsung Galaxy S23 Ultra (SM-S918U1) | 13 | QTI SM8550 | Snapdragon 8 Gen 2 | 1+4+3 | 1+2+2+3 | big.LITTLE, tri-cluster | 3360.0 | Adreno (TM) 740 | 10.80 | unknown | yes |
| Samsung Galaxy S23+ (SM-S916U1) | 13 | QTI SM8550 | Snapdragon 8 Gen 2 | 1+4+3 | 1+2+2+3 | big.LITTLE, tri-cluster | 3360.0 | Adreno (TM) 740 | 6.91 | unknown | yes |
| Samsung Galaxy S24 (SM-S921U1) | 14 | QTI SM8650 | Snapdragon 8 Gen 3 | 1+3+2+2 | 1+3+2+2 | big.LITTLE, quad-cluster | 3398.4 | Adreno (TM) 750 | 6.93 | unknown | yes |
| Samsung Galaxy S24 Ultra (SM-S928U1) | 14 | QTI SM8650 | Snapdragon 8 Gen 3 for Galaxy | 1+3+2+2 | 1+3+2+2 | big.LITTLE, quad-cluster | 3398.4 | Adreno (TM) 750 | 10.83 | 12 | yes |
| Samsung Galaxy S24+ (SM-S926U1) | 14 | QTI SM8650 | Snapdragon 8 Gen 3 | 1+3+2+2 | 1+3+2+2 | big.LITTLE, quad-cluster | 3398.4 | Adreno (TM) 750 | 10.81 | unknown | yes |
| Samsung Galaxy S25 (SM-S931U1) | 16 | QTI SM8750 | Snapdragon 8 Elite for Galaxy | 2+6 | 2+6 | all-big-core, dual-cluster | 4473.6 | Adreno (TM) 830 | 10.85 | 12 | yes |
| Samsung Galaxy S25 Ultra (SM-S938U1) | 15 | QTI SM8750 | Snapdragon 8 Elite for Galaxy | 2+6 | 2+6 | all-big-core, dual-cluster | 4473.6 | Adreno (TM) 830 | 10.86 | 12 | yes |
| Samsung Galaxy S25+ (SM-S936U1) | 15 | QTI SM8750 | Snapdragon 8 Elite | 2+6 | 2+6 | all-big-core, dual-cluster | 4473.6 | Adreno (TM) 830 | 10.86 | unknown | yes |
| Samsung Galaxy S26 (SM-S942U1) | 16 | QTI SM8850 | Snapdragon 8 Elite Gen 5 for Galaxy | 2+6 | 2+6 | all-big-core, dual-cluster | 4742.4 | Adreno (TM) 840 | 10.86 | 12 | yes |
| Samsung Galaxy S26 Ultra (SM-S948U1) | 16 | QTI SM8850 | Snapdragon 8 Elite Gen 5 for Galaxy | 2+6 | 2+6 | all-big-core, dual-cluster | 4742.4 | Adreno (TM) 840 | 10.86 | 12 (256/512 GB storage); 16 (1 TB) | yes |
| Samsung Galaxy S26+ (SM-S947U1) | 16 | QTI SM8850 | Snapdragon 8 Elite Gen 5 | 2+6 | 2+6 | all-big-core, dual-cluster | 4742.4 | Adreno (TM) 840 | 10.86 | unknown | yes |
| Samsung Galaxy S9 (Unlocked) (SM-G960U1) | 9 | sdm845 | unknown | 4+4 | 4+4 | big.LITTLE, dual-cluster | 2803.2 | Adreno (TM) 630 | 3.50 | unknown | yes |
| Samsung Galaxy S9+ (Unlocked) (SM-G965U1,SM-G965U) | 8.0.0 | sdm845 | unknown | 4+4 | 4+4 | big.LITTLE, dual-cluster | 2803.2 | Adreno (TM) 630 | 4.89 | unknown | yes |
| Samsung Galaxy Tab A7 Lite (SM-T220) | 13 | Mediatek MT8768WT | unknown | 4+4 | 4+4 | big.LITTLE, dual-cluster | 2301.0 | PowerVR Rogue GE8320 | 2.79 | unknown | yes |
| Samsung Galaxy Tab A9 (SM-X110) | 13 | Mediatek MT8781V/NA | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2200.0 | Mali-G57 MC2 | 3.65 | unknown | yes |
| Samsung Galaxy Tab S11 (SM-X730) | 16 | Mediatek MT6991 | MediaTek Dimensity 9400+ (MT6991) | 1+3+4 | 1+3+4 | all-big-core, tri-cluster | 3730.0 | Mali-G925-Immortalis MC12 | 11.25 | 12 | yes |
| Samsung Galaxy Tab S6 (WiFi) (SM-T860) | 9 | msmnile | unknown | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 2841.6 | Adreno (TM) 640 | 5.32 | unknown | yes |
| Samsung Galaxy Tab S7 (SM-T870) | 11 | kona | unknown | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 3091.2 | Adreno (TM) 650 | 5.52 | unknown | yes |
| Samsung Galaxy Tab S8 (SM-X700) | 12 | QTI SM8450 | unknown | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 2995.2 | Adreno (TM) 730 | 7.13 | unknown | yes |
| Samsung Galaxy Tab S9 (SM-X710) | 13 | QTI SM8550 | Snapdragon 8 Gen 2 for Galaxy | 1+4+3 | 1+2+2+3 | big.LITTLE, tri-cluster | 3360.0 | Adreno (TM) 740 | 7.05 | 8 (128 GB storage); 12 (256 GB) | yes |
| Xiaomi 12 (2201123G) | 12 | QTI SM8450 | unknown | 1+3+4 | 1+3+4 | big.LITTLE, tri-cluster | 2995.2 | Adreno (TM) 730 | 7.03 | unknown | yes |
| Xiaomi 13 (2211133G) | 13 | QTI SM8550 | Snapdragon 8 Gen 2 | 1+4+3 | 1+2+2+3 | big.LITTLE, tri-cluster | 3187.2 | Adreno (TM) 740 | 10.90 | 8 or 12 (global listing) | yes |
| Xiaomi 13 Pro (2210132G) | 13 | QTI SM8550 | Snapdragon 8 Gen 2 | 1+4+3 | 1+2+2+3 | big.LITTLE, tri-cluster | 3187.2 | Adreno (TM) 740 | 10.90 | unknown | yes |
| Xiaomi Redmi Note 10 5G (M2103K19G) | 11 | mt6833 | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2203.0 | Mali-G57 MC2 | 7.50 | unknown | yes |
| Xiaomi Redmi Note 13 (2312DRAABC) | 15 | Mediatek MT6833 | unknown | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2400.0 | Mali-G57 MC2 | 5.44 | unknown | yes |
| Xiaomi Redmi Note 13 Pro+ 5G (23090RA98G) | 15 | Mediatek MT6886 | Dimensity 7200-Ultra | 2+6 | 2+6 | big.LITTLE, dual-cluster | 2800.0 | Mali-G610 MC4 | 7.20 | 8 or 12 (global listing) | yes |

<!-- devices-table:end -->

Notes on the table and the candidate set:

- **Snapdragon 8 Elite (SM8750)** reports Qualcomm part `0x001` on all eight cores. The kernel
  names that part `ORYON_X1` (it was added for the Snapdragon X Elite); the record shows the
  kernel's macro name as `design` and says so in `design_source`. It is not a claim that the
  phone has an X Elite core.
- **Snapdragon 8 Elite Gen 5 (SM8850)** reports part `0x002`, which the pinned kernel does not
  list yet, so its design is `QCOM:0x002`. Its performance cores read 3,628.8 MHz against
  Qualcomm's published "up to 3.62 GHz"; we keep both, not a rounded one.
- **Tensor G6** has **seven** cores: one C1-Ultra at 4,109 MHz, four C1-Pro at 3,379 and two
  C1-Pro at 2,649. No efficiency-class core, so it is all-big-core by our rule. Google does not
  publish the configuration, so this is device evidence only.
- **Tensor G5** is 1+5+2 by design and clock, but its five Cortex-A725 cores sit in **two**
  clock domains (three and two), so its topology is 1+3+2+2.
- **The Dimensity 9400+ is pinned by clock.** MT6991 is the part code of both the Dimensity
  9400 (prime up to 3.62 GHz) and the 9400+ (up to 3.73 GHz). The Tab S11 reads 3,730 MHz and
  Samsung names the 9400+. A secondary listing (GSMArena) gives 3.63 GHz for this tablet, which
  is the non-plus clock: an example of why secondary sources are not used.
- **Matrix extensions differ.** `/proc/cpuinfo` shows `sme` (with BF16, F16, F32 and I8
  outer-product variants) on the SM8850 phones and on no other probed phone. Tensor G6's C1
  cores do not expose SME to userspace although Arm advertises SME2 for C1. The Galaxy A17 and
  A36 lack `i8mm` and `bf16`, so XNNPACK takes different kernels there; the agreement track
  will show whether that changes outputs.
- **Pixel 9 (GUR25)** is not Google's US model code (G2YBB is); Device Farm's hosting region
  says nothing about where a unit was sold.
- **Older phones** (Android 9 and earlier) have no thermal service, so probe v3 recorded its
  absence as a failed required dump on five of them; v3.1 treats it as optional below Android
  10. Their other dumps are complete. None of them is in the standard.

### Units are not interchangeable

Device Farm runs a job on any free unit of a model. Across the first two probes, **17 of 19
models landed on a different physical unit** (only the Pixel 10 Pro and the Tab S11 repeated),
and **two models differed in firmware between their units**: the Redmi Note 13 Pro+ 5G
(HyperOS `OS2.0.201.0` against `OS2.0.206.0`) and the Galaxy S24 Ultra (`S928U1UES2AXE4`
against `S928U1UES4AXKF`). `adb push` throughput also varies by unit: on the S24 Ultra, 28
MiB/s on one unit and 82 on the other. Every result therefore carries the unit hash and the
build fingerprint, and the device id changes with the build.

### Benchmark-relevant capabilities

From the second probe (`2026-10-04-v2`, the 19 candidates) unless stated.

| Capability | Finding | Consequence |
|---|---|---|
| `cmd power set-fixed-performance-mode-enabled true` | Accepted, exit 0, no output, on 19 of 19 | Whether it steadies clocks is measured in P1 |
| `perf_event_paranoid` | -1 on 19 of 19 | Necessary but not sufficient for counters |
| `simpleperf stat -a -e cpu-cycles,instructions` (system-wide, 1 s) | Failed on 19 of 19, three ways: "System wide profiling needs root privilege" (6: Galaxy S24 Ultra, S25 Ultra, Tab S9, A56, Xiaomi 13, Redmi Note 13 Pro+); "Event type 'cpu-cycles' is not supported" (8: Pixel 9, all three Pixel 10s, Galaxy S25, A36, A17, Tab S11); "Can't record kernel samples, try cpu-cycles:u" (5: all three Pixel 11s, both Galaxy S26s) | No system-wide cycle counting as probed. Probe v3 also tries user mode; P1 tests per-process counting on a `profileable` build |
| `scaling_cur_freq` per core | Readable on every core of 19 of 19 | Sampled clocks are available |
| Thermal HAL via `dumpsys thermalservice` | Thermal status, plus skin and battery temperatures in the HAL's **current** block, on 19 of 19 (names vary: `SKIN`, `skin`, `VIRTUAL-SKIN`) | Recorded during every request. The dump also prints a **cached** block that can be minutes old; it is never used as a reading |
| Temperatures before any model ran | HAL current battery 27.6 to 35.3 C; hottest CPU-type sensor per phone 29.5 to 41.3 C; one Galaxy S24 Ultra at thermal status 1 | Cooldown gate before every measured block |
| Raw thermal zones in sysfs | Readable on some phones, not on Pixels | Not relied on |
| Video capture | On by default in Device Farm jobs | Every later run sets `videoCapture=false`, since screen recording would compete for CPU and GPU |
| Vendor NPU runtimes present | Qualcomm QNN/SNPE on every Snapdragon; MediaTek Neuron/APUSys on both MediaTek phones; Google EdgeTPU on every Pixel; Samsung ENN/Eden on both Exynos phones | Starting point for a v2 NPU track |
| Host download from the Hub | 12.4 to 35.7 MB/s, median 32.0, on an 11 MB tokenizer, across the 38 jobs of the first two probes | Models are fetched by the host, not uploaded through Device Farm |
| `adb push` of 256 MiB of zeros | 25.4 to 89.1 MiB/s across those 38 jobs, 27 of them between 25 and 30 | Roughly 15 to 50 s for a 1.3 GB model, an extrapolation P1 replaces with real model pushes |

## The standard set

The previous standard came from Firebase Device Streaming: Snapdragon 8 Elite (Galaxy S25
Ultra), Tensor G5 (Pixel 10 family), Dimensity 9300+ (Galaxy Tab S10+) and Exynos 2500 (Galaxy
Z Flip7). With the whole catalogue probed:

- The Galaxy S25 Ultra and the Pixel 10 family are offered.
- **No Dimensity 9300-class chip is offered.** MediaTek platforms in the catalogue: MT6991 (Tab
  S11), MT6886, MT6877, MT6835, MT6833, MT6789 and MT6765.
- **No Exynos 2500 and no current flagship Exynos is offered.** Exynos platforms in the
  catalogue: s5e8855 (Galaxy A56, Exynos 1580 per Samsung), s5e8845 (A55), s5e8835 (A26, A35,
  A54), s5e8825 (A25, A53), s5e8535 (A17, Exynos 1330 per Samsung), Exynos 9611 (A51), and
  `universal990` on the 2020 Galaxy Note20 (SM-N980F: Samsung custom cores plus Cortex-A76 and
  A55, Android 11). Every Galaxy S-series unit carries a US model code and a Snapdragon. The A56
  is the only Exynos phone with Armv9 cores (Cortex-A720 and A520, from MIDR).

| Tier | Device | Why | Pin |
|---|---|---|---|
| A | Galaxy S25 Ultra (SM8750, 12 GB) | Same chip as the previous standard | Android 15 |
| A | Pixel 10 (Tensor G5, 12 GB) | Same chip as the previous standard; the Pro and Pro XL (16 GB) are alternates for memory-bound windows | Android 16 |
| A | Galaxy Tab S11 (Dimensity 9400+, 12 GB) | The newest MediaTek chip offered, replacing the Tab S10+ (9300+). It is a tablet, so its sustained thermals are not a phone's; stated beside every result | Android 16 |
| A | Galaxy A56 (Exynos 1580, 8 GB) | The only Exynos with Armv9 cores on offer. **Mid-range, not a stand-in for the Exynos 2500**; stated beside every result. Owner decision: keep it in Tier A or leave Exynos out of Tier A | Android 15 |
| A+ | Galaxy S26 Ultra (SM8850, 12 GB) | Current Qualcomm flagship | Android 16 |
| A+ | Pixel 11 (Tensor G6, 12 GB) | Current Google chip, all-big-core seven-core design | Android 17 |
| B | Galaxy A36 (Snapdragon 6 Gen 3, 6 GB) | Mid-range Qualcomm; no `i8mm` | Android 16 |
| B | Redmi Note 13 Pro+ 5G (Dimensity 7200-Ultra, 8 GB) | Mid-range MediaTek | Android 15 |
| B | Galaxy A17 (Exynos 1330, 4 GB) | The memory floor | Android 16 |
| C | Galaxy S24 Ultra (SM8650, 12 GB) | The chip of the 2026-09-21 cross-chip rerun in OpenWeights | Android 14 |
| C | Pixel 9 (Tensor G4, 12 GB) | Previous Google generation | Android 16 |

RAM figures are the makers' marketed capacities (`specs.yaml`). Tier A is a vendor coverage set,
not a set of equals: three flagship-class chips and one mid-range one. The other 72 probed models
remain in `devices.json` as alternates and as a record of what the catalogue held.

## Re-probing

Device Farm updates its fleet. The probe is re-run monthly and before any collection phase,
with the catalogue snapshotted each time; a changed build fingerprint or device ARN starts a
new device id, and results are never pooled across ids.
