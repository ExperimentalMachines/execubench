"""The benchmark harness: runs on the Device Farm host, drives ExecuServe on the phone over adb.

Not yet run on a device (phase P1). Every piece that does not need the ExecuServe benchmark
build (docs/EXECUSERVE-CONTRACT.md) is implemented and unit-tested here: adb calls, staging
with hash checks on host and phone, the state sampler, the streaming client and the record
writer. `run.py` refuses to start against a server that does not advertise the contract.
"""
