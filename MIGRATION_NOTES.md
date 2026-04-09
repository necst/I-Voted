# MLIR-AIE porting notes

## Baseline
- Goal: keep mlir-aie as external submodule/toolchain, keep I-Voted as app/example repo
- Current app sanity check: `cd ivoted && make run_py`

## Record
- `git submodule status`
- `python -m pip show llvm-aie mlir_aie`
- `which aiecc`
- `which aiecc.py`

## Known local workaround
- PEANO triple compatibility symlink fix currently needed for wheel layout mismatch
