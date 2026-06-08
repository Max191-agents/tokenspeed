# Custom Triton Build Setup Notes

Date: 2026-06-08

This documents the custom Triton setup used for the buffer-descriptor-bounds
experiment:

- Fork: `https://github.com/panditsa/triton`
- Branch: `sanketp/buffer-descriptor-bounds`
- Commit tested: `ab75b783d969e5f89b02ec14dae367210844e2cc`

## References Used

- Triton README source-build flow:
  `pip install -r python/requirements.txt`, then `pip install -e .`.
- Triton README build tips:
  `TRITON_BUILD_WITH_CLANG_LLD=true`, `TRITON_BUILD_WITH_CCACHE=true`,
  `TRITON_HOME=<path>`, `MAX_JOBS=<n>`, and `--no-build-isolation`.
- pip local-project install docs for editable installs.
- uv package-management docs for local/editable installs and requirements-file
  installs.

## Build The Fork

```bash
mkdir -p projects/triton/custom-builds
git clone --filter=blob:none --single-branch \
  --branch sanketp/buffer-descriptor-bounds \
  https://github.com/panditsa/triton.git \
  projects/triton/custom-builds/buffer-descriptor-bounds

cd projects/triton/custom-builds/buffer-descriptor-bounds
python3 -m venv --prompt triton-bdb .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r python/requirements.txt

PATH="$PWD/.venv/bin:$PATH" \
TRITON_HOME="$PWD/.triton-home" \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_WITH_CCACHE=true \
MAX_JOBS=12 \
.venv/bin/python -m pip install -e . --no-build-isolation
```

The tested build produced:

- Distribution metadata: `triton-3.7.0+gitab75b783`
- Import package: `triton`
- `triton.__version__`: `3.7.0`
- Native libraries:
  - `python/triton/_C/libtriton.so`
  - `python/triton/_C/libproton.so`

The source checkout plus local Triton cache was about 13 GB after the first
build. Most of the first build time was spent unpacking/downloading Triton's
pinned LLVM into `TRITON_HOME`.

## TokenSpeed Integration

The fork builds the import package `triton`. TokenSpeed's kernel and MLA
packages normally import `tokenspeed_triton`, so changing only the distribution
name with `TRITON_WHEEL_NAME=tokenspeed-triton` is not sufficient.

The integration added here uses:

- `TOKENSPEED_TRITON_PACKAGE=<import package>` as the selector.
- Default value: `tokenspeed_triton`.
- Custom fork value: `triton`.

`tokenspeed_kernel._triton` and `tokenspeed_mla._triton` read this selector and
import the selected package. The kernel package also redirects third-party
plain `import triton` call-sites to the selected package when its redirect
context manager is used.

The custom fork does not export `triton.aggregate` or
`triton.experimental.gluon.aggregate`; it keeps the decorator internally as
`triton.language.core._aggregate`. `tokenspeed_kernel._triton` adds a
compatibility alias when that internal decorator is present.

## Setup Helper

`claude-docs/tokenspeed/scripts/tokenspeed-setup.sh` now accepts:

```bash
tokenspeed-setup <worktree-root> \
  --custom-triton projects/triton/custom-builds/buffer-descriptor-bounds
```

The setup helper:

1. Creates or refreshes the worktree venv.
2. Installs Triton's `python/requirements.txt` into that venv.
3. Installs the custom Triton checkout with
   `pip install -e . --no-build-isolation`.
4. Writes a venv-local `tokenspeed-custom-triton.pth` that sets
   `TOKENSPEED_TRITON_PACKAGE=triton` for Python processes launched from that
   venv.
5. Writes `<venv>/tokenspeed-custom-triton.env` for inspection.

Running the helper again without `--custom-triton` removes that venv-local
selector.

## Validation

Focused import tests:

```bash
../.venv/bin/python -m pytest \
  tokenspeed-kernel/test/test_triton_import.py \
  tokenspeed-mla/test/test_triton_import.py \
  -q

TOKENSPEED_TEST_TRITON_PYTHONPATH=\
/home/mdawkins/claude-workspace/projects/triton/custom-builds/buffer-descriptor-bounds/python \
../.venv/bin/python -m pytest \
  tokenspeed-kernel/test/test_triton_import.py \
  tokenspeed-mla/test/test_triton_import.py \
  -q
```

Results:

- Default selector: `2 passed, 2 skipped`.
- Custom fork selector: `4 passed`.

Setup-helper smoke:

```bash
tokenspeed-setup \
  /home/mdawkins/claude-workspace/projects/tokenspeed/agents-rmsnorm-gluon-kernel \
  --venv /home/mdawkins/claude-workspace/projects/tokenspeed/agents-rmsnorm-gluon-kernel/.venv-custom-triton-smoke \
  --no-deps \
  --custom-triton /home/mdawkins/claude-workspace/projects/triton/custom-builds/buffer-descriptor-bounds
```

Result:

- Custom Triton editable install succeeded.
- Plain Python from the smoke venv imported the custom fork from the checkout.
- The venv-local selector set `TOKENSPEED_TRITON_PACKAGE=triton`.
- Loading `tokenspeed_kernel._triton` by file selected the custom fork and
  exposed `gluon.aggregate`.

A broader smoke using `--with-kernel --no-deps` reached the custom Triton install
successfully, then failed during TokenSpeed kernel's native-build dependency
step because that build path tried to download a 6.2 GB ROCm torch wheel into
the throwaway venv and hit `ENOSPC`. That failure was outside the custom Triton
setup path.
