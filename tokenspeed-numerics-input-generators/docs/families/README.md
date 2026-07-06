# Operation Family Input Generators

This directory is a navigation layer for the operation-family generators in
`tokenspeed_numerics_input_generators`. The generator implementations currently
live in flat Python modules to preserve the public import surface, while these
family directories provide the README structure described by the design
philosophy.

Each README describes the operation-level semantics, the generators in that
family, and the main generated values. Detailed field-level documentation lives
on the config and values classes in code.

## Families

- [activation](activation/README.md): elementwise and gated activations,
  including fused quantized SwiGLU.
- [attention](attention/README.md): MHA, MLA, attention metadata, cache layouts,
  and related fused attention helpers.
- [communication](communication/README.md): collective-style tensor movement and
  expert-parallel routing inputs.
- [core](core/README.md): shared base interfaces, tensor generation, and custom
  dtype semantics.
- [embedding](embedding/README.md): rotary embedding inputs and references.
- [gemm](gemm/README.md): dense, scaled, and quantized GEMM inputs.
- [kvcache](kvcache/README.md): cache stores, page-table gathers, and cache-row
  transfers.
- [layernorm](layernorm/README.md): RMSNorm, QK RMSNorm, fused RoPE/gate, and
  parallel RMSNorm inputs.
- [moe](moe/README.md): routed MoE layers and expert block-alignment metadata.
- [quantization](quantization/README.md): FP8, MXFP8, MXFP4, and NVFP4
  quantization inputs.
- [sampling](sampling/README.md): argmax, scalar gather, min-p, and top-k/top-p
  sampling inputs.
- [transforms](transforms/README.md): standalone deterministic tensor
  transforms such as Walsh-Hadamard.
