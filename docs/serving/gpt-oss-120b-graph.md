# GPT-OSS 120B MXFP4/FP8 Graph

This page sketches the inference graph for
`amd/gpt-oss-120b-w-mxfp4-a-fp8` as implemented by TokenSpeed's
`GptOssForCausalLM` path.

Sources:

- Hugging Face config:
  `https://huggingface.co/amd/gpt-oss-120b-w-mxfp4-a-fp8/blob/main/config.json`
- Hugging Face model card:
  `https://huggingface.co/amd/gpt-oss-120b-w-mxfp4-a-fp8`
- TokenSpeed implementation:
  `python/tokenspeed/runtime/models/gpt_oss.py`

## Model Constants

| Field | Value |
|---|---:|
| Architecture | `GptOssForCausalLM` |
| Vocabulary | 201088 |
| Hidden size | 2880 |
| Decoder layers | 36 |
| Query heads | 64 |
| KV heads | 8 |
| Head dim | 64 |
| QKV projection width | 5120 = 4096 Q + 512 K + 512 V |
| Experts per layer | 128 |
| Experts per token | 4 |
| Expert intermediate size | 2880 |
| RMSNorm epsilon | `1e-5` |
| RoPE | YaRN, theta 150000, factor 32, original context 4096 |
| Max positions | 131072 |
| Attention pattern | layers 0,2,... use sliding attention; layers 1,3,... use full attention |
| Sliding window | config value 128; TokenSpeed uses 127 internally because its window is exclusive |
| Quantization | AMD Quark; expert weights FP4/MXFP4 group 32 with E8M0 scales; expert inputs FP8 E4M3 |
| Quantization exclusions | self-attention projections, router, and LM head |

## Causal-LM Graph

`T` is the active token count for the current prefill/decode step. Tensor and
expert parallel sharding changes local shapes, but the logical graph is:

```mermaid
flowchart TD
  ids["input_ids [T]"] --> emb["VocabParallelEmbedding<br/>201088 -> 2880"]
  emb --> blocks["36 x GptOssDecoderLayer<br/>layers 0..35"]
  blocks --> final_norm["Final RMSNorm<br/>hidden [T, 2880]"]
  final_norm --> head["ParallelLMHead / LM head<br/>2880 -> 201088"]
  head --> logits["LogitsProcessor<br/>gather/select logits for sampling"]
```

The TokenSpeed layer compiler can insert all-reduce, reduce-scatter, all-gather,
or fused reduce-plus-RMSNorm ops between these nodes depending on the configured
attention TP, dense TP, MoE TP, and expert parallel placement.

## Decoder Layer

Each layer is a pre-norm attention block followed by a pre-norm routed MoE block:

```mermaid
flowchart TD
  x0["x_l [T, 2880]"] --> n0["input RMSNorm"]
  n0 --> qkv["QKVParallelLinear + bias<br/>Q [T, 64, 64]<br/>K,V [T, 8, 64]"]
  qkv --> rope["YaRN RoPE on Q,K<br/>optional fused K/V cache write"]
  rope --> attn_choice{"layer type"}
  attn_choice -->|"even layers"| swa["Paged GQA sliding attention<br/>config window 128"]
  attn_choice -->|"odd layers"| full["Paged GQA full attention"]
  swa --> sinks["attention sinks<br/>one learned sink per Q head"]
  full --> sinks
  sinks --> out_proj["RowParallel o_proj + bias<br/>4096 -> 2880"]
  out_proj --> add_attn["residual add<br/>x_l + attn"]
  add_attn --> n1["post-attention RMSNorm"]
  n1 --> moe["GPT-OSS Sparse MoE<br/>router + top-4 experts"]
  moe --> add_moe["residual add<br/>x_l + attn + moe"]
  add_moe --> x1["x_(l+1) [T, 2880]"]
```

Attention is grouped-query attention: 64 query heads share 8 KV heads, so each KV
head serves 8 query heads. RoPE is applied over the full 64-d head dimension.
`PagedAttention` owns the paged KV cache update/read path; TokenSpeed passes the
learned attention sinks into that attention kernel.

## Routed Expert Graph

The router and experts operate on the post-attention normalized hidden states:

```mermaid
flowchart TD
  h["post-attn norm h [T, 2880]"] --> router["router / gate<br/>2880 -> 128, bias<br/>BF16 TinyGemm fast path when eligible"]
  router --> topk["top-4 by router logit<br/>softmax over selected logits"]
  topk --> dispatch["dispatch token copies to selected experts<br/>ragged metadata / gather indices"]

  dispatch --> e1["for each selected expert e"]
  e1 --> q1["FP8 quantize h<br/>using gate_up input_scale"]
  q1 --> w13["MXFP4 gate_up matmul + bias<br/>2880 -> 5760"]
  w13 --> split["interleaved split<br/>gate [2880], up [2880]"]
  split --> act["GPT-OSS SwiGLU<br/>silu(alpha * gate) * (up + 1)<br/>alpha 1.702, limit 7.0"]
  act --> q2["FP8 quantize intermediate<br/>using down input_scale"]
  q2 --> w2["MXFP4 down matmul + bias<br/>2880 -> 2880"]
  w2 --> weight["multiply by route weight"]
  weight --> combine["scatter-add top-4 expert outputs"]
  combine --> y["MoE output [T, 2880]"]
```

The AMD checkpoint stores one tensor group per expert:

```text
model.layers.<l>.mlp.experts.<e>.gate_up_proj.{weight,weight_scale,bias,input_scale}
model.layers.<l>.mlp.experts.<e>.down_proj.{weight,weight_scale,bias,input_scale}
```

TokenSpeed maps those tensors into fused MoE parameters:

| Checkpoint tensor | TokenSpeed parameter |
|---|---|
| `gate_up_proj.weight` | `w13_weight` |
| `gate_up_proj.weight_scale` | `w13_weight_scale` |
| `gate_up_proj.bias` | `w13_weight_bias` |
| `gate_up_proj.input_scale` | `w13_input_scale` / `w13_act_scale` |
| `down_proj.weight` | `w2_weight` |
| `down_proj.weight_scale` | `w2_weight_scale` |
| `down_proj.bias` | `w2_weight_bias` |
| `down_proj.input_scale` | `w2_input_scale` / `w2_act_scale` |

For the AMD Quark W4A8 path, the Triton MXFP4 backend quantizes the expert
input and intermediate activations to FP8 before the two expert GEMMs. The
logical top-4 combine is still weighted by the router softmax values.
