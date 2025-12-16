# AdaCodec-NVFP4

This repository contains a **toy research project** built on top of *SpinQuant* [[paper]](https://arxiv.org/pdf/2405.16406), targeting **W4A4 inference with NVFP4** for large language models.

The core motivation is that, under W4A4 settings, **activation quantization dominates the performance degradation**, especially for NVFP4 due to its global scaling and limited exponent range. This project explores a lightweight, activation-focused approach that is complementary to existing weight-only quantization methods.

---

## Method Overview

We propose **AdaCodec**, a two-stage optimization framework designed to mitigate activation quantization error while remaining compatible with standard PTQ pipelines.

### Stage 1: Adaptive Codec for Activation Quantization

Consider a linear layer:
$Y = XW$

We introduce a trainable **encoder–decoder pair** ($R_1$, $R_2$) around the activation:

$\hat{X} = Q(X R_1) R_2$,

so that the linear layer becomes:

$Y = Q(X R_1) R_2 W$.


Key design choices:

* $R_1$ and $R_2$ are initialized as Random Hadamard matrices.
* Both matrices are constrained to the **Stiefel manifold** (orthogonal matrices).
* We use **block-diagonal structures** both to reduce overhead and to better fit fine-grained quantization:
  $R_i = \mathrm{diag}(R_{i1}, R_{i2}, \ldots, R_{in})$, $\quad i \in {1,2}$,
  
  where each (R_{ij}) is a small orthogonal block.

This stage exclusively targets **activation quantization**, making AdaCodec orthogonal to weight quantization techniques.

---

### Stage 2: Decoder Fusion and Weight Quantization

We fuse the decoder into the weight matrix and apply quantization:
$
Y = Q(X R_1), Q(R_2 W).
$

At this stage:

* The problem reduces to standard **weight PTQ with error compensation**.
* For fast validation, we adopt **GPTQ**.
* Since AdaCodec focuses on activations, it can be naturally combined with other weight-only methods such as **AWQ**.

---

## Experimental Results

We evaluate on **LLaMA-2-7B** using the **WikiText-2** dataset. All experiments use **W4A4KV16** unless otherwise specified.

| Method                     |    PPL   |
| :------------------------- | :------: |
| W16A16KV16                 |   5.47   |
| RTN                        |   5.90   |
| GPTQ                       |   5.83   |
| AdaCodec (W16A4KV16 training, W4A4KV4 evaluation) |   6.15   |
| **GPTQ + AdaCodec**        | **5.42** |
| Fine-Grained RHT           |   6.96   |
| GPTQ + Fine-Grained RHT    |   5.76   |

**Observation:** AdaCodec alone improves activation robustness, and when combined with GPTQ, it consistently outperforms vanilla GPTQ under NVFP4 W4A4 inference.

---

## Usage

### Requirements

* Python 3.9
* PyTorch ≥ 2.0 (CUDA build required for `fast-hadamard-transform`)
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

### Running the Pipeline

Before running, set the following paths in the scripts:

* `output_rotation_path`
* `optimized_rotation_path`
* `output_dir`
* `logging_dir`

For gated models (e.g., Meta LLaMA), provide your HuggingFace token via `access_token`.

#### Step 1: Optimize Rotation Matrices

```bash
bash scripts/10_optimize_rotation.sh
```

#### Step 2: PTQ Evaluation

After optimization, place the learned rotations into `optimized_rotation_path`, then run:

```bash
bash scripts/2_eval_ptq.sh
```

---

## Limitations and Future Work

1. **NVFP4 scaling strategy**: We currently fix the global scaling factor to (1/6). The choice of NVFP4 parameters is critical but tricky; we plan to integrate **FourOverSix** for adaptive scaling.
2. **Stronger weight-side compensation**: Replace GPTQ with more advanced error-compensation algorithms.
3. **Broader combinations**: Explore integration with AWQ and other activation-aware PTQ methods.

---

## Notes

This project is intended as a **toy validation** of whether activation-focused codecs can generalize to NVFP4. The results suggest that carefully structured activation transforms remain effective even under strict 4-bit floating-point constraints.
