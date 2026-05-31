# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Nano-vLLM is a lightweight vLLM implementation in ~1,400 lines of Python. It targets readability while supporting key optimizations: prefix caching, tensor parallelism, CUDA graphs, `torch.compile`, and chunked prefill. Currently only the Qwen3 model family is implemented.

## Install and run

```bash
pip install -e .
```

No test suite or linter config exists in this repo.

## Architecture

The entrypoint is `LLMEngine` (`nanovllm/engine/llm_engine.py`). `LLM` (`nanovllm/llm.py`) is a trivial alias for `LLMEngine`.

```
LLM / LLMEngine
├── Scheduler         — prefill/decode scheduling, chunked prefill, preemption
├── BlockManager      — PagedAttention KV-cache blocks, prefix-cache via xxhash
├── ModelRunner       — owns the model, prepares inputs, runs forward + sampling
│   ├── Qwen3ForCausalLM  — the only supported model (nanovllm/models/qwen3.py)
│   └── Sampler        — temperature-scaled softmax sampling (no greedy)
└── Sequence          — per-request state (token_ids, block_table, status)
```

### Key design choices

- **Global context**: A module-level `Context` dataclass (`nanovllm/utils/context.py`) carries per-step metadata (cu_seqlens, slot_mapping, block_tables) to layers without threading it through every function signature. Set before each model forward, reset after.
- **Tensor parallelism**: Uses `torch.multiprocessing(spawn)` + `SharedMemory` for IPC. Rank 0 writes serialized method calls to shared memory; worker ranks loop reading and executing them. TP is implemented via `ColumnParallelLinear` / `RowParallelLinear` / `MergedColumnParallelLinear` / `QKVParallelLinear` in `nanovllm/layers/linear.py`.
- **Weight loading**: Each `nn.Parameter` gets a `weight_loader` callable attached at init time. `load_model()` iterates safetensor keys and dispatches to the appropriate loader, supporting fused (packed) layers via `packed_modules_mapping` on the model.
- **CUDA graphs**: Captured at startup in `ModelRunner.capture_cudagraph()` for batch sizes [1, 2, 4, 8, 16, 32, ...] up to `max_num_seqs`. Decode uses graphs when `bs <= 512`; prefill always runs eagerly.
- **Prefix caching**: Block-level, content-hashed with xxhash. `BlockManager.can_allocate()` walks a sequence's blocks checking for hash matches before deciding how many new blocks are needed.
- **Chunked prefill**: When a new prefill doesn't fit in `max_num_batched_tokens`, it's split across steps. Only the first waiting sequence can be chunked (not subsequent ones).

### File map

| Area | Files |
|------|-------|
| Public API | `nanovllm/__init__.py`, `nanovllm/llm.py`, `nanovllm/sampling_params.py`, `nanovllm/config.py` |
| Engine | `nanovllm/engine/llm_engine.py`, `scheduler.py`, `block_manager.py`, `model_runner.py`, `sequence.py` |
| Model | `nanovllm/models/qwen3.py` |
| Layers | `nanovllm/layers/attention.py`, `linear.py`, `rotary_embedding.py`, `sampler.py`, `layernorm.py`, `activation.py`, `embed_head.py` |
| Utils | `nanovllm/utils/context.py`, `loader.py` |
