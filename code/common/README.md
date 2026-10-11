# code/common — shared library

Everything reused by more than one phase lives here, so each phase notebook stays short:
load model → attach adapter → attach head → train → predict → package.

| File | Contents | First filled in |
|---|---|---|
| `tokenizer.py` | Load the Phase-1 BPE tokenizer; special-token IDs | Phase 3 (from Phase 1 code) |
| `models.py` | Class definitions for A (decoder), B (enc-dec), C (LSTM), D (TCN) + `load_backbone(name, ckpt)` | Phase 3 (from Phase 1 code) |
| `encode.py` | `get_hidden_states(model, ids, mask)` — one interface for all four architectures | Phase 3 |
| `peft/lora.py` | LoRA, rsLoRA, AdaLoRA variants | Phase 3 (from Phase 2 S2) |
| `peft/lorma.py` | LORMA | Phase 3 (from Phase 2 S3) |
| `peft/prompt.py` | Prompt / prefix tuning | when needed |
| `peft/ia3.py` | IA³ | when needed |
| `kd.py` | Distillation losses (soft-label KL + hard CE), teacher-logit caching | Phase 3 (from Phase 2 S4) |
| `heads.py` | Pooling (mean / last / attention) + classification heads | Phase 3 |
| `train_utils.py` | Seeding, AMP + DataParallel for 2× T4, checkpoint save/resume, param counting, GPU-mem logging | Phase 3 |
| `submission.py` | Write `prediction.json`, validate IDs / formats, zip with file at root | Phase 3 |

Rule of thumb: if you copy a cell from one phase notebook into another, move it here instead.
