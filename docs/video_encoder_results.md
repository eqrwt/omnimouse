# Video encoder: pretraining, freezing and fine-tuning — results

What this branch adds, and what came out of it. All runs: OmniMouse-5M on the 8 mice of
`scaling_laws_8` (64,074 neurons), 3 seeds per setting unless marked.

## What was tried

| setting | how to run it |
| --- | --- |
| random encoder, trained with the model (baseline) | `experiment=omnimouse_5M` |
| MAE-pretrained encoder, trained with the model | `+model.hiera_init_ckpt=<best_encoder.pt>` |
| MAE-pretrained encoder, frozen | `+model.hiera_init_ckpt=... model.freeze_hiera=true` |
| encoder at a 10× lower learning rate ("disc-LR") | `trainer.encoder_lr=7e-5` (rest of the model stays at 7e-4) |
| MAE encoder + disc-LR | `experiment=omnimouse_5M_mae_disclr model.hiera_init_ckpt=<best_encoder.pt>` |
| ConvNeXtV2 instead of Hiera | `+model.video_backbone=convnextv2` |

MAE pretraining (`pretraining/mae/`): the encoder learns to fill in 75% hidden patches of the
stimulus movies, 150k steps. Two corpora: 32 sessions, and corpus A = 257 sessions
(288,462 independent 2-s clips, 160 h; natural movies 63%, Monet2 20%, Trippy 17%).

## Best result

**Train the video encoder at a 10× lower learning rate than the rest of the model.**
That is where almost all of the gain comes from. Starting it from an MAE-pretrained encoder
adds a little more, and in our seeds the runs were more repeatable (SD 0.002 vs 0.007), but
on its own pretraining is a small effect.

| model (Hiera encoder) | validation, step 600k | Sensorium 2022 Main | Sensorium 2022 Bonus |
| --- | :-: | :-: | :-: |
| random, normal LR (baseline) | 0.241 ± 0.007 | 0.167 ± 0.010 | 0.208 ± 0.010 |
| MAE (32 sessions), normal LR | 0.250 ± 0.008 | 0.177 ± 0.009 | 0.219 ± 0.011 |
| random, disc-LR | 0.267 ± 0.007 | 0.181 (1 seed) | 0.223 (1 seed) |
| **MAE (32 sessions), disc-LR** | **0.270 ± 0.002** | 0.196 ± 0.005 | 0.243 ± 0.003 |
| **MAE (corpus A), disc-LR** | 0.269 ± 0.001 | **0.202 ± 0.001** | **0.248 ± 0.002** |
| ConvNeXtV2-128, normal LR | – | 0.185 ± 0.005 | 0.225 ± 0.007 |
| ConvNeXtV2-192, normal LR | – | 0.188 ± 0.006 | 0.229 ± 0.009 |

Validation = mean single-trial correlation over the 7 neural-prediction tasks, mean ± SD over
3 seeds, all at the same step. Sensorium 2022 = the competition's test scans
(`eval_experiment=sens2022`); published reference: winner 0.33 / 0.45, OmniMouse-80M trained
on 323 sessions 0.37 / 0.45.

How the +0.029 gain over the baseline splits (validation, 600k):
- lower encoder learning rate alone: **+0.026** (89%)
- MAE pretraining on top of it: **+0.003** (11%), within seed noise
- MAE pretraining without the lower LR: +0.009 (about 1.4 standard errors)

## Other results, one line each

- **Frozen encoders** (100k steps): a frozen MAE encoder beats a frozen random one by
  +0.023 (3/3 seeds), so the pretrained features are real. A frozen ConvNeXtV2 is no better
  than random (+0.001). Freezing costs a random encoder 0.024 but an MAE encoder only 0.002.
- **What to pretrain on**: corpus A (natural movies + parametric noise) beats corpus B
  (natural movies only) when frozen, +4.9%, 3/3 seeds.
- **Bigger model**: widening OmniMouse to d_model 512 / 8 heads (same 4-block encoder) with
  MAE + disc-LR adds +0.007 at 500k steps (0.272 vs 0.265), all 3 seeds above every 5M seed.
  This is not the paper's OmniMouse-20M, which also has a deeper encoder.
- **Linear-probe-then-fine-tune** does not help: once unfrozen at the normal LR it falls back
  onto the random-init curve.
- **Unseen stimulus types** (Gabor, RDK, dots, Monet2; `ood_result2` + `ood_bonus`): every
  setting drops by about the same 35–37%. Frozen vs fine-tuned makes no difference.
- **A mouse the model has never seen** (`scripts/make_clip_configs.py`): fitting only its
  neurons' parameters goes from 0.041 with 2 clips to 0.192 with 256; 32 clips already give
  81% of that.

## Caveats

- Sensorium for random + disc-LR is one seed (at 500k), and it is the weakest of the three
  validation seeds. Corpus A was scored at its best checkpoint (820k), the other Hiera rows at
  ~600k, ConvNeXtV2 at 875k–1M.
- 3 seeds per setting: differences below about 0.005 on validation are within seed noise.
- No run reached the planned 1M steps; all stopped at the 36-hour job limit.
