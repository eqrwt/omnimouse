# `neuron_order_override` — what it adds, and the IBL results it made possible

## What this branch adds

OmniMouse normally picks at random, on every forward pass, which neurons the model sees and
which it must predict. `neuron_order_override` lets you name that split instead:

```python
order = torch.cat([visible_ids, other_ids, target_ids])   # a permutation of the session's neurons
out = model(..., neuron_order_override=order)              # or put it in the batch as "neuron_order"
```

The first `n_visible_neurons` entries are shown to the model and the last
`max_n_reconstructed_neurons` are predicted; the counts still come from the masking strategy.
The order is checked to be an integer permutation on every call, because a malformed order
does not crash, it silently scores the wrong neurons. With no override, behaviour is unchanged.

This makes evaluations like "predict region A's neurons from region B only", or "score every
neuron exactly once, with its own past hidden", possible on a trained model without retraining.
The IBL code that uses it is in
[sensorium-competition/ibl_to_experanto](https://github.com/sensorium-competition/ibl_to_experanto).

## Main results (IBL brain-wide map, 110 recordings, 149,344 neurons, 12 labs)

OmniMouse adapted to Neuropixels spikes (no video; pupil, wheel, whisker motion and task
events as inputs), compared with ridge regression given the same inputs.
Score: correlation between predicted and real spike counts per neuron on a held-out test period.

| model | test score | vs ridge | recordings where better |
| --- | :-: | :-: | :-: |
| one shared model, 110 mice (pre-registered, 50 epochs) | 0.158 | +0.018 [0.013, 0.024] | 91 / 110 |
| same, with whisker motion (pre-registered) | 0.171 | +0.019 [0.013, 0.025] | 86 / 98 |
| 110 mice after the final learning-rate decay | – | +0.025 [0.019, 0.030] | 99 / 110 |

**What the override showed — where the gain comes from.** Hiding one input at a time splits
each neuron's prediction into own past, other neurons, movement and task. On 20 recordings
scored by every model version, the 110-mouse model's whole advantage is in reading the
**other recorded neurons**: that term goes from 0.011–0.013 to 0.031, larger in 20/20
recordings, while movement, task and own past do not change.

## Limits (from the November audit)

- Much of the score is behaviour: with no neural input at all the decayed model already
  reaches 0.130 of its 0.164 (110 recordings); neural input adds +0.034.
- From neurons alone the model ties the best simple baseline (ridge or "repeat the neuron's
  recent rate", whichever is better per recording).
- The model ignores a neuron's own past: on synthetic data with planted own-past neurons it
  scores 0.001 where ridge scores 0.33.
- The 1-mouse → 20-mouse → 110-mouse comparison also changed the training recipe, so it is not
  by itself proof that more mice help.
- Per-region shares of movement and of other-neuron input differ between mice; only overall
  predictability and the task share replicate across mice.
