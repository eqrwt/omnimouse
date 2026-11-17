"""Hiera MAE pretraining on OmniMouse session stimulus videos.

Builds the SAME Hiera as the downstream 5M OmniMouse (OmniMouseArgs defaults) so the
pretrained encoder loads straight back into model.hiera.backbone.

Pretext task: hide `mask_ratio` of the movie's space-time mask-units, encode the visible
ones with Hiera, and reconstruct the HIDDEN units' pixels with a small transformer decoder.
Target = full per-patch, per-patch-normalized pixels of each mask-unit block (240 units x 576 px).

Saves best_encoder.pt (encoder state_dict only) to --out. Use it downstream with
`model.hiera_init_ckpt=<out>/best_encoder.pt`; see docs/VIDEO_ENCODER.md.

    python pretraining/mae/mae_pretrain.py                 # first 32 non-blank sessions
    CORPUS=A python pretraining/mae/mae_pretrain.py        # prebuilt corpus, see build_corpus.py
"""
import os, glob, json, math, time, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset
from experanto.datasets import ChunkDataset
from omnimouse.modeling.hiera_encoder import Hiera  # same module as downstream -> identical state_dict keys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# Same default as paths.data_dir in configs/paths/default.yaml.
DATA_DIR = os.environ.get("OMNIMOUSE_DATA_DIR", os.path.join(REPO, "data"))

# Hiera config = the ACTUAL 5M downstream video encoder, from configs/model/omnimouse.yaml
# (embed_dim=128, num_heads=2, mlp_ratio=2.67, patch_padding=(2,1,3)) + the omnimouse_5M
# experiment override stages=(2,2). patch_padding=(2,1,3) gives a 30x16x32 token grid that
# mask_unit_size=(1,8,8) divides cleanly -> masking works AND pos_embed matches the downstream
# model, so the pretrained encoder loads straight into model.hiera.backbone.
HIERA_KW = dict(input_size=(60, 36, 64), in_chans=1, embed_dim=128, num_heads=2,
    stages=(2, 2), q_pool=1, q_stride=(1, 2, 2), mask_unit_size=(1, 8, 8),
    patch_kernel=(6, 7, 7), patch_stride=(2, 2, 2), patch_padding=(2, 1, 3),
    sep_pos_embed=True, drop_path_rate=0.0, mlp_ratio=2.67)


class HieraMAE(nn.Module):
    def __init__(self, hiera_kw, mask_ratio=0.75, dec_dim=256, dec_depth=4, dec_heads=8):
        super().__init__()
        self.encoder = Hiera(**hiera_kw)
        self.mask_ratio = mask_ratio
        self.grid = tuple(self.encoder.mask_spatial_shape)          # (30, 2, 4)
        self.n_units = math.prod(self.grid)                         # 240
        T, H, W = hiera_kw["input_size"]
        gt, gh, gw = self.grid
        self.pt, self.ph, self.pw = T // gt, H // gh, W // gw        # (2, 18, 16)
        self.C = hiera_kw["in_chans"]
        self.px = self.C * self.pt * self.ph * self.pw               # 576
        with torch.no_grad():                                       # dry run -> enc width + tokens/unit
            d = torch.zeros(1, self.C, T, H, W)
            m = self.encoder.get_random_mask(d, mask_ratio)
            v = self.encoder(d, mask=m)
            self.enc_dim = v.shape[-1]
            self.tpu = v.shape[1] // int(m[0].sum())
        self.enc_to_dec = nn.Linear(self.enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(self.n_units, dec_dim))
        layer = nn.TransformerEncoderLayer(dec_dim, dec_heads, dec_dim * 4,
                                           batch_first=True, activation="gelu")
        self.decoder = nn.TransformerEncoder(layer, dec_depth)
        self.pred = nn.Linear(dec_dim, self.px)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.dec_pos, std=0.02)

    def make_target(self, video):                                  # (B,C,60,36,64) -> (B,240,576) norm'd
        B = video.size(0)
        gt, gh, gw = self.grid
        x = video.reshape(B, self.C, gt, self.pt, gh, self.ph, gw, self.pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, self.n_units, self.px)
        mu = x.mean(-1, keepdim=True); sd = x.std(-1, keepdim=True)
        return (x - mu) / (sd + 1e-6)

    def forward(self, video):
        B = video.size(0)
        mask = self.encoder.get_random_mask(video, self.mask_ratio)     # (B,240) True=keep
        vis = self.encoder(video, mask=mask)
        vis = vis.reshape(B, -1, self.tpu, self.enc_dim).mean(2)         # (B, n_vis, enc_dim)
        vis = self.enc_to_dec(vis)
        full = self.mask_token.view(1, 1, -1).expand(B, self.n_units, -1).clone()
        full[mask] = vis.reshape(-1, vis.size(-1))                       # scatter visible
        full = full + self.dec_pos
        pred = self.pred(self.decoder(full))                            # (B,240,576)
        tgt = self.make_target(video)
        return F.mse_loss(pred[~mask], tgt[~mask])                      # HIDDEN units only


def _vcfg(tier="train", stride=1, blanks=True):
    return OmegaConf.create({"screen": {"keep_nans": False, "sampling_rate": 30, "chunk_size": 60,
        "valid_condition": [{"tier": tier}], "offset": 0, "sample_stride": stride, "include_blanks": blanks,
        "transforms": {"normalization": {"means": [113.0], "stds": [59.0]}}, "interpolation": {"rescale": False}}})


# ---------------------------------------------------------------- corpus mode
# The tier field cannot give composition-matched splits (tier:train is 47% natural
# video, tier:validation is 80%), so mae/build_corpus.py precomputes its own split
# and caches it. Here we just replay that index. See build_corpus.py for why.
CORPUS_SPLITS = ("train", "validation", "test")


def _expand(runs):
    """[[first, count], ...] -> flat window index list. build_corpus stores contiguous
    runs because one trial's windows are adjacent in _valid_screen_times."""
    out = []
    for first, count in runs:
        out.extend(range(first, first + count))
    return out


def corpus_loaders(corpus, batch, workers):
    """Build train/validation/test loaders from a prebuilt corpus.

    All three splits are served from ONE ChunkDataset per session (three Subsets over
    it), because constructing a ChunkDataset is the expensive part -- doing it per
    split would triple an already slow startup across ~257 sessions.
    """
    from torch.utils.data import Subset
    cfg = _vcfg("train", stride=1, blanks=False)      # eligible pool is ALWAYS tier:train
    subs = {k: [] for k in CORPUS_SPLITS}
    nwin = {k: 0 for k in CORPUS_SPLITS}
    for sess, splits in corpus["sessions"].items():
        if not any(splits.get(k) for k in CORPUS_SPLITS):
            continue
        ds = ChunkDataset(os.path.join(DATA_DIR, sess), None, None, modality_config=cfg,
                          seed=0, out_keys=["screen"], normalize_timestamps=False)
        n = len(ds)
        for split in CORPUS_SPLITS:
            runs = splits.get(split)
            if not runs:
                continue
            idx = _expand(runs)
            # the cached indices were computed against exactly this config; if the data
            # changed underneath us they would silently point at the wrong clips
            if idx[-1] >= n:
                raise RuntimeError(f"corpus index out of range for {sess}: "
                                   f"max idx {idx[-1]} >= len {n} - rebuild the corpus")
            subs[split].append(Subset(ds, idx)); nwin[split] += len(idx)

    out = {}
    for split in CORPUS_SPLITS:
        if not subs[split]:
            out[split] = (None, 0); continue
        w = workers if split == "train" else max(2, workers // 4)
        dl = DataLoader(ConcatDataset(subs[split]), batch_size=batch, shuffle=True,
                        drop_last=True, num_workers=w, pin_memory=True,
                        collate_fn=lambda b: torch.stack([x["screen"] for x in b]))
        out[split] = (dl, nwin[split])
    return out


def make_loader(sess_list, batch, workers, shuffle, tier):
    dss = [ChunkDataset(s, None, None, modality_config=_vcfg(tier), seed=0, out_keys=["screen"],
                        normalize_timestamps=False) for s in sess_list]
    dss = [d for d in dss if len(d) > 0]          # a session may have 0 chunks for a given tier
    ds = ConcatDataset(dss)
    dl = DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=True, num_workers=workers,
                    pin_memory=True, collate_fn=lambda b: torch.stack([x["screen"] for x in b]))
    return dl, len(ds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", type=int, default=int(os.environ.get("NUM_SESSIONS", 32)))
    # defaults = the recipe used for the reported encoders (150k steps, batch 64)
    p.add_argument("--steps", type=int, default=int(os.environ.get("MAX_STEPS", 150000)))
    p.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", 64)))
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 8)))
    p.add_argument("--val_sessions", type=int, default=int(os.environ.get("VAL_SESSIONS", 4)))
    p.add_argument("--val_every", type=int, default=int(os.environ.get("VAL_EVERY", 500)))
    p.add_argument("--val_batches", type=int, default=int(os.environ.get("VAL_BATCHES", 20)))
    p.add_argument("--mask_ratio", type=float, default=0.75)
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", 5e-5)))
    p.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", 0)))  # 0 = auto (10% of steps)
    p.add_argument("--out", default=os.environ.get("OUT_DIR", os.path.join(HERE, "out")))
    p.add_argument("--save_every", type=int, default=int(os.environ.get("SAVE_EVERY", 1000)))
    p.add_argument("--fresh", action="store_true", help="ignore any existing ckpt and start over")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.sessions, a.steps, a.batch, a.workers = min(a.sessions, 2), min(a.steps, 20), min(a.batch, 4), 0
        a.val_sessions, a.val_every, a.val_batches = 1, 10, 3
        a.warmup = 2
    if a.warmup == 0:
        a.warmup = max(1, a.steps // 10)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = HieraMAE(HIERA_KW, mask_ratio=a.mask_ratio).to(dev).train()
    enc_p = sum(x.numel() for x in model.encoder.parameters()) / 1e6
    print(f"[model] grid={model.grid} n_units={model.n_units} px/unit={model.px} "
          f"enc_dim={model.enc_dim} tpu={model.tpu} encoder={enc_p:.2f}M dev={dev}", flush=True)

    corpus_tag = os.environ.get("CORPUS", "")
    if corpus_tag:
        # CORPUS is a tag written by build_corpus.py ("A", "B") or a path to a corpus json
        cpath = (corpus_tag if corpus_tag.endswith(".json")
                 else os.path.join(HERE, f"corpus_{corpus_tag}.json"))
        corpus = json.load(open(cpath))
        # LEAKAGE GUARD: the Sensorium final_test scans must never be pre-trained on.
        bad = [x for x in corpus["sessions"] if "26872" in x or "27204" in x]
        if bad:
            raise SystemExit(f"FATAL: Sensorium test scans in corpus {corpus_tag}: {bad}")
        _ld = corpus_loaders(corpus, a.batch, a.workers)
        dl, nchunks = _ld["train"]
        val_dl, nval = _ld["validation"]
        test_dl, ntest = _ld["test"]
        nsess = len(corpus["sessions"])
        cov = a.steps * a.batch * 2.0 / max(nchunks / 30.0, 1e-9)
        print(f"[data] corpus {corpus_tag}: {nsess} sessions, keep={corpus['meta']['keep_stim']}, "
              f"tier={corpus['meta']['eligible_tier']}, blanks EXCLUDED", flush=True)
        for nm, n in (("train", nchunks), ("val", nval), ("test", ntest)):
            print(f"[data]   {nm:5s} {n:>11,} windows = {n//60:>8,} independent clips "
                  f"= {n/30/3600:6.1f} h", flush=True)
        print(f"[data] batch={a.batch} steps={a.steps} -> each second of unique video seen "
              f"~{cov:.0f}x", flush=True)
    else:
        all_sess = sorted(glob.glob(os.path.join(DATA_DIR, "*")))

        def not_blank(s):   # skip all-gray/blank sessions (their per-patch target normalizes to ~0)
            d = ChunkDataset(s, None, None, modality_config=_vcfg(), seed=0, out_keys=["screen"],
                             normalize_timestamps=False)
            return len(d) > 0 and float(d[0]["screen"].std()) > 0.05

        good = []
        for s_ in all_sess:
            if len(good) >= a.sessions:
                break
            if not_blank(s_):
                good.append(s_)
        train_sess = good
        test_scans = [s_ for s_ in all_sess if ("26872" in s_ or "27204" in s_)]
        dl, nchunks = make_loader(train_sess, a.batch, a.workers, True, "train")
        val_dl, nval = make_loader(train_sess, a.batch, max(2, a.workers // 4), True, "validation")
        test_dl, ntest = (make_loader(test_scans, a.batch, max(2, a.workers // 4), True, "test")
                          if test_scans else (None, 0))
        nsess = len(train_sess)
        print(f"[data] train {nsess} sess / {nchunks} chunks (tier:train) | "
              f"val {nval} chunks (tier:validation) | test {ntest} chunks "
              f"| batch={a.batch} steps={a.steps}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    def _lr(step):                                   # linear warmup -> cosine decay to 1% of peak
        if step < a.warmup:
            return (step + 1) / max(1, a.warmup)
        prog = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)

    # ---------- RESUME: always continue from the last full checkpoint if present ----------
    ckpt_path = os.path.join(a.out, "ckpt_full.pt")
    best_val, step, wandb_id = 1e9, 0, None
    if os.path.exists(ckpt_path) and not a.fresh:
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step = int(ck.get("step", 0)); best_val = float(ck.get("best_val", 1e9)); wandb_id = ck.get("wandb_id")
        print(f"[resume] continuing from {ckpt_path} at step {step}/{a.steps} (best_val={best_val:.4f})", flush=True)
    else:
        print(f"[fresh] starting new run (no ckpt at {ckpt_path})" if not a.fresh else "[fresh] --fresh: ignoring any ckpt", flush=True)

    use_wandb = (not a.smoke) and bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "omnimouse-mae"),
                         entity=os.environ.get("WANDB_ENTITY"),
                         name=os.environ.get("RUN_NAME", "hiera-mae"),
                         id=wandb_id, resume="allow" if wandb_id else None,
                         config=dict(mask_ratio=a.mask_ratio, lr=a.lr, steps=a.steps, sessions=nsess,
                                     split="tier:train/validation/test", batch=a.batch,
                                     **{f"hiera_{k}": v for k, v in HIERA_KW.items()}))
        wandb_id = run.id

    @torch.no_grad()
    def evaluate(loader):                               # mean MAE loss on a held-out loader
        if loader is None:
            return float("nan")
        model.eval(); tot, n = 0.0, 0
        for i, video in enumerate(loader):
            if i >= a.val_batches:
                break
            tot += model(video.to(dev, non_blocking=True)).item(); n += 1
        model.train()
        return tot / max(n, 1)

    def save_full():
        tmp = ckpt_path + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        step=step, best_val=best_val, hiera_kw=HIERA_KW, wandb_id=wandb_id), tmp)
        os.replace(tmp, ckpt_path)   # atomic -> a killed job never leaves a half-written ckpt
        torch.save(model.encoder.state_dict(), os.path.join(a.out, "final_encoder.pt"))

    t0 = time.time()
    while step < a.steps:
        for video in dl:
            video = video.to(dev, non_blocking=True)
            loss = model(video)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % (1 if a.smoke else 50) == 0:
                r = loss.item(); rate = (step + 1) / max(time.time() - t0, 1e-9)
                print(f"step {step:6d}/{a.steps}  loss={r:.4f}  lr={sched.get_last_lr()[0]:.2e}  {rate:.1f} it/s", flush=True)
                if use_wandb:
                    wandb.log({"mae/loss": r}, step=step)   # lr not logged as its own graph (boring monotonic decay)
            if step % a.val_every == 0:                 # held-out validation loss (like the Colab val_loss)
                vl = evaluate(val_dl)
                print(f"   [val] step {step:6d}  val_loss={vl:.4f}  (best={best_val:.4f})", flush=True)
                if use_wandb:
                    wandb.log({"mae/val_loss": vl}, step=step)
                if vl < best_val:                       # best_encoder.pt chosen by VAL loss
                    best_val = vl
                    torch.save(model.encoder.state_dict(), os.path.join(a.out, "best_encoder.pt"))
            step += 1
            if step % a.save_every == 0:
                save_full()
            if step >= a.steps:
                break
    vl = evaluate(val_dl)                                     # final val + best check
    if use_wandb:
        wandb.log({"mae/val_loss": vl}, step=step)
    if vl < best_val:
        best_val = vl
        torch.save(model.encoder.state_dict(), os.path.join(a.out, "best_encoder.pt"))
    tl = evaluate(test_dl)
    if use_wandb and tl == tl:
        wandb.log({"mae/test_loss": tl}, step=step)
    save_full()
    print(f"[done] step={step}/{a.steps}  best_val={best_val:.4f}  final_val={vl:.4f}  test_loss={tl:.4f}  encoder -> {a.out}/best_encoder.pt", flush=True)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
