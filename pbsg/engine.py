# -*- coding: utf-8 -*-
"""Training / evaluation loop, masked metrics, and the timing hooks required by section 7.3."""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .builders import GraphBuilder, spmm
from .model import GWNBackbone


# ----------------------------------------------------------------- data
class WindowDS(Dataset):
    def __init__(self, xn, mask, starts, tin, tout):
        self.xn = xn
        self.mask = mask
        self.starts = np.asarray(starts, dtype=np.int64)
        self.tin = tin
        self.tout = tout

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        x = self.xn[s:s + self.tin]
        y = self.xn[s + self.tin:s + self.tin + self.tout]
        m = self.mask[s + self.tin:s + self.tin + self.tout]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


def load_arrays(processed_dir):
    x = np.load(processed_dir / 'x.npy')
    mask = np.load(processed_dir / 'mask.npy').astype(np.float32)
    sc = np.load(processed_dir / 'scaler.npz')
    road = np.load(processed_dir / 'road.npz')
    with open(processed_dir / 'meta.json') as f:
        meta = json.load(f)
    return x, mask, sc['mean'], sc['std'], road, meta


# ----------------------------------------------------------------- metrics
def masked_metrics(pred, true, mask, horizon_marks=(2, 5, 11)):
    """pred/true/mask shape (B, Tout, N) in ORIGINAL flow units (section 5.5 item 3)."""
    m = mask.bool()
    if m.sum() == 0:
        return {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')}
    d = (pred - true)[m]
    mae = d.abs().mean().item()
    rmse = (d ** 2).mean().sqrt().item()
    den = true[m].abs()
    good = den >= 1.0                       # MAPE denominator threshold (reported)
    if good.sum() == 0:
        mape = float('nan')
    else:
        mape = ((d[good].abs() / den[good]) * 100.0).mean().item()
    out = {'mae': mae, 'rmse': rmse, 'mape': mape}
    for h in horizon_marks:
        if h < pred.shape[1]:
            mm = mask[:, h, :].bool()
            if mm.sum():
                out['mae_h%d' % (h + 1)] = (pred[:, h, :] - true[:, h, :])[mm].abs().mean().item()
    return out


# ----------------------------------------------------------------- build helper
def make_builder(kind, meta, road, cand_path=None, K=32, M=64, m=4, d=16,
                 seed=2026, temp=1.0, k_topk=None):
    from .graphop import csr_from_edges
    N = meta['N']
    indptr, indices = csr_from_edges(road['src'], road['dst'], N)
    C = F = None
    if kind in ('pbsg', 'fixed_cand', 'rand_bud'):
        if cand_path is None:
            raise ValueError('%s needs a cached candidate file' % kind)
        z = np.load(cand_path)
        C, F = z['C'], z['F']
    if k_topk is None:
        k_topk = int(K) + int(m)        # section 6.2 default: K=32, m=4 -> k=36
    return GraphBuilder(kind, N=N, K=K, M=M, m=m, d=d,
                        road_indptr=indptr, road_indices=indices,
                        cand_C=C, cand_F=F, seed=seed, k_topk=k_topk, temp=temp)


# ----------------------------------------------------------------- train
def profile_inference(model, builder, x_one, dev, warmup=10, reps=100):
    """Inference latency at batch = 1 plus graph-builder peak memory (section 7.3/7.5)."""
    out = {'latency_p50_ms': None, 'latency_p95_ms': None,
           'graph_peak_mb': None, 'infer_peak_mb': None}
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            A = builder()
            model(x_one, A)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            A = builder()
            model(x_one, A)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.asarray(ts)
        out['latency_p50_ms'] = float(np.percentile(ts, 50))
        out['latency_p95_ms'] = float(np.percentile(ts, 95))
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            builder()
            torch.cuda.synchronize()
            out['graph_peak_mb'] = float(torch.cuda.max_memory_allocated() / 2 ** 20)
            torch.cuda.reset_peak_memory_stats()
            A = builder()
            model(x_one, A)
            torch.cuda.synchronize()
            out['infer_peak_mb'] = float(torch.cuda.max_memory_allocated() / 2 ** 20)
    return out


def run(cfg, meta, cand_path, log=print):
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    d = cfg['processed_dir']
    x, mask, mean, std, road, meta = load_arrays(d)
    xn = ((x - mean) / std).astype(np.float32)
    tin, tout = meta['tin'], meta['tout']
    sp = {k: np.asarray(v, dtype=np.int64) for k, v in
          json.load(open(d / 'meta.json'))['window_arrays'].items()}
    tr = WindowDS(xn, mask, sp['train'], tin, tout)
    va = WindowDS(xn, mask, sp['val'], tin, tout)
    te = WindowDS(xn, mask, sp['test'], tin, tout)
    bs = cfg.get('batch_size', 32)
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=cfg.get('workers', 2),
                       drop_last=True, pin_memory=(dev.type == 'cuda'))
    dl_va = DataLoader(va, batch_size=bs, shuffle=False, num_workers=cfg.get('workers', 2))
    dl_te = DataLoader(te, batch_size=bs, shuffle=False, num_workers=cfg.get('workers', 2))

    builder = make_builder(cfg['kind'], meta, road, cand_path, K=cfg['K'], M=cfg['M'],
                           m=cfg['m'], d=cfg.get('d_emb', 16),
                           seed=cfg.get('cand_seed', 2026),
                           temp=cfg.get('temperature', 1.0),
                           k_topk=cfg.get('k_topk')).to(dev)
    model = GWNBackbone(tin, tout, channels=cfg.get('channels', 32),
                        dilations=tuple(cfg.get('dilations', (1, 2, 4, 4))),
                        dropout=cfg.get('dropout', 0.0)).to(dev)

    params = list(model.parameters())
    if builder.emb is not None:
        params += [builder.emb]
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.Adam(params, lr=cfg.get('lr', 1e-3),
                           weight_decay=cfg.get('weight_decay', 0.0))
    lossf = nn.L1Loss(reduction='none')

    mean_t = torch.as_tensor(mean, device=dev).view(1, 1, -1)
    std_t = torch.as_tensor(std, device=dev).view(1, 1, -1)

    def epoch_pass(loader, train=True):
        tot, cnt = 0.0, 0
        t_graph = 0.0
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            A = builder()
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t_graph += time.perf_counter() - t0
            pred = model(xb, A)
            l = lossf(pred, yb)
            l = (l * mb).sum() / mb.sum().clamp(min=1.0)
            if train:
                opt.zero_grad(set_to_none=True)
                l.backward()
                opt.step()
            tot += l.item() * xb.shape[0]
            cnt += xb.shape[0]
        return tot / max(1, cnt), t_graph

    hist = []
    best, best_state, bad = float('inf'), None, 0
    max_epochs = cfg.get('max_epochs', 30)
    patience = cfg.get('patience', 5)
    t_start = time.perf_counter()
    for ep in range(1, max_epochs + 1):
        te0 = time.perf_counter()
        model.train()
        trl, tg = epoch_pass(dl_tr, train=True)
        model.eval()
        with torch.no_grad():
            val, _ = epoch_pass(dl_va, train=False)
        dt = time.perf_counter() - te0
        hist.append({'epoch': ep, 'train_loss': trl, 'val_loss': val,
                     'epoch_sec': dt, 'graph_sec': tg,
                     'graph_edge_nnz': int(builder.last_stats.get('edges', -1))})
        log('  ep %02d  train %.4f  val %.4f  %.1fs  graph %.3fs  A_edges %d'
            % (ep, trl, val, dt, tg, builder.last_stats.get('edges', -1)))
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if builder.emb is not None:
                best_state['__emb'] = builder.emb.detach().clone()
        else:
            bad += 1
            if bad >= patience:
                log('  early stop at epoch %d (patience %d)' % (ep, patience))
                break
    wall = time.perf_counter() - t_start

    if best_state is not None:
        emb = best_state.pop('__emb', None)
        model.load_state_dict(best_state)
        if emb is not None:
            builder.emb.data.copy_(emb)

    model.eval()
    res = {}
    with torch.no_grad():
        for name, loader in (('val', dl_va), ('test', dl_te)):
            preds, trues, masks = [], [], []
            for xb, yb, mb in loader:
                xb = xb.to(dev)
                A = builder()
                p = model(xb, A)
                preds.append(p.cpu())
                trues.append(yb)
                masks.append(mb)
            p = torch.cat(preds) * std_t.cpu() + mean_t.cpu()
            y = torch.cat(trues) * std_t.cpu() + mean_t.cpu()
            m = torch.cat(masks)
            res[name] = masked_metrics(p, y, m)
            res[name]['n_windows'] = int(p.shape[0])
    res['train'] = {'final_train_loss': hist[-1]['train_loss'],
                    'final_val_loss': hist[-1]['val_loss']}

    x_one = tr[0][0].unsqueeze(0).to(dev)
    prof = profile_inference(model, builder, x_one, dev,
                             reps=cfg.get('prof_reps', 100))
    peak_train_mb = None
    if dev.type == 'cuda':
        peak_train_mb = float(torch.cuda.max_memory_allocated() / 2 ** 20)

    return {
        'config': {k: v for k, v in cfg.items() if k != 'processed_dir'},
        'n_params': int(sum(p.numel() for p in model.parameters())
                        + (builder.emb.numel() if builder.emb is not None else 0)),
        'epochs_run': len(hist),
        'best_val_loss': best,
        'wall_sec': wall,
        'sec_per_epoch': float(np.mean([h['epoch_sec'] for h in hist])),
        'graph_sec_per_epoch': float(np.mean([h['graph_sec'] for h in hist])),
        'A_edges': int(builder.last_stats.get('edges', -1)),
        'peak_train_mb': peak_train_mb,
        'profile': prof,
        'metrics': res,
        'history': hist,
    }
