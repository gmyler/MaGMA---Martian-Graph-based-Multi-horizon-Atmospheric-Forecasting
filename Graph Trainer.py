
import os, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import subgraph
from tqdm import tqdm
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


data.edge_index = data.edge_index.long().contiguous().cpu()
data.edge_type  = data.edge_type.long().contiguous().cpu()
data.x          = data.x.float().contiguous().cpu()
data.x_vert     = data.x_vert.float().contiguous().cpu()
data.y          = data.y.float().contiguous().cpu()
data.storm_mask = data.storm_mask.bool().contiguous().cpu()
data.cas_target = data.cas_target.float().contiguous().cpu()

ei = data.edge_index
et = data.edge_type
N  = data.num_nodes

ei_min = int(ei.min().item())
ei_max = int(ei.max().item())
et_min = int(et.min().item())
et_max = int(et.max().item())

assert ei_min >= 0, f"edge_index has negative id: {ei_min}"
assert ei_max < N, f"edge_index out of range: max={ei_max}, N={N}"
assert et_min >= 0, f"edge_type has negative type: {et_min}"

num_rel = et_max + 1
print("CPU graph OK. num_rel =", num_rel)


T_CHUNK = int(data.T_CHUNK)
SEQ_DIM = int(data.seq_dim)


if hasattr(data, "N_VARS"):
    N_VARS = int(data.N_VARS)
else:
    assert SEQ_DIM % T_CHUNK == 0, "seq_dim must be n_vars * T_CHUNK"
    N_VARS = SEQ_DIM // T_CHUNK

print(f" Temporal config: N_VARS={N_VARS}, T_CHUNK={T_CHUNK}, SEQ_DIM={SEQ_DIM}")



def safe(x, limit=80.0):
    x = torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit)
    return torch.clamp(x, -limit, limit)

def finite_rows(yhat, ytrue):
    return torch.isfinite(yhat).all(dim=1) & torch.isfinite(ytrue).all(dim=1)

def inject_tf_noise(x_b, seq_dim, n_vars, t_chunk, noise=0.002):
    if noise <= 0:
        return x_b
    idxs = [i*t_chunk + (t_chunk-1) for i in range(n_vars)]
    idxs = torch.tensor(idxs, device=x_b.device)
    x_b = x_b.clone()
    x_b[:, idxs] += noise * torch.randn((x_b.size(0), len(idxs)), device=x_b.device)
    return x_b

def oversample_storm(nodes_cpu, storm_mask_cpu, storm_ratio=0.25):
    nodes_cpu = np.asarray(nodes_cpu)
    storms = nodes_cpu[storm_mask_cpu[nodes_cpu]]
    calm   = nodes_cpu[~storm_mask_cpu[nodes_cpu]]

    k = len(nodes_cpu)
    if k == 0:
        return nodes_cpu

    desired_storm = int(storm_ratio * k)
    desired_calm  = k - desired_storm

    if len(storms) == 0:
        storms_s = np.array([], dtype=np.int64)
        desired_calm = k
    elif len(storms) >= desired_storm:
        storms_s = np.random.choice(storms, desired_storm, replace=False)
    else:
        storms_s = np.random.choice(storms, desired_storm, replace=True)

    if len(calm) >= desired_calm:
        calm_s = np.random.choice(calm, desired_calm, replace=False)
    else:
        calm_s = np.random.choice(calm, desired_calm, replace=True)

    out = np.concatenate([storms_s, calm_s])
    np.random.shuffle(out)
    return out


class TemporalEncoder(nn.Module):
    def __init__(self, n_vars, t_chunk, emb=64):
        super().__init__()
        self.n_vars=n_vars
        self.t_chunk=t_chunk
        self.net = nn.Sequential(
            nn.Conv1d(n_vars, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, emb, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x_seq_flat):
        Nn = x_seq_flat.shape[0]
        xs = x_seq_flat.view(Nn, self.n_vars, self.t_chunk)
        return self.net(xs).squeeze(-1)


class VerticalGRUBlock(nn.Module):
    def __init__(self, v_in, v_hid=128):
        super().__init__()
        self.in_proj = nn.Linear(v_in, v_hid)
        self.gru = nn.GRU(v_hid, v_hid, batch_first=True)
    def forward(self, x_vert):
        x = safe(self.in_proj(x_vert))
        h, _ = self.gru(x)
        return safe(h)

class VerticalPhysicsMLP(nn.Module):
    def __init__(self, v_hid=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(v_hid, v_hid),
            nn.GELU(),
            nn.Linear(v_hid, v_hid)
        )
        self.norm = nn.LayerNorm(v_hid)
    def forward(self, h_vert):
        h0 = h_vert
        h  = safe(self.mlp(h_vert))
        return safe(self.norm(h + h0))

class VerticalSelfAttention(nn.Module):
    def __init__(self, v_hid=128, n_heads=4, dropout=0.05):
        super().__init__()
        self.attn = nn.MultiheadAttention(v_hid, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(v_hid)
    def forward(self, h_vert):
        h0 = h_vert
        h_attn, _ = self.attn(h_vert, h_vert, h_vert, need_weights=False)
        return safe(self.norm(h0 + safe(h_attn)))

class VerticalProjector(nn.Module):
    def __init__(self, v_hid=128, hid=256):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(v_hid, hid),
            nn.GELU(),
            nn.LayerNorm(hid)
        )
    def forward(self, h_vert):
        return safe(self.out(h_vert.mean(dim=1)))



class V4_5_INTEGRATED_V5READY(nn.Module):
    def __init__(self, in_dim, seq_dim, v_in, hid=256, num_rel=4,
                 n_vars=6, t_chunk=8, t_emb=64, dropout=0.05,
                 vert_batch=2048, out_dim=7):
        super().__init__()


        self.raw_in_dim = in_dim
        self.seq_dim = seq_dim
        self.n_vars = n_vars          
        self.t_chunk = t_chunk
        self.out_dim = out_dim       

        self.hid = hid
        self.dropout = dropout
        self.num_rel = num_rel
        self.vert_batch = vert_batch

        self.extra_state_dim = 2 * n_vars


        self.temp_enc = TemporalEncoder(
            n_vars=n_vars,
            t_chunk=t_chunk,
            emb=t_emb
        )


        static_dim = (in_dim - seq_dim) + self.extra_state_dim

        self.temp_proj = nn.Linear(t_emb, hid)

        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, hid),
            nn.GELU(),
            nn.LayerNorm(hid)
        )

        self.vert_gru = VerticalGRUBlock(v_in=v_in, v_hid=128)
        self.vert_phys = VerticalPhysicsMLP(v_hid=128)
        self.vert_attn = VerticalSelfAttention(
            v_hid=128,
            n_heads=4,
            dropout=dropout
        )
        self.vert_proj = VerticalProjector(v_hid=128, hid=hid)


        self.vert_gamma = nn.Sequential(
            nn.Linear(hid, hid // 4),
            nn.GELU(),
            nn.Linear(hid // 4, 1)
        )


        self.edge_emb = nn.Embedding(num_rel, hid)


        self.cas_head = nn.Sequential(
            nn.Linear(hid, hid // 2),
            nn.GELU(),
            nn.Linear(hid // 2, 1)
        )


        self.cas_scale = nn.Parameter(torch.tensor(1.0))

        self.router = nn.Sequential(
            nn.Linear(hid + 1, hid // 2),
            nn.GELU(),
            nn.Linear(hid // 2, 3)
        )


        self.exp0_convs = nn.ModuleList([
            RGCNConv(hid, hid, num_relations=num_rel, num_bases=2)
        ])
        self.exp0_norms = nn.ModuleList([
            nn.LayerNorm(hid)
        ])

        self.exp1_convs = nn.ModuleList([
            RGCNConv(hid, hid, num_relations=num_rel, num_bases=2),
            RGCNConv(hid, hid, num_relations=num_rel, num_bases=2)
        ])
        self.exp1_norms = nn.ModuleList([
            nn.LayerNorm(hid),
            nn.LayerNorm(hid)
        ])

        self.exp2_convs = nn.ModuleList([
            RGCNConv(hid, hid, num_relations=num_rel, num_bases=2),
            RGCNConv(hid, hid, num_relations=num_rel, num_bases=2),
            RGCNConv(hid, hid, num_relations=num_rel, num_bases=2),
            RGCNConv(hid, hid, num_relations=num_rel, num_bases=2)
        ])
        self.exp2_norms = nn.ModuleList([
            nn.LayerNorm(hid) for _ in range(4)
        ])

        self.out1 = nn.Linear(hid, out_dim)
        self.out3 = nn.Linear(hid, out_dim)
        self.out6 = nn.Linear(hid, out_dim)


    def extract_last_and_trend(self, x):
        x_seq = x[:, :self.seq_dim]

        last_idxs = [
            i * self.t_chunk + (self.t_chunk - 1)
            for i in range(self.n_vars)
        ]

        prev_idxs = [
            i * self.t_chunk + (self.t_chunk - 2)
            for i in range(self.n_vars)
        ]

        last_idxs = torch.tensor(last_idxs, device=x.device)
        prev_idxs = torch.tensor(prev_idxs, device=x.device)

        last_step = x_seq[:, last_idxs]
        prev_step = x_seq[:, prev_idxs]

        trend = last_step - prev_step

        return safe(last_step), safe(trend)

    def make_persistence_baseline(self, last_step):

        if last_step.shape[1] == self.out_dim:
            return last_step

        if last_step.shape[1] < self.out_dim:
            pad = torch.zeros(
                last_step.shape[0],
                self.out_dim - last_step.shape[1],
                device=last_step.device,
                dtype=last_step.dtype
            )
            return torch.cat([last_step, pad], dim=1)

        return last_step[:, :self.out_dim]

    def run_expert(self, h, convs, norms, edge_index, edge_type):
        src, dst = edge_index

        for conv, ln in zip(convs, norms):
            h0 = h
            h = safe(conv(h, edge_index, edge_type))

            rel_e = self.edge_emb(edge_type)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, rel_e)

            deg = torch.bincount(
                dst,
                minlength=h.shape[0]
            ).clamp_min(1).unsqueeze(1)

            agg = agg / deg

            h = h + agg
            h = F.gelu(h)
            h = ln(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h0

        return safe(h)

    def forward_vertical_batched(self, x_vert):
        Nn = x_vert.shape[0]
        bs = self.vert_batch
        outs = []

        for s in range(0, Nn, bs):
            xb = x_vert[s:s + bs]
            hv = self.vert_gru(xb)
            hv = self.vert_phys(hv)
            hv = self.vert_attn(hv)
            h_col = self.vert_proj(hv)
            outs.append(h_col)

        return torch.cat(outs, dim=0)

    def forward(self, x, x_vert, edge_index, edge_type, router_temp=1.0):
        h_col = self.forward_vertical_batched(x_vert)


        gamma = 0.5 + 1.5 * torch.sigmoid(self.vert_gamma(h_col))

        last_step, trend = self.extract_last_and_trend(x)
        baseline = self.make_persistence_baseline(last_step)

        x_seq = x[:, :self.seq_dim]
        x_sta = x[:, self.seq_dim:]


        x_sta_aug = torch.cat([x_sta, last_step, trend], dim=1)

        h_ts = self.temp_proj(self.temp_enc(x_seq))
        h_st = self.static_proj(x_sta_aug)

        h_base = safe(gamma * h_col + h_ts + h_st)

        cas_logit = self.cas_head(h_base)
        cas = self.cas_scale * torch.sigmoid(cas_logit)

        logits = self.router(torch.cat([h_base, cas], dim=1))
        gates = F.softmax(logits / router_temp, dim=1)

        h0 = self.run_expert(
            h_base,
            self.exp0_convs,
            self.exp0_norms,
            edge_index,
            edge_type
        )

        h1 = self.run_expert(
            h_base,
            self.exp1_convs,
            self.exp1_norms,
            edge_index,
            edge_type
        )

        h2 = self.run_expert(
            h_base,
            self.exp2_convs,
            self.exp2_norms,
            edge_index,
            edge_type
        )

        h_mix = gates[:, 0:1] * h0 + gates[:, 1:2] * h1 + gates[:, 2:3] * h2

        res1 = safe(self.out1(h_mix))
        res3 = safe(self.out3(h_mix))
        res6 = safe(self.out6(h_mix))

        y1_hat = safe(baseline + res1)
        y3_hat = safe(baseline + res3)
        y6_hat = safe(baseline + res6)

        return (
            y1_hat,
            y3_hat,
            y6_hat,
            cas_logit,
            gates
        )

def coupled_corr_loss(yhat, ytrue, couple_subsample=8192, eps=1e-6):
    mask = finite_rows(yhat, ytrue)
    yh = yhat[mask]; yt = ytrue[mask]
    if yh.shape[0] < 64:
        return torch.zeros((), device=yhat.device)
    if yh.shape[0] > couple_subsample:
        idx = torch.randperm(yh.shape[0], device=yhat.device)[:couple_subsample]
        yh = yh[idx]; yt = yt[idx]
    yh = safe(yh); yt = safe(yt)
    yh_c = yh - yh.mean(0, keepdim=True)
    yt_c = yt - yt.mean(0, keepdim=True)
    yh_z = yh_c / yh_c.std(0, keepdim=True).clamp_min(eps)
    yt_z = yt_c / yt_c.std(0, keepdim=True).clamp_min(eps)
    n = yh_z.shape[0]
    corr_hat = (yh_z.T @ yh_z) / (n-1+eps)
    corr_tru = (yt_z.T @ yt_z) / (n-1+eps)
    return F.mse_loss(corr_hat, corr_tru)


def head_loss(yhat, ytrue, storm_mask=None, w=None):
    mask = finite_rows(yhat, ytrue)
    if mask.sum() < 32:
        return torch.zeros((), device=yhat.device)
    yh = yhat[mask]; yt = ytrue[mask]
    mse = (yh-yt)**2
    if w is not None:
        mse = mse * w.view(1,-1)
    if storm_mask is not None:
        storm_w = torch.where(storm_mask[mask], 2.0, 1.0).view(-1,1)
        mse = mse * storm_w
    return mse.mean()


def split_heads(Y):
    return Y[:,0:7], Y[:,7:14], Y[:,14:21]


def latent_smoothness_loss(z):
    """
    Penalises sharp changes in latent space.
    z: Tensor [N, D] or [T, N, D]
    """
    if z is None:
        return torch.zeros((), device=z.device)
    if z.ndim == 2:
        if z.shape[0] < 2:
            return torch.zeros((), device=z.device)
        dz = z[1:] - z[:-1]
    else:
        if z.shape[1] < 2:
            return torch.zeros((), device=z.device)
        dz = z[:,1:] - z[:,:-1]
    return (dz ** 2).mean()

def output_smoothness_loss(yhat):
    """
    Penalises unphysical spikes in predictions.
    """
    if yhat.shape[0] < 2:
        return torch.zeros((), device=yhat.device)
    dy = yhat[1:] - yhat[:-1]
    return (dy ** 2).mean()


def latent_norm_loss(z, target=1.0):
    """
    Keeps latent magnitude bounded.
    """
    if z is None:
        return torch.zeros((), device=z.device)
    return ((z.norm(dim=-1) - target) ** 2).mean()


t_ids_cpu = np.array([k[0] for k in ids], dtype=np.int64)
t_unique  = np.unique(t_ids_cpu)
n_train_t = int(0.8 * len(t_unique))
train_t   = t_unique[:n_train_t]
val_t     = t_unique[n_train_t:]

train_mask_cpu = np.isin(t_ids_cpu, train_t)
val_mask_cpu   = np.isin(t_ids_cpu, val_t)

train_nodes_cpu = np.where(train_mask_cpu)[0]
val_nodes_cpu   = np.where(val_mask_cpu)[0]

print("Train nodes:", len(train_nodes_cpu), "Val nodes:", len(val_nodes_cpu))

def make_time_batches(nodes_cpu, t_ids_cpu, batch_t=8):
    t_list = np.unique(t_ids_cpu[nodes_cpu])
    for i in range(0, len(t_list), batch_t):
        ts = t_list[i:i+batch_t]
        idx = nodes_cpu[np.isin(t_ids_cpu[nodes_cpu], ts)]
        if len(idx) > 0:
            yield idx


vert_batch = 2048
hid = 256

model = V4_5_INTEGRATED_V5READY(
    in_dim=data.x.shape[1],
    seq_dim=SEQ_DIM,
    v_in=data.x_vert.shape[-1],
    hid=hid,
    num_rel=num_rel,
    n_vars=N_VARS,
    t_chunk=T_CHUNK,
    t_emb=64,
    dropout=0.05,
    vert_batch=vert_batch,
    out_dim=7
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

use_amp = (device.type == "cuda")
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)


w = torch.tensor([1, 1, 4, 1, 1, 1, 5], dtype=torch.float32, device=device)

lam_couple = 0.03
lam_cas    = 0.15
lam_gate   = 1e-3

EPOCHS = 150
best_val=float("inf")
best_state=None
patience=25
bad=0

batch_t = 8
grad_accum = 2

epoch_bar = tqdm(range(EPOCHS), desc=f"Training SUBGRAPH (batch_t={batch_t}, accum={grad_accum})")

for ep in epoch_bar:
    router_temp = max(0.7, 1.5 - ep * 0.01)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    step = 0

    for nodes_cpu in make_time_batches(train_nodes_cpu, t_ids_cpu, batch_t=batch_t):
        nodes_cpu = oversample_storm(nodes_cpu, data.storm_mask.numpy(), storm_ratio=0.25)
        nodes_cpu = torch.tensor(nodes_cpu, dtype=torch.long)

        sub_ei, sub_et = subgraph(
            nodes_cpu, data.edge_index, data.edge_type,
            relabel_nodes=True, num_nodes=N
        )

        x_b      = data.x[nodes_cpu].to(device, non_blocking=True)
        xv_b     = data.x_vert[nodes_cpu].to(device, non_blocking=True)
        y_b      = data.y[nodes_cpu].to(device, non_blocking=True)
        storm_b  = data.storm_mask[nodes_cpu].to(device, non_blocking=True)
        cas_b    = data.cas_target[nodes_cpu].to(device, non_blocking=True)

        sub_ei   = sub_ei.to(device, non_blocking=True)
        sub_et   = sub_et.to(device, non_blocking=True)

        x_b = inject_tf_noise(
            x_b,
            seq_dim=SEQ_DIM,
            n_vars=N_VARS,
            t_chunk=T_CHUNK,
            noise=0.002
        )

        with torch.cuda.amp.autocast(enabled=use_amp):
            y1_hat,y3_hat,y6_hat, cas_logit, gates = model(
                x_b, xv_b, sub_ei, sub_et, router_temp=router_temp
            )
            y1_t,y3_t,y6_t = split_heads(y_b)

            L1 = head_loss(y1_hat, y1_t, storm_b, w)
            L3 = head_loss(y3_hat, y3_t, storm_b, w)
            L6 = head_loss(y6_hat, y6_t, storm_b, w)
            C3 = coupled_corr_loss(y3_hat, y3_t)

            cas_true = cas_b.view(-1,1)
            cas_loss = F.binary_cross_entropy_with_logits(cas_logit, cas_true)

            gate_entropy = -(gates * torch.log(gates + 1e-8)).sum(1).mean()

            loss = (
                L1 + L3 + L6
                + lam_couple*C3
                + lam_cas*cas_loss
                - lam_gate*gate_entropy
                + 0.01 * (
                    output_smoothness_loss(y1_hat)
                    + output_smoothness_loss(y3_hat)
                    + output_smoothness_loss(y6_hat)
                )
              )

            loss = loss / grad_accum

        scaler.scale(loss).backward()
        running_loss += loss.item() * grad_accum
        step += 1

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        del x_b, xv_b, y_b, storm_b, cas_b, sub_ei, sub_et

    if step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


    model.eval()
    val_losses = []
    with torch.no_grad():
        for nodes_cpu in make_time_batches(val_nodes_cpu, t_ids_cpu, batch_t=batch_t):
            nodes_cpu = torch.tensor(nodes_cpu, dtype=torch.long)

            sub_ei, sub_et = subgraph(
                nodes_cpu, data.edge_index, data.edge_type,
                relabel_nodes=True, num_nodes=N
            )

            x_b      = data.x[nodes_cpu].to(device, non_blocking=True)
            xv_b     = data.x_vert[nodes_cpu].to(device, non_blocking=True)
            y_b      = data.y[nodes_cpu].to(device, non_blocking=True)
            storm_b  = data.storm_mask[nodes_cpu].to(device, non_blocking=True)
            cas_b    = data.cas_target[nodes_cpu].to(device, non_blocking=True)

            sub_ei   = sub_ei.to(device, non_blocking=True)
            sub_et   = sub_et.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                y1_hat,y3_hat,y6_hat, cas_logit, gates = model(
                    x_b, xv_b, sub_ei, sub_et, router_temp=1.0
                )
                y1_t,y3_t,y6_t = split_heads(y_b)

                V1 = head_loss(y1_hat, y1_t, storm_b, w)
                V3 = head_loss(y3_hat, y3_t, storm_b, w)
                V6 = head_loss(y6_hat, y6_t, storm_b, w)
                VC = coupled_corr_loss(y3_hat, y3_t)

                cas_true = cas_b.view(-1,1)
                cas_loss = F.binary_cross_entropy_with_logits(cas_logit, cas_true)

                gate_entropy = -(gates * torch.log(gates + 1e-8)).sum(1).mean()

                val_loss = (
                    V1 + V3 + V6
                    + lam_couple*VC
                    + lam_cas*cas_loss
                    - lam_gate*gate_entropy
                    + 0.01 * (
                        output_smoothness_loss(y1_hat)
                        + output_smoothness_loss(y3_hat)
                        + output_smoothness_loss(y6_hat)
                    )
                )

            val_losses.append(val_loss.item())
            del x_b, xv_b, y_b, storm_b, cas_b, sub_ei, sub_et

    mean_train = running_loss / max(1, step)
    mean_val   = float(np.mean(val_losses)) if len(val_losses) else mean_train

    epoch_bar.set_postfix({
        "train": f"{mean_train:.3f}",
        "val":   f"{mean_val:.3f}",
        "temp":  f"{router_temp:.2f}"
    })

    if mean_val < best_val - 1e-4:
        best_val = mean_val
        best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            print("Early stop.")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    print("Loaded best val model:", best_val)
