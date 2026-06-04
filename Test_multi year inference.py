

import torch
import numpy as np
from torch_geometric.utils import subgraph
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_t = 8

def make_time_batches(ids, batch_t=8):
    t_ids = np.array([k[0] for k in ids])
    t_unique = np.unique(t_ids)

    for i in range(0, len(t_unique), batch_t):
        ts = t_unique[i:i+batch_t]
        idx = np.where(np.isin(t_ids, ts))[0]
        if len(idx) > 0:
            yield idx

results = {}

model.eval()

for y in sorted(graphs.keys()):

    print(f"\nInference MY{y}")

    g = graphs[y]

    x  = g["x"].float().cpu()
    xv = g["x_vert"].float().cpu()
    ei = g["edge_index"].long().cpu()
    et = g["edge_type"].long().cpu()

    ids = g["ids"]
    if ids is None:
        raise RuntimeError(f"MY{y}: ids missing")

    N = x.shape[0]

    preds = []
    trues = []
    cas_preds = []
    cas_trues = []

    for nodes in make_time_batches(ids, batch_t=batch_t):

        nodes = torch.tensor(nodes, dtype=torch.long)

        sub_ei, sub_et = subgraph(
            nodes, ei, et,
            relabel_nodes=True,
            num_nodes=N
        )

        x_b  = x[nodes].to(device)
        xv_b = xv[nodes].to(device)

        sub_ei = sub_ei.to(device)
        sub_et = sub_et.to(device)

        with torch.no_grad():
            y1, y3, y6, cas_logit, _ = model(
                x_b, xv_b, sub_ei, sub_et, router_temp=1.0
            )

        y_pred = torch.cat([y1, y3, y6], dim=1)

        preds.append(y_pred.cpu())

        if g.get("y") is not None:
            trues.append(g["y"][nodes].cpu())

        if g.get("cas_target") is not None:
            cas_preds.append(torch.sigmoid(cas_logit).cpu())
            cas_trues.append(g["cas_target"][nodes].view(-1,1).cpu())

        del x_b, xv_b, sub_ei, sub_et

    Y_pred = torch.cat(preds, dim=0).numpy()

    out = {"Y_pred": Y_pred}

    if len(trues) > 0:
        out["Y_true"] = torch.cat(trues, dim=0).numpy()

    if len(cas_preds) > 0:
        out["CAS_pred"] = torch.cat(cas_preds, dim=0).numpy()
        out["CAS_true"] = torch.cat(cas_trues, dim=0).numpy()

    results[y] = out

    print(f"MY{y} done | preds: {Y_pred.shape}")



import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


VAR_NAMES = ["ps", "tsurf", "dust", "u", "v", "temp", "uplift"]
HORIZONS = ["H1", "H3", "H6"]


def split_heads(Y):
    return Y[:,0:7], Y[:,7:14], Y[:,14:21]


def valid_mask(y_true, y_pred):
    return np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)


print("TABLE A — REGRESSION METRICS")

table_A = {}

for y in sorted(results.keys()):
    res = results[y]

    if "Y_true" not in res:
        print(f"MY{y}: no targets, skipping")
        continue

    Y_true = res["Y_true"]
    Y_pred = res["Y_pred"]

    y1_t, y3_t, y6_t = split_heads(Y_true)
    y1_p, y3_p, y6_p = split_heads(Y_pred)

    table_A[y] = {}

    for h_name, yt, yp in zip(HORIZONS, [y1_t, y3_t, y6_t], [y1_p, y3_p, y6_p]):

        mask = valid_mask(yt, yp)
        yt = yt[mask]
        yp = yp[mask]

        print(f"\nMY{y} — {h_name}")

        metrics = {}

        for i, var in enumerate(VAR_NAMES):

            mae = mean_absolute_error(yt[:,i], yp[:,i])
            rmse = np.sqrt(mean_squared_error(yt[:,i], yp[:,i]))
            r2 = r2_score(yt[:,i], yp[:,i])

            metrics[var] = (mae, rmse, r2)

            print(f"{var:8s} | MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

        table_A[y][h_name] = metrics



print("TABLE B — CAS CLASSIFICATION")


table_B = {}

for y in sorted(results.keys()):
    res = results[y]

    if "CAS_true" not in res:
        print(f"MY{y}: no CAS targets")
        continue

    y_true = res["CAS_true"].flatten()
    y_pred = res["CAS_pred"].flatten()

    y_bin = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_true, y_bin)
    prec = precision_score(y_true, y_bin)
    rec = recall_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)
    auc = roc_auc_score(y_true, y_pred)

    table_B[y] = (acc, prec, rec, f1, auc)

    print(f"\nMY{y}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")


print("TABLE C — OVERALL GENERALISATION")


for y in sorted(results.keys()):
    res = results[y]

    if "Y_true" not in res:
        continue

    Y_true = res["Y_true"]
    Y_pred = res["Y_pred"]

    mask = valid_mask(Y_true, Y_pred)

    yt = Y_true[mask]
    yp = Y_pred[mask]

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)

    print(f"\nMY{y}")
    print(f"Overall MAE : {mae:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Overall R2  : {r2:.4f}")