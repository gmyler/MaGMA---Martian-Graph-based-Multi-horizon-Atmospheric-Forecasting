

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import subgraph
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


YEARS_TO_RUN = [29, 30, 31, 32, 34]

VAR_NAMES = ["ps", "tsurf", "dust", "u", "v", "temp", "uplift"]
HORIZONS = ["H1", "H3", "H6"]

HORIZON_OFFSETS = {
    "H1": 0,
    "H3": 7,
    "H6": 14
}

SEQ_DIM = int(data.seq_dim)
T_CHUNK = int(data.T_CHUNK)

if hasattr(data, "N_VARS"):
    N_VARS = int(data.N_VARS)
else:
    N_VARS = SEQ_DIM // T_CHUNK

batch_t = 8


FOCUS_VAR = "dust"
FOCUS_HORIZON = "H3"
focus_col = HORIZON_OFFSETS[FOCUS_HORIZON] + VAR_NAMES.index(FOCUS_VAR)

def make_time_batches(ids, batch_t=8):
    t_ids = np.array([k[0] for k in ids])
    t_unique = np.unique(t_ids)

    for i in range(0, len(t_unique), batch_t):
        ts = t_unique[i:i + batch_t]
        idx = np.where(np.isin(t_ids, ts))[0]
        if len(idx) > 0:
            yield idx


def valid_mask(y_true, y_pred):
    return np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)


def compute_metrics(y_true, y_pred):
    mask = valid_mask(y_true, y_pred)

    yt = y_true[mask]
    yp = y_pred[mask]

    if len(yt) == 0:
        return np.nan, np.nan, np.nan, 0

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)

    return mae, rmse, r2, len(yt)


def compute_single_col_metrics(y_true, y_pred, col_idx):
    yt = y_true[:, col_idx]
    yp = y_pred[:, col_idx]

    mask = np.isfinite(yt) & np.isfinite(yp)

    yt = yt[mask]
    yp = yp[mask]

    if len(yt) == 0:
        return np.nan, np.nan, np.nan, 0

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)

    return mae, rmse, r2, len(yt)


def apply_input_ablation(x_b, xv_b, ablation_name):

    x_b = x_b.clone()
    xv_b = xv_b.clone()

    if ablation_name == "full":
        return x_b, xv_b

    if ablation_name == "no_vertical_column":
        xv_b[:] = 0.0
        return x_b, xv_b

    if ablation_name == "no_temporal_sequence":
        x_b[:, :SEQ_DIM] = 0.0
        return x_b, xv_b

    if ablation_name == "no_engineered_static":
        x_b[:, SEQ_DIM:] = 0.0
        return x_b, xv_b

    return x_b, xv_b


def apply_edge_ablation(sub_ei, sub_et, ablation_name):


    if ablation_name == "full":
        return sub_ei, sub_et

    if ablation_name == "no_graph_edges":
        empty_ei = torch.empty((2, 0), dtype=torch.long)
        empty_et = torch.empty((0,), dtype=torch.long)
        return empty_ei, empty_et

    if ablation_name == "no_similarity_edges":
        keep = sub_et != 3
        return sub_ei[:, keep], sub_et[keep]

    if ablation_name == "no_temporal_edges":
        keep = (sub_et != 1) & (sub_et != 2)
        return sub_ei[:, keep], sub_et[keep]

    if ablation_name == "no_spatial_edges":
        keep = sub_et != 0
        return sub_ei[:, keep], sub_et[keep]

    return sub_ei, sub_et


ABLATIONS = [
    {
        "name": "full",
        "label": "Full model",
        "description": "All inputs and graph relations active"
    },
    {
        "name": "no_vertical_column",
        "label": "No vertical column",
        "description": "Vertical tensor x_vert replaced with normalized mean"
    },
    {
        "name": "no_temporal_sequence",
        "label": "No temporal sequence",
        "description": "Raw temporal sequence features replaced with normalized mean"
    },
    {
        "name": "no_engineered_static",
        "label": "No engineered/static features",
        "description": "Engineered and static features replaced with normalized mean"
    },
    {
        "name": "no_graph_edges",
        "label": "No graph edges",
        "description": "Relational graph propagation removed using empty edge set"
    },
    {
        "name": "no_similarity_edges",
        "label": "No similarity edges",
        "description": "KNN/state-similarity edges removed"
    },
    {
        "name": "no_temporal_edges",
        "label": "No temporal graph edges",
        "description": "Direct and multi-step temporal edges removed"
    },
    {
        "name": "no_spatial_edges",
        "label": "No spatial graph edges",
        "description": "Local spatial adjacency edges removed"
    }
]


def run_ablation_inference_for_year(year, ablation_name):
    g = graphs[year]

    x = g["x"].float().cpu()
    xv = g["x_vert"].float().cpu()
    ei = g["edge_index"].long().cpu()
    et = g["edge_type"].long().cpu()

    ids = g["ids"]

    if ids is None:
        raise RuntimeError(f"MY{year}: ids missing")

    if g.get("y") is None:
        raise RuntimeError(f"MY{year}: y missing")

    N = x.shape[0]

    preds = []
    trues = []

    model.eval()

    with torch.no_grad():
        for nodes in make_time_batches(ids, batch_t=batch_t):

            nodes = torch.tensor(nodes, dtype=torch.long)

            sub_ei, sub_et = subgraph(
                nodes,
                ei,
                et,
                relabel_nodes=True,
                num_nodes=N
            )

            sub_ei, sub_et = apply_edge_ablation(
                sub_ei,
                sub_et,
                ablation_name
            )

            x_b = x[nodes].to(device)
            xv_b = xv[nodes].to(device)

            x_b, xv_b = apply_input_ablation(
                x_b,
                xv_b,
                ablation_name
            )

            sub_ei = sub_ei.to(device)
            sub_et = sub_et.to(device)

            y1, y3, y6, cas_logit, gates = model(
                x_b,
                xv_b,
                sub_ei,
                sub_et,
                router_temp=1.0
            )

            y_pred = torch.cat([y1, y3, y6], dim=1)

            preds.append(y_pred.cpu())
            trues.append(g["y"][nodes].cpu())

            del x_b, xv_b, sub_ei, sub_et

    Y_pred = torch.cat(preds, dim=0).numpy()
    Y_true = torch.cat(trues, dim=0).numpy()

    return Y_true, Y_pred


all_rows = []

for ab in ABLATIONS:
    ab_name = ab["name"]
    ab_label = ab["label"]

    print(f"Running ablation: {ab_label}")


    for year in YEARS_TO_RUN:

        print(f"MY{year}...")

        Y_true, Y_pred = run_ablation_inference_for_year(
            year=year,
            ablation_name=ab_name
        )

        mae, rmse, r2, n_valid = compute_metrics(Y_true, Y_pred)

        dust_mae, dust_rmse, dust_r2, dust_n = compute_single_col_metrics(
            Y_true,
            Y_pred,
            focus_col
        )

        all_rows.append({
            "Ablation": ab_label,
            "Ablation key": ab_name,
            "Description": ab["description"],
            "Martian Year": f"MY{year}",
            "Year": year,
            "Regime": "GDS / extreme" if year == 34 else "Regular",
            "Overall MAE": mae,
            "Overall RMSE": rmse,
            "Overall R2": r2,
            f"{FOCUS_VAR}_{FOCUS_HORIZON}_MAE": dust_mae,
            f"{FOCUS_VAR}_{FOCUS_HORIZON}_RMSE": dust_rmse,
            f"{FOCUS_VAR}_{FOCUS_HORIZON}_R2": dust_r2,
            "Valid samples": n_valid
        })

ablation_year_df = pd.DataFrame(all_rows)



full_lookup = ablation_year_df[
    ablation_year_df["Ablation key"] == "full"
][["Year", "Overall R2", f"{FOCUS_VAR}_{FOCUS_HORIZON}_R2"]].rename(
    columns={
        "Overall R2": "Full Overall R2",
        f"{FOCUS_VAR}_{FOCUS_HORIZON}_R2": f"Full {FOCUS_VAR}_{FOCUS_HORIZON}_R2"
    }
)

ablation_year_df = ablation_year_df.merge(
    full_lookup,
    on="Year",
    how="left"
)

ablation_year_df["Delta Overall R2"] = (
    ablation_year_df["Overall R2"] - ablation_year_df["Full Overall R2"]
)

ablation_year_df[f"Delta {FOCUS_VAR}_{FOCUS_HORIZON}_R2"] = (
    ablation_year_df[f"{FOCUS_VAR}_{FOCUS_HORIZON}_R2"]
    - ablation_year_df[f"Full {FOCUS_VAR}_{FOCUS_HORIZON}_R2"]
)

ablation_year_df.to_csv(
    "inference_component_sensitivity_by_year.csv",
    index=False
)


print("INFERENCE COMPONENT SENSITIVITY — BY YEAR")

display(ablation_year_df.round(4))

agg_df = ablation_year_df.groupby(
    ["Ablation", "Ablation key", "Description"]
).agg(
    Mean_MAE=("Overall MAE", "mean"),
    Mean_RMSE=("Overall RMSE", "mean"),
    Mean_R2=("Overall R2", "mean"),
    Mean_Delta_R2=("Delta Overall R2", "mean"),
    Dust_H3_Mean_R2=(f"{FOCUS_VAR}_{FOCUS_HORIZON}_R2", "mean"),
    Dust_H3_Mean_Delta_R2=(f"Delta {FOCUS_VAR}_{FOCUS_HORIZON}_R2", "mean")
).reset_index()


agg_df = agg_df.sort_values("Mean_Delta_R2", ascending=True)

agg_df.to_csv(
    "inference_component_sensitivity_overall.csv",
    index=False
)

print("INFERENCE COMPONENT SENSITIVITY — OVERALL")

display(agg_df.round(4))

regime_agg_df = ablation_year_df.groupby(
    ["Ablation", "Ablation key", "Regime"]
).agg(
    Mean_MAE=("Overall MAE", "mean"),
    Mean_RMSE=("Overall RMSE", "mean"),
    Mean_R2=("Overall R2", "mean"),
    Mean_Delta_R2=("Delta Overall R2", "mean"),
    Dust_H3_Mean_R2=(f"{FOCUS_VAR}_{FOCUS_HORIZON}_R2", "mean"),
    Dust_H3_Mean_Delta_R2=(f"Delta {FOCUS_VAR}_{FOCUS_HORIZON}_R2", "mean")
).reset_index()




display(regime_agg_df.round(4))
