

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

YEARS_TO_COMPARE = [30, 34]
VARIABLE = "dust"
HORIZON = "H3"

VAR_NAMES = ["ps", "tsurf", "dust", "u", "v", "temp", "uplift"]

HORIZON_OFFSETS = {
    "H1": 0,
    "H3": 7,
    "H6": 14
}

var_idx = VAR_NAMES.index(VARIABLE)
col_idx = HORIZON_OFFSETS[HORIZON] + var_idx


def regional_r2_dataframe(year, results, graphs, col_idx, min_points=10):


    if year not in results:
        raise RuntimeError(f"MY{year}: missing from results")

    if year not in graphs:
        raise RuntimeError(f"MY{year}: missing from graphs")

    res = results[year]

    if "Y_true" not in res:
        raise RuntimeError(f"MY{year}: Y_true missing")

    if "Y_pred" not in res:
        raise RuntimeError(f"MY{year}: Y_pred missing")

    ids = graphs[year].get("ids", None)

    if ids is None:
        raise RuntimeError(f"MY{year}: ids missing")

    Y_true = res["Y_true"]
    Y_pred = res["Y_pred"]

    true_vals = Y_true[:, col_idx]
    pred_vals = Y_pred[:, col_idx]

    ids_arr = np.array(ids, dtype=object)

    valid = np.isfinite(true_vals) & np.isfinite(pred_vals)

    true_vals = true_vals[valid]
    pred_vals = pred_vals[valid]
    ids_valid = ids_arr[valid]

    region_to_rows = {}

    for i, item in enumerate(ids_valid):
        t_id, lat_id, lon_id, z_id = item
        region_key = (int(lat_id), int(lon_id))

        if region_key not in region_to_rows:
            region_to_rows[region_key] = []

        region_to_rows[region_key].append(i)

    regional_scores = []

    for region_key, rows in region_to_rows.items():
        rows = np.array(rows)

        if len(rows) < min_points:
            continue

        yt = true_vals[rows]
        yp = pred_vals[rows]

        if np.nanstd(yt) < 1e-8:
            continue

        r2 = r2_score(yt, yp)

        regional_scores.append({
            "Year": f"MY{year}",
            "year_numeric": year,
            "Variable": VARIABLE,
            "Horizon": HORIZON,
            "Region": f"lat{region_key[0]}_lon{region_key[1]}",
            "lat_id": region_key[0],
            "lon_id": region_key[1],
            "R2": r2,
            "n_points": len(rows)
        })

    if len(regional_scores) == 0:
        raise RuntimeError(f"MY{year}: no valid regional R2 scores computed")

    return pd.DataFrame(regional_scores), true_vals, pred_vals, ids_valid



regional_dfs = []
regional_context = {}

for year in YEARS_TO_COMPARE:
    df_year, true_vals, pred_vals, ids_valid = regional_r2_dataframe(
        year=year,
        results=results,
        graphs=graphs,
        col_idx=col_idx,
        min_points=10
    )

    regional_dfs.append(df_year)

    regional_context[year] = {
        "true_vals": true_vals,
        "pred_vals": pred_vals,
        "ids_valid": ids_valid
    }

regional_df = pd.concat(regional_dfs, ignore_index=True)

regional_df.to_csv(
    f"regional_r2_all_regions_{VARIABLE}_{HORIZON}_MY30_MY34.csv",
    index=False
)


display(regional_df.head().round(3))



summary_rows = []

for year in YEARS_TO_COMPARE:
    year_label = f"MY{year}"
    df_y = regional_df[regional_df["Year"] == year_label].copy()

    best_row = df_y.loc[df_y["R2"].idxmax()]
    worst_row = df_y.loc[df_y["R2"].idxmin()]

    summary_rows.append({
        "Year": year_label,
        "Variable": VARIABLE.capitalize(),
        "Horizon": HORIZON,
        "Mean regional R2": df_y["R2"].mean(),
        "Median regional R2": df_y["R2"].median(),
        "Worst regional R2": worst_row["R2"],
        "Best regional R2": best_row["R2"],
        "Worst region": worst_row["Region"],
        "Best region": best_row["Region"],
        "Valid regions": len(df_y)
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    f"regional_r2_summary_{VARIABLE}_{HORIZON}_MY30_MY34.csv",
    index=False
)


print("REGIONAL R2 SUMMARY — PAPER TABLE")

display(summary_df.round(3))


plt.figure(figsize=(8, 5))

for year in YEARS_TO_COMPARE:
    year_label = f"MY{year}"
    vals = regional_df.loc[regional_df["Year"] == year_label, "R2"].values

    plt.hist(
        vals,
        bins=30,
        alpha=0.55,
        density=True,
        label=f"{year_label}"
    )

plt.axvline(
    regional_df.loc[regional_df["Year"] == "MY30", "R2"].median(),
    linestyle="--",
    linewidth=2,
    label="MY30 median"
)

plt.axvline(
    regional_df.loc[regional_df["Year"] == "MY34", "R2"].median(),
    linestyle=":",
    linewidth=2,
    label="MY34 median"
)

plt.xlabel("Regional $R^2$")
plt.ylabel("Density")
plt.title("Regional Dust-Column Forecast Skill: MY30 vs MY34, H3")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(
    f"fig_04_regional_r2_distribution_MY30_MY34_{VARIABLE}_{HORIZON}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.savefig(
    f"fig_04_regional_r2_distribution_MY30_MY34_{VARIABLE}_{HORIZON}.pdf",
    bbox_inches="tight"
)
plt.show()


YEAR_EXAMPLE = 34
year_label = f"MY{YEAR_EXAMPLE}"

df_ex = regional_df[regional_df["Year"] == year_label].copy()
df_ex = df_ex.sort_values("R2", ascending=False).reset_index(drop=True)

best_row = df_ex.iloc[0]
median_row = df_ex.iloc[len(df_ex) // 2]
worst_row = df_ex.iloc[-1]

selected = [
    ("Best", best_row),
    ("Median", median_row),
    ("Worst", worst_row)
]

ctx = regional_context[YEAR_EXAMPLE]
true_vals = ctx["true_vals"]
pred_vals = ctx["pred_vals"]
ids_valid = ctx["ids_valid"]


def rows_for_region(ids_valid, lat_id, lon_id):
    rows = []
    for i, item in enumerate(ids_valid):
        t_id, la, lo, z_id = item
        if int(la) == int(lat_id) and int(lo) == int(lon_id):
            rows.append(i)
    return np.array(rows)


def time_for_rows(ids_valid, rows):
    times = []
    for r in rows:
        t_id, lat_id, lon_id, z_id = ids_valid[r]
        times.append(int(t_id))
    return np.array(times)


fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)

for ax, (label, row) in zip(axes, selected):

    lat_id = int(row["lat_id"])
    lon_id = int(row["lon_id"])
    r2 = float(row["R2"])

    rows = rows_for_region(ids_valid, lat_id, lon_id)

    t = time_for_rows(ids_valid, rows)
    y_true_region = true_vals[rows]
    y_pred_region = pred_vals[rows]

    order = np.argsort(t)

    t = t[order]
    y_true_region = y_true_region[order]
    y_pred_region = y_pred_region[order]

    ax.plot(t, y_true_region, label="Truth", linewidth=2)
    ax.plot(t, y_pred_region, label="Prediction", linestyle="--", linewidth=2)

    ax.set_title(
        f"{label} region lat{lat_id}_lon{lon_id} "
        f"(lat={lat_id}, lon={lon_id}) | $R^2$={r2:.3f}"
    )
    ax.set_ylabel("Dust column")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", frameon=True)

axes[-1].set_xlabel("Time chunk")

fig.suptitle(
    f"Regional Prediction Examples for MY{YEAR_EXAMPLE} Dust Column, {HORIZON}",
    fontsize=14,
    y=0.995
)

plt.tight_layout()

plt.savefig(
    f"fig_05_MY34_regional_examples_{VARIABLE}_{HORIZON}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.savefig(
    f"fig_05_MY34_regional_examples_{VARIABLE}_{HORIZON}.pdf",
    bbox_inches="tight"
)
plt.show()
