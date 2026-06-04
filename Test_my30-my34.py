import numpy as np
import matplotlib.pyplot as plt


VAR_IDX = {
    "dust": 2,
    "uplift": 6
}

YEARS = [30, 34]  
SAMPLE_SIZE = 5000 

def get_h1(Y):
    return Y[:, 0:7]


fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for row, var in enumerate(["dust", "uplift"]):
    for col, y in enumerate(YEARS):

        ax = axes[row, col]

        res = results[y]
        Y_true = res["Y_true"]
        Y_pred = res["Y_pred"]

        yt = get_h1(Y_true)
        yp = get_h1(Y_pred)

     
        mask = np.isfinite(yt).all(axis=1) & np.isfinite(yp).all(axis=1)
        yt = yt[mask]
        yp = yp[mask]

    
        t = yt[:, VAR_IDX[var]]
        p = yp[:, VAR_IDX[var]]

   
        if len(t) > SAMPLE_SIZE:
            idx = np.random.choice(len(t), SAMPLE_SIZE, replace=False)
            t = t[idx]
            p = p[idx]

    
        ax.scatter(t, p, alpha=0.3, s=10)

 
        min_v = min(t.min(), p.min())
        max_v = max(t.max(), p.max())
        ax.plot([min_v, max_v], [min_v, max_v], linestyle='--', color='black')

    
        ax.set_title(f"MY{y} — {var} (H1)")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")


plt.tight_layout()
plt.show()



plt.tight_layout()


plt.savefig(
    "my30_my34_dust_uplift_h1_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()