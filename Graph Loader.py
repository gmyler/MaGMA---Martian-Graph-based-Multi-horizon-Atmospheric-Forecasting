

import numpy as np
import torch
import pickle


GRAPH_PATH = # file path

with open(GRAPH_PATH, "rb") as f:
    raw = pickle.load(f)

graph = raw["data"]
ids   = raw["ids"]

data = graph  

print("Loaded graph")
print(type(data))

print("Nodes:", data.num_nodes)
print("Edges:", data.edge_index.shape[1])
print("Node feature shape:", data.x.shape)
print("Vertical feature shape:", data.x_vert.shape)
print("Targets shape:", data.y.shape)

avg_degree = data.edge_index.shape[1] / data.num_nodes
print("Average node degree:", avg_degree)


edge_types = data.edge_type.numpy()

for i in range(edge_types.max() + 1):
    count = np.sum(edge_types == i)
    print(f"Edge type {i}: {count}")

print("""
Edge meanings:
0 = spatial (local grid)
1 = temporal (t → t+1)
2 = temporal skip (t → t+g)
3 = similarity (KNN)
""")


storm_nodes = data.storm_mask.sum().item()
total_nodes = data.num_nodes

print("Storm nodes:", storm_nodes)
print("Storm ratio:", storm_nodes / total_nodes)



X = data.x.numpy()

print("Feature mean (first 10):", X.mean(axis=0)[:10])
print("Feature std  (first 10):", X.std(axis=0)[:10])


Y = data.y.numpy()

print("Target mean:", np.nanmean(Y, axis=0))
print("Target std:", np.nanstd(Y, axis=0))


print("Vertical levels:", data.Z_levels)
print("Vertical vars:", data.vert_vars)

sample_vert = data.x_vert[0].numpy()

print("Sample column shape:", sample_vert.shape)

t_ids = np.array([k[0] for k in ids])

print("Unique time chunks:", len(np.unique(t_ids)))
print("Time range:", t_ids.min(), "→", t_ids.max())


lat_ids = np.array([k[1] for k in ids])
lon_ids = np.array([k[2] for k in ids])

print("Unique lat bins:", len(np.unique(lat_ids)))
print("Unique lon bins:", len(np.unique(lon_ids)))

print("Lat range:", lat_ids.min(), "→", lat_ids.max())
print("Lon range:", lon_ids.min(), "→", lon_ids.max())



sample_nodes = np.random.choice(data.num_nodes, 5, replace=False)

for n in sample_nodes:
    t, la, lo, z = ids[n]
    print(f"Node {n}: t={t}, lat_bin={la}, lon_bin={lo}")



src = data.edge_index[0].numpy()
dst = data.edge_index[1].numpy()

print("Min node id:", src.min(), dst.min())
print("Max node id:", src.max(), dst.max())

if src.max() < data.num_nodes and dst.max() < data.num_nodes:
    print("Edge indices valid")
else:
    print("Edge index problem")



print("Any NaNs in X:", np.isnan(X).any())
print("Any NaNs in Y:", np.isnan(Y).any())
print("Any NaNs in x_vert:", np.isnan(data.x_vert.numpy()).any())

