
import glob, numpy as np, xarray as xr, torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

TRAIN_GLOB = # File path to data


T_CHUNK   = 8
LAT_PATCH = 3
LON_PATCH = 3
H_LIST    = [1, 3, 6]    
MAX_GAP   = 2
K_SIM     = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def open_and_merge(files):
    return xr.concat([xr.open_dataset(f, decode_times=False) for f in files], dim="time")

def chunk_indices(n, L):
    out=[]; i=0
    while i<n:
        j=min(i+L,n)
        out.append((i,j))
        i=j
    return out

def build_full_chunks(ds):
    t_bins  = chunk_indices(ds.sizes["time"], T_CHUNK)
    lat_bins= chunk_indices(ds.sizes["lat"],  LAT_PATCH)
    lon_bins= chunk_indices(ds.sizes["lon"],  LON_PATCH)
    z_bins  = [(0, ds.sizes["lev"])]
    chunks=[]
    for t0,t1 in t_bins:
        for la0,la1 in lat_bins:
            for lo0,lo1 in lon_bins:
                chunks.append({
                    "t0":t0,"t1":t1,
                    "la0":la0,"la1":la1,
                    "lo0":lo0,"lo1":lo1,
                    "z0":0,"z1":ds.sizes["lev"],
                    "z_dim":"lev","has_z":True
                })
    return chunks,t_bins,lat_bins,lon_bins,z_bins

def spatial_ids(chunks, t_bins, lat_bins, lon_bins):
    ids = []

    lat_patch = lat_bins[0][1] - lat_bins[0][0]
    lon_patch = lon_bins[0][1] - lon_bins[0][0]

    for c in chunks:
        t_id  = c["t0"] // T_CHUNK
        la_id = c["la0"] // lat_patch
        lo_id = c["lo0"] // lon_patch
        ids.append((t_id, la_id, lo_id, 0))

    return ids


def chunk_series(ds,c,var):
    sl_t=slice(c["t0"],c["t1"])
    sl_la=slice(c["la0"],c["la1"])
    sl_lo=slice(c["lo0"],c["lo1"])
    da=ds[var]
    if "lev" in da.dims:
        ts = da.isel(time=sl_t, lev=slice(c["z0"],c["z1"]),
                     lat=sl_la, lon=sl_lo).mean(dim=("lev","lat","lon"))
    else:
        ts = da.isel(time=sl_t, lat=sl_la, lon=sl_lo).mean(dim=("lat","lon"))
    arr = np.array(ts.values, dtype=np.float32)
    if len(arr) < T_CHUNK:  # pad if at very end
        pad = np.full((T_CHUNK-len(arr),), arr[-1], np.float32)
        arr = np.concatenate([arr, pad])
    return arr  # (T_CHUNK,)


def vertical_stats(ds,c,var):
    da = ds[var]
    if "lev" not in da.dims:
        return [0.0,0.0,0.0,0.0]
    t_last = c["t1"]-1
    sl_la=slice(c["la0"],c["la1"])
    sl_lo=slice(c["lo0"],c["lo1"])
    col = da.isel(time=t_last, lat=sl_la, lon=sl_lo).mean(dim=("lat","lon"))  # (lev,)
    arr = np.array(col.values, dtype=np.float32)
    if arr.ndim==0: arr=np.array([arr],np.float32)

    bot = float(arr[:max(1,len(arr)//5)].mean())   
    top = float(arr[-max(1,len(arr)//5):].mean())  
    vstd= float(arr.std())
    lapse = bot - top
    return [bot, top, vstd, lapse]


def vertical_column(ds, c, var):

    da = ds[var]
    t_last = c["t1"] - 1
    sl_la = slice(c["la0"], c["la1"])
    sl_lo = slice(c["lo0"], c["lo1"])
    Z = ds.sizes["lev"]

    if "lev" in da.dims:
        col = da.isel(time=t_last, lev=slice(c["z0"],c["z1"]),
                      lat=sl_la, lon=sl_lo).mean(dim=("lat","lon"))  # (lev,)
        arr = np.array(col.values, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((Z,), float(arr), np.float32)
        if arr.shape[0] != Z:
      
            if arr.shape[0] < Z:
                arr = np.pad(arr, (0, Z-arr.shape[0]), mode="edge")
            else:
                arr = arr[:Z]
        return arr
    else:
   
        val = da.isel(time=t_last, lat=sl_la, lon=sl_lo).mean(dim=("lat","lon"))
        v = float(val.values)
        return np.full((Z,), v, np.float32)


def compute_uplift(ps,ts,dust,u,v,temp, ps2,ts2,dust2,u2,v2,temp2):
    wind_fut=np.sqrt(u2*u2+v2*v2)
    dT = temp2 - temp
    return wind_fut * np.maximum(dT,0) * np.log(np.maximum(dust2,0)+1e-6)

def build_targets(ds,chunks,vars_list,H):
    Y=[]
    for c in chunks:
        f=dict(c)
        f["t0"]+=H*T_CHUNK
        f["t1"]+=H*T_CHUNK
        if f["t1"]>ds.sizes["time"]:
            Y.append([np.nan]*len(vars_list))
        else:
            arr=[]
            for v in vars_list:
                s = chunk_series(ds,f,v).mean()
                arr.append(float(s))
            Y.append(arr)
    return np.array(Y,np.float32)

def build_edges(chunks, Xn, t_bins, lat_bins, lon_bins):
    ids = spatial_ids(chunks, t_bins, lat_bins, lon_bins)
    id_to_n = {k: i for i, k in enumerate(ids)}
    edges = []
    etypes = []

    n_lon = len(lon_bins)
    ids_arr = np.array(ids)
    t_idx_all = ids_arr[:, 0]


    for i, (t, la, lo, z) in tqdm(enumerate(ids),
                                  total=len(ids),
                                  desc="Edge pass 1/3: Local + temporal"):
   
        for dla in [-1, 0, 1]:
            for dlo in [-1, 0, 1]:
                if dla == 0 and dlo == 0:
                    continue
                la2 = la + dla
                lo2 = (lo + dlo) % n_lon
                k = (t, la2, lo2, z)
                if k in id_to_n:
                    edges.append([i, id_to_n[k]])
                    etypes.append(0)


        k = (t + 1, la, lo, z)
        if k in id_to_n:
            edges.append([i, id_to_n[k]])
            etypes.append(1)

 
        for g in range(1, MAX_GAP + 1):
            k = (t + g, la, lo, z)
            if k in id_to_n:
                edges.append([i, id_to_n[k]])
                etypes.append(2)



    Xnorm = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-8)

 
    unique_t = np.unique(t_idx_all)

    for t in tqdm(unique_t, desc="Edge pass 2/3: K-SIM similarity edges"):
        idx = np.where(t_idx_all == t)[0]
        if len(idx) < 3:
            continue
        knn = NearestNeighbors(
            n_neighbors=min(K_SIM + 1, len(idx)),
            metric="cosine"
        ).fit(Xnorm[idx])

        _, nbrs = knn.kneighbors(Xnorm[idx])
        for ii, neighs in enumerate(nbrs):
            for jj in neighs[1:]:
                edges.append([idx[ii], idx[jj]])
                etypes.append(3)

    return (
        torch.tensor(edges).t().long(),
        torch.tensor(etypes).long()
    )


files=sorted(glob.glob(TRAIN_GLOB))
ds=open_and_merge(files)

core_vars=["ps","tsurf","dustcol","u","v","temp"]

chunks,t_bins,lat_bins,lon_bins,z_bins=build_full_chunks(ds)
ids=spatial_ids(chunks,t_bins,lat_bins,lon_bins)

lat_idx = np.array([k[1] for k in ids], np.float32)/len(lat_bins)
lon_idx = np.array([k[2] for k in ids], np.float32)/len(lon_bins)

la_ids = [k[1] for k in ids]
lo_ids = [k[2] for k in ids]

print("lat bins unique:", sorted(set(la_ids)), "count =", len(set(la_ids)))
print("lon bins unique:", sorted(set(lo_ids)), "count =", len(set(lo_ids)))


print("Nodes:", len(chunks))


vert_vars = ["temp", "u", "v", "dustcol", "ps"]

Z = ds.sizes["lev"]
Vv = len(vert_vars)


X_list=[]
Xvert_list=[]
storm_flag=[]

for c in tqdm(chunks, desc="Building V4.5 node features + x_vert"):
    seqs=[]
    last_vals=[]
    for v in core_vars:
        s = chunk_series(ds,c,v)
        seqs.append(s)
        last_vals.append(float(s[-1]))

    seqs = np.stack(seqs, axis=0)
    seq_flat = seqs.reshape(-1)

    ps,ts,dust,u,vv,temp = last_vals

    wind_now = np.sqrt(u*u + vv*vv)
    logd_now = np.log(np.maximum(dust,0)+1e-6)
    sqrd     = np.sqrt(np.maximum(dust,0))
    t_an     = temp - temp  # placeholder
    p_an     = ps   - ps
    flag     = 1.0 if dust>0.4 else 0.0

    uplift_curr = wind_now * max(temp,0.0) * logd_now


    vstats=[]
    for vname in core_vars:
        vstats += vertical_stats(ds,c,vname)

    eng = [logd_now, sqrd, wind_now, 0.0, 0.0, flag,
           lat_idx[len(X_list)], lon_idx[len(X_list)], uplift_curr]

    X_list.append(np.concatenate([seq_flat, np.array(eng,np.float32), np.array(vstats,np.float32)]))
    storm_flag.append(flag)

 
    cols=[]
    for vvname in vert_vars:
        cols.append(vertical_column(ds, c, vvname))  
    cols = np.stack(cols, axis=1)  
    Xvert_list.append(cols)

X_full  = np.array(X_list, np.float32)           
X_vert  = np.stack(Xvert_list, axis=0).astype(np.float32) 


last_ps   = X_full[:, (0*T_CHUNK)+(T_CHUNK-1)]
last_temp = X_full[:, (5*T_CHUNK)+(T_CHUNK-1)]
t_an = last_temp - last_temp.mean()
p_an = last_ps   - last_ps.mean()

seq_dim = len(core_vars)*T_CHUNK
eng_start = seq_dim
X_full[:, eng_start+3] = t_an
X_full[:, eng_start+4] = p_an


Xm = X_full.mean(0, keepdims=True)
Xs = X_full.std(0,  keepdims=True) + 1e-8
Xn = (X_full - Xm) / Xs


Xv_mean = X_vert.reshape(-1, Vv).mean(axis=0, keepdims=True)       
Xv_std  = X_vert.reshape(-1, Vv).std(axis=0, keepdims=True) + 1e-8  
Xv_norm = (X_vert - Xv_mean) / Xv_std                              


Y_heads=[]
for H in H_LIST:
    Y_core = build_targets(ds,chunks,core_vars,H)  

    ps,ts,dust,u,vv,temp = last_ps, X_full[:,(1*T_CHUNK)+(T_CHUNK-1)], X_full[:,(2*T_CHUNK)+(T_CHUNK-1)], \
                           X_full[:,(3*T_CHUNK)+(T_CHUNK-1)], X_full[:,(4*T_CHUNK)+(T_CHUNK-1)], last_temp

    ps2,ts2,dust2,u2,v2,temp2 = [Y_core[:,i] for i in range(6)]
    uplift = compute_uplift(ps,ts,dust,u,vv,temp, ps2,ts2,dust2,u2,v2,temp2)

    Y_fullH = np.concatenate([Y_core, uplift.reshape(-1,1)], axis=1)
    Y_heads.append(Y_fullH)

Y_heads = np.stack(Y_heads, axis=1)  

Y_all = Y_heads.reshape(-1,7)
Ym = np.nanmean(Y_all, axis=0, keepdims=True)
Ys = np.nanstd(Y_all, axis=0, keepdims=True) + 1e-8

Yn_heads=[]
for k in range(Y_heads.shape[1]):
    Yn_heads.append((Y_heads[:,k,:] - Ym) / Ys)
Yn = np.concatenate(Yn_heads, axis=1)  

edge_index, edge_type = build_edges(chunks, Xn, t_bins, lat_bins, lon_bins)


storm_mask = torch.tensor(np.array(storm_flag)>0, dtype=torch.bool)

data = Data(
    x=torch.tensor(Xn, dtype=torch.float32),
    x_vert=torch.tensor(Xv_norm, dtype=torch.float32),   
    y=torch.tensor(Yn, dtype=torch.float32),
    edge_index=edge_index,
    edge_type=edge_type,
    storm_mask=storm_mask,
    cas_target=storm_mask.float()                       
)

ei = data.edge_index  
et = data.edge_type
N  = data.num_nodes

print("Validating edges on CPU...")
print("N =", N, "| edges =", ei.shape[1])


ei_min = int(ei.min().item())
ei_max = int(ei.max().item())
et_min = int(et.min().item())
et_max = int(et.max().item())

print("edge_index min/max:", ei_min, ei_max)
print("edge_type  min/max:", et_min, et_max)

assert ei_min >= 0, f"edge_index has negative id: {ei_min}"
assert ei_max < N, f"edge_index out of range: max={ei_max} vs N={N}"
assert et_min >= 0, f"edge_type has negative type: {et_min}"

num_rel = et_max + 1
print("num_relations inferred:", num_rel)

print("edge_index / edge_type valid.")

data.X_mean = torch.tensor(Xm.squeeze(), dtype=torch.float32)
data.X_std  = torch.tensor(Xs.squeeze(), dtype=torch.float32)
data.Y_mean = torch.tensor(Ym.squeeze(), dtype=torch.float32)
data.Y_std  = torch.tensor(Ys.squeeze(), dtype=torch.float32)


data.Xv_mean = torch.tensor(Xv_mean.squeeze(), dtype=torch.float32)
data.Xv_std  = torch.tensor(Xv_std.squeeze(), dtype=torch.float32)
data.vert_vars = vert_vars
data.Z_levels  = Z

data.seq_dim   = seq_dim
data.H_LIST    = H_LIST
data.T_CHUNK   = T_CHUNK
data.core_vars = core_vars


print("x:", data.x.shape, "| x_vert:", data.x_vert.shape, "| y:", data.y.shape, "| edges:", data.edge_index.shape)
print("seq_dim:", data.seq_dim, "| horizons:", data.H_LIST, "| vert_vars:", data.vert_vars)
