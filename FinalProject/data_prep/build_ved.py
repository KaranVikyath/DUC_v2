"""Build VED.mat / VED10k.mat from the Vehicle Energy Dataset (Apache-2.0).

Source: https://github.com/gsoh/VED  @ 6baa4963782d515a67d32a5490bd5d11f5d9bf0d
        Data/VED_DynamicData_Part1.7z, Part2.7z, VED_Static_Data_*.xlsx
Paper:  Oh, LeBlanc & Peng, "Vehicle Energy Dataset (VED)", IEEE T-ITS 2020.

Steps (all deterministic, seed 0):
  1. extract both .7z archives (py7zr) -> 54 weekly CSVs, 22,436,808 rows
  2. per-(VehId, non-null pattern) row counts -> per-column coverage by EngineType,
     and per-vehicle fully-observed row counts for candidate column sets
  3. columns = the LARGEST set of engine signals for which >= 10 vehicles still have
     >= ROWS_PER_VEH fully observed rows (IDs, time, lat/lon never used as features)
  4. rows fully observed on those columns; drop engine-off records
     (Vehicle Speed == 0 and Engine RPM == 0: parked with the logger on)
  5. the 10 vehicles with the most remaining rows; label = rank of VehId ascending
  6. per vehicle, one random permutation (rng = default_rng(0), drawn in label order)
     of its eligible rows, which spans all of its trips; VED.mat takes the first 8,500,
     VED10k.mat the first 1,000 (so VED10k is a subset of VED). Rows are then sorted by
     (label, DayNum, Trip, Timestamp). Values stay in ORIGINAL units (the harness z-scores).

Usage:  py -3.10 data_prep/build_ved.py   (clone github.com/gsoh/VED at VED_COMMIT into
        third_party/ved_raw/repo first; py7zr is needed to extract the archives)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

FP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
HERE = os.path.join(FP, "third_party", "ved_raw")           # raw downloads (gitignored)
REPO = os.path.join(HERE, "repo")                            # git clone of gsoh/VED
CSV = os.path.join(HERE, "csv")
OUT = os.path.join(FP, "data")
SITE = os.path.join(FP, "third_party", "_site")
VED_COMMIT = "6baa4963782d515a67d32a5490bd5d11f5d9bf0d"

ROWS_PER_VEH = 8500          # 10 x 8,500 = 85,000
ROWS_PER_VEH_SMALL = 1000    # 10 x 1,000 = 10,000
N_VEH = 10
SEED = 0

SIG = ["Vehicle Speed[km/h]", "MAF[g/sec]", "Engine RPM[RPM]", "Absolute Load[%]",
       "OAT[DegC]", "Fuel Rate[L/hr]", "Air Conditioning Power[kW]",
       "Air Conditioning Power[Watts]", "Heater Power[Watts]", "HV Battery Current[A]",
       "HV Battery SOC[%]", "HV Battery Voltage[V]", "Short Term Fuel Trim Bank 1[%]",
       "Short Term Fuel Trim Bank 2[%]", "Long Term Fuel Trim Bank 1[%]",
       "Long Term Fuel Trim Bank 2[%]"]
KEYS = ["DayNum", "VehId", "Trip", "Timestamp(ms)"]
SPEED, RPM = "Vehicle Speed[km/h]", "Engine RPM[RPM]"

# Candidate column sets, most to least complete (checked in step 3, largest first).
CANDIDATES = {
    "core3 (speed, MAF, RPM)": [0, 1, 2],
    "core4 (+Absolute Load)": [0, 1, 2, 3],
    "core4 + STFT1 + LTFT1": [0, 1, 2, 3, 12, 14],
    "core4 + OAT": [0, 1, 2, 3, 4],
    "core4 + OAT + STFT1 + LTFT1": [0, 1, 2, 3, 4, 12, 14],
    "core4 + STFT1/2 + LTFT1/2": [0, 1, 2, 3, 12, 13, 14, 15],
    "core4 + OAT + STFT1/2 + LTFT1/2": [0, 1, 2, 3, 4, 12, 13, 14, 15],
    "core4 + OAT + Fuel Rate": [0, 1, 2, 3, 4, 5],
    "HV: speed,MAF,RPM,OAT,AC kW,HV I/SOC/V": [0, 1, 2, 4, 6, 9, 10, 11],
    "HV + STFT1/2": [0, 1, 2, 4, 6, 9, 10, 11, 12, 13],
    "all 16": list(range(16)),
}


def extract():
    if len(glob.glob(os.path.join(CSV, "VED_*_week.csv"))) == 54:
        return
    sys.path.insert(0, SITE)
    import py7zr                                              # 0.22.0 in _site
    os.makedirs(CSV, exist_ok=True)
    for p in ("VED_DynamicData_Part1.7z", "VED_DynamicData_Part2.7z"):
        with py7zr.SevenZipFile(os.path.join(REPO, "Data", p), "r") as z:
            z.extractall(CSV)


def static_types():
    s1 = pd.read_excel(os.path.join(REPO, "Data", "VED_Static_Data_ICE&HEV.xlsx"))
    s2 = pd.read_excel(os.path.join(REPO, "Data", "VED_Static_Data_PHEV&EV.xlsx"))
    s1 = s1.rename(columns={"Vehicle Type": "EngineType"})
    st = pd.concat([s1, s2], ignore_index=True).drop_duplicates("VehId").set_index("VehId")
    return st


def read_week(f, cols):
    """One weekly file, only `cols`, plus src_row (0-based data row in the CSV).

    The CSV is parsed once into a float32 parquet cache (same row order, lat/lon
    dropped) so that later passes read only the columns they need: the build then
    fits in well under 1 GB of RAM.
    """
    pq = os.path.join(HERE, "parquet", os.path.basename(f).replace(".csv", ".parquet"))
    if not os.path.exists(pq):
        dt = {c: np.float32 for c in SIG}
        dt.update({"DayNum": np.float64, "VehId": np.int32, "Trip": np.int32,
                   "Timestamp(ms)": np.int64})
        os.makedirs(os.path.dirname(pq), exist_ok=True)
        pd.read_csv(f, usecols=KEYS + SIG, dtype=dt)[KEYS + SIG].to_parquet(pq, index=False)
    df = pd.read_parquet(pq, columns=list(cols))
    df["src_row"] = np.arange(len(df), dtype=np.int64)
    return df


def main():
    extract()
    files = sorted(glob.glob(os.path.join(CSV, "VED_*_week.csv")))
    assert len(files) == 54, len(files)
    st = static_types()

    # ---- pass 1: non-null pattern counts per vehicle ---------------------------------
    agg, total = [], 0
    for f in files:
        df = read_week(f, ["VehId"] + SIG)
        total += len(df)
        nn = df[SIG].notna().to_numpy()
        m = (nn.astype(np.int64) << np.arange(16)).sum(1)
        agg.append(pd.DataFrame({"VehId": df.VehId.to_numpy(), "mask": m})
                   .groupby(["VehId", "mask"]).size().rename("n").reset_index())
    A = pd.concat(agg).groupby(["VehId", "mask"]).n.sum().reset_index()
    A["type"] = A.VehId.map(st.EngineType).fillna("unknown")
    bits = ((A["mask"].to_numpy()[:, None] >> np.arange(16)) & 1).astype(bool)
    n = A.n.to_numpy()
    report = {"total_rows": int(total), "n_vehicles": int(A.VehId.nunique()),
              "rows_by_type": {k: int(v) for k, v in A.groupby("type").n.sum().items()},
              "vehicles_by_type": {k: int(v) for k, v in A.groupby("type").VehId.nunique().items()}}
    types = sorted(A.type.unique())
    cov = {}
    for j, c in enumerate(SIG):
        row = {"all": 100.0 * n[bits[:, j]].sum() / n.sum()}
        for t in types:
            s = (A.type == t).to_numpy()
            row[t] = 100.0 * n[s & bits[:, j]].sum() / n[s].sum()
        cov[c] = row
    report["coverage_pct"] = cov

    cand = {}
    for name, idx in CANDIDATES.items():
        need = sum(1 << j for j in idx)
        ok = (A["mask"].to_numpy() & need) == need
        pv = A[ok].groupby("VehId").n.sum().sort_values(ascending=False)
        cand[name] = dict(F=len(idx), complete_rows=int(pv.sum()), vehicles=int(len(pv)),
                          veh_ge_target=int((pv >= ROWS_PER_VEH).sum()),
                          top10_min=int(pv.iloc[N_VEH - 1]) if len(pv) >= N_VEH else 0)
    report["candidates"] = cand
    feasible = [k for k, v in cand.items() if v["veh_ge_target"] >= N_VEH]
    chosen = max(feasible, key=lambda k: (cand[k]["F"], cand[k]["complete_rows"]))
    cols = [SIG[j] for j in CANDIDATES[chosen]]
    report["chosen_set"] = chosen
    report["columns"] = cols

    # ---- pass 2: fully observed rows on the chosen columns ----------------------------
    parts = []
    for fi, f in enumerate(files):
        df = read_week(f, KEYS + cols)
        df = df.loc[df[cols].notna().all(1), KEYS + ["src_row"] + cols]
        df["src_file"] = np.int16(fi)
        parts.append(df)
    D = pd.concat(parts, ignore_index=True)
    del parts
    n_complete = len(D)
    off = (D[SPEED] == 0) & (D[RPM] == 0)
    report["complete_rows"] = int(n_complete)
    report["engine_off_dropped"] = int(off.sum())
    D = D.loc[~off]
    per_veh = D.groupby("VehId").size().sort_values(ascending=False)
    top = per_veh.head(N_VEH)
    assert top.min() >= ROWS_PER_VEH, top
    vehids = np.sort(top.index.to_numpy())                    # label k <-> vehids[k-1]
    report["top_vehicles_by_rows"] = {int(v): int(c) for v, c in per_veh.head(25).items()}

    rng = np.random.default_rng(SEED)
    big, small, vinfo = [], [], []
    for k, v in enumerate(vehids, start=1):
        Dv = D.loc[D.VehId == v].sort_values(["DayNum", "Trip", "Timestamp(ms)"])
        perm = rng.permutation(len(Dv))
        sb = Dv.iloc[perm[:ROWS_PER_VEH]].assign(lab=k)
        ss = Dv.iloc[perm[:ROWS_PER_VEH_SMALL]].assign(lab=k)
        big.append(sb)
        small.append(ss)
        s = st.loc[v] if v in st.index else None
        vinfo.append(dict(label=k, VehId=int(v), eligible_rows=int(len(Dv)),
                          trips_total=int(Dv.Trip.nunique()),
                          trips_in_VED=int(sb.Trip.nunique()),
                          trips_in_VED10k=int(ss.Trip.nunique()),
                          max_rows_one_trip_VED=int(sb.Trip.value_counts().iloc[0]),
                          EngineType=None if s is None else str(s.EngineType),
                          engine=None if s is None else str(s["Engine Configuration & Displacement"]),
                          weight_lb=None if s is None else str(s["Generalized_Weight"])))
    report["vehicles"] = vinfo

    def save(parts_, fn):
        S = pd.concat(parts_).sort_values(["lab", "DayNum", "Trip", "Timestamp(ms)"])
        fea = S[cols].to_numpy(np.float64)
        assert np.isfinite(fea).all()
        sio.savemat(os.path.join(OUT, fn), {
            "fea": fea,                                              # (N, F) original units
            "lab": S["lab"].to_numpy(np.int32)[:, None],             # (N, 1) int 1..10
            "vehid": vehids.astype(np.float64)[:, None],             # label k -> VehId
            "columns": np.array(cols, dtype=object),
            "row_vehid": S["VehId"].to_numpy(np.float64)[:, None],
            "row_trip": S["Trip"].to_numpy(np.float64)[:, None],
            "row_daynum": S["DayNum"].to_numpy(np.float64)[:, None],
            "row_timestamp_ms": S["Timestamp(ms)"].to_numpy(np.float64)[:, None],
            "row_src_file": S["src_file"].to_numpy(np.float64)[:, None],   # index into sorted CSV list
            "row_src_row": S["src_row"].to_numpy(np.float64)[:, None],     # 0-based data row
            "source": f"github.com/gsoh/VED@{VED_COMMIT} (Apache-2.0); build_ved.py seed {SEED}",
        }, do_compression=True)
        return fea.shape

    report["VED.mat"] = save(big, "VED.mat")
    report["VED10k.mat"] = save(small, "VED10k.mat")
    report["csv_files"] = [os.path.basename(f) for f in files]
    with open(os.path.join(HERE, "ved_build_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "csv_files"}, indent=1, default=str))


if __name__ == "__main__":
    main()
