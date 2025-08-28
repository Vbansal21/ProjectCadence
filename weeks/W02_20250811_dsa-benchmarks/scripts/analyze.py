#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pathlib
import sys

path = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else "data/results/bench_uniform.csv"
)
df = pd.read_csv(path)

# median + p10/p90 per (ds,impl,workload,N)
g = df.groupby(["ds", "impl", "workload", "N"])
stats = (
    g["ns"]
    .agg(
        median="median",
        p10=lambda x: np.percentile(x, 10),
        p90=lambda x: np.percentile(x, 90),
    )
    .reset_index()
)


# pivot custom vs stl (choose matching workload names)
def normalize_workload(w):
    return w.replace("(reserve)", "")


stats["wl_norm"] = stats["workload"].map(normalize_workload)
pivot = stats.pivot_table(index=["ds", "wl_norm", "N"], columns="impl", values="median")
pivot["speedup_custom_vs_stl"] = pivot["stl"] / pivot["custom"]

outdir = path.parent
stats.to_csv(
    outdir / f"{path.parts[-1].replace('.csv', '_analysis')}_summary_stats.csv",
    index=False,
)
pivot.to_csv(
    outdir / f"{path.parts[-1].replace('.csv', '_analysis')}_speedup_table.csv"
)

print(
    "Wrote:",
    outdir / f"{path.parts[-1].replace('.csv', '_analysis')}_summary_stats.csv",
)
print(
    "Wrote:",
    outdir / f"{path.parts[-1].replace('.csv', '_analysis')}_speedup_table.csv",
)
