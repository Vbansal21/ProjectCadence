#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pathlib

path = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else "data/results/bench_uniform.csv"
)
df = pd.read_csv(path)

# ns/op per workload vs N (median)
g = df.groupby(["ds", "workload", "N"])["ns"].median().reset_index()
for (ds, wl), sub in g.groupby(["ds", "workload"]):
    sub = sub.sort_values("N")
    plt.figure()
    plt.loglog(sub["N"], sub["ns"] / sub["N"], marker="o")
    plt.xlabel("N")
    plt.ylabel("ns/op (approx)")
    plt.title(f"{ds} :: {wl}")
    out = path.parent / f"{ds}_{wl}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
print("Wrote plots to", path.parent)
