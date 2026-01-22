#!/usr/bin/env python3
"""
ISEF Dry Lab: In-silico bacterial growth-curve simulation (Gompertz model)

Inputs (in same folder as this script):
  - parameters.csv
  - scenarios.csv
  - effects.csv

Outputs:
  - simulated_curves.csv
  - growth_metrics.csv
  - best_concentrations.csv

Notes:
  - Dopamine and acetylcholine are treated as constants at 500 uM for every scenario.
  - Effects are rule-based (% modifiers) and should be described as hypothesis-driven assumptions.
"""

from __future__ import annotations
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401

# -----------------------------
# Constants (project assumptions)
# -----------------------------
DOPAMINE_UM = 500
ACETYLCHOLINE_UM = 500

TOTAL_TIME_HR = 24.0
DT_HR = 0.5
TIME_POINTS = np.arange(0.0, TOTAL_TIME_HR + DT_HR, DT_HR)

# Replicate noise (small; makes replicates non-identical)
NOISE_SD = 0.015  # ~1.5%

np.random.seed(42)


def read_csv_robust(path: str) -> pd.DataFrame:
    """Read a CSV with a couple of encoding fallbacks (handles Excel/Mac quirks)."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_param_name(p: str) -> str:
    """Map many possible spellings to: 'mumax', 'lag', 'plateau'."""
    p0 = normalize_text(p).lower()
    p0 = p0.replace("µ", "u")  # just in case someone reintroduces micro symbol
    p0 = p0.replace(" ", "")
    p0 = p0.replace("_", "")
    if p0 in {"mumax", "umax", "mu", "um"} or "max" in p0:
        return "mumax"
    if "lag" in p0:
        return "lag"
    if "plateau" in p0 or p0 in {"k"}:
        return "plateau"
    return p0


def direction_sign(direction: str) -> float:
    """Increase -> +1, Decrease -> -1, else +1."""
    d = normalize_text(direction).lower()
    if "decrease" in d or d in {"-", "neg", "negative"}:
        return -1.0
    return 1.0


def get_baseline(params: pd.DataFrame, bacteria: str) -> dict:
    row = params[params["bacteria"] == bacteria]
    if row.empty:
        raise ValueError(f"Baseline not found for bacteria='{bacteria}'. Check parameters.csv")
    r = row.iloc[0]
    return {
        "lag": float(r["lag_lambda"]),
        "mumax": float(r["mumax"]),
        "K": float(r["plateau_K"]),
        "od0": float(r["initial_od600_t0"]),
    }


def get_modifiers(effects: pd.DataFrame, bacteria: str, micronutrient: str, conc_um: float) -> dict:
    """Return percent modifiers for lag, mumax, K (plateau)."""
    micronutrient = normalize_text(micronutrient)
    if micronutrient == "":
        return {"lag_pct": 0.0, "mumax_pct": 0.0, "K_pct": 0.0}

    # Filter matching rules
    sub = effects[
        (effects["bacteria"] == bacteria) &
        (effects["micronutrient"] == micronutrient) &
        (effects["concentration_uM"].astype(float) == float(conc_um))
    ]
    if sub.empty:
        # No matching rule -> no change
        return {"lag_pct": 0.0, "mumax_pct": 0.0, "K_pct": 0.0}

    mods = {"lag_pct": 0.0, "mumax_pct": 0.0, "K_pct": 0.0}
    for _, r in sub.iterrows():
        param = norm_param_name(r["parameter"])
        mag = float(r["magnitude_percent"])
        sign = direction_sign(r.get("direction", "Increase"))

        if param == "lag":
            # For lag, direction matters: decrease means shorter lag (negative % effect)
            mods["lag_pct"] += sign * mag
        elif param == "mumax":
            mods["mumax_pct"] += sign * mag
        elif param == "plateau":
            mods["K_pct"] += sign * mag

    return mods


def gompertz_od(t: np.ndarray, od0: float, K: float, mumax: float, lag: float) -> np.ndarray:
    """
    Modified Gompertz with explicit starting OD:
      OD(t) = od0 + (K - od0) * exp( -exp( (mumax*e/(K-od0))*(lag - t) + 1 ) )
    This keeps OD(t) near od0 at early time.
    """
    e = np.e
    span = max(K - od0, 1e-9)
    return od0 + span * np.exp(-np.exp((mumax * e / span) * (lag - t) + 1.0))


def compute_metrics(time: np.ndarray, od: np.ndarray, od0: float) -> dict:
    plateau = float(np.max(od))
    slopes = np.diff(od) / (time[1] - time[0])
    mumax_curve = float(np.max(slopes))

    threshold = od0 + 0.10 * (plateau - od0)
    lag_curve = np.nan
    for i in range(len(od)):
        if od[i] >= threshold:
            lag_curve = float(time[i])
            break

    auc = float(np.sum((od[:-1] + od[1:]) * 0.5 * (time[1] - time[0])))
    return {"lag_hr": lag_curve, "mumax_od_per_hr": mumax_curve, "plateau_od": plateau, "auc_0_24": auc}


def graph_interaction(
    metrics_df: pd.DataFrame,
    dv: str = "auc_0_24",
    error: str = "sd",
) -> None:
    if dv not in metrics_df.columns:
        print(f"Dependent variable '{dv}' not found in metrics.")
        return

    if error not in {"sd", "sem"}:
        print("Error type must be 'sd' or 'sem'.")
        return

    grouped = metrics_df.groupby(["bacteria", "concentration_uM"])[dv]
    stats = grouped.agg(["mean", "std", "count"]).reset_index()
    if error == "sem":
        stats["err"] = stats["std"] / np.sqrt(stats["count"])
    else:
        stats["err"] = stats["std"]

    plt.figure(figsize=(10, 6))
    for bacteria, sub in stats.groupby("bacteria"):
        sub = sub.sort_values("concentration_uM")
        plt.errorbar(
            sub["concentration_uM"],
            sub["mean"],
            yerr=sub["err"],
            marker="o",
            capsize=3,
            label=bacteria,
        )

    plt.title(f"Interaction Plot: {dv} vs Concentration")
    plt.xlabel("Concentration (uM)")
    plt.ylabel(dv)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Bacteria")
    plt.tight_layout()
    plt.show()


def graph_interaction_by_micronutrient(
    metrics_df: pd.DataFrame,
    dv: str = "auc_0_24",
    error: str = "sd",
    save_dir: str | None = None,
) -> None:
    def micronutrient_key(name: str) -> str:
        return normalize_text(name).lower().replace(" ", "")

    micronutrient_levels = {
        "iron": {"label": "Iron", "levels": [1, 10, 20]},
        "vitaminb5": {"label": "Vitamin B5", "levels": [10, 50, 100]},
        "vitaminb6": {"label": "Vitamin B6", "levels": [10, 50, 100]},
        "choline": {"label": "Choline", "levels": [200, 400, 600]},
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for m_key, cfg in micronutrient_levels.items():
        label = cfg["label"]
        levels = cfg["levels"]
        sub = metrics_df[metrics_df["micronutrient"].map(micronutrient_key) == m_key]
        if sub.empty:
            print(f"No data for micronutrient '{label}'.")
            continue

        sub = sub[sub["concentration_uM"].isin(levels)]
        if sub.empty:
            print(f"No data for micronutrient '{label}' at {levels}.")
            continue

        grouped = sub.groupby(["bacteria", "concentration_uM"])[dv]
        stats = grouped.agg(["mean", "std", "count"]).reset_index()
        if error == "sem":
            stats["err"] = stats["std"] / np.sqrt(stats["count"])
        else:
            stats["err"] = stats["std"]

        plt.figure(figsize=(10, 6))
        for bacteria, g in stats.groupby("bacteria"):
            g = g.sort_values("concentration_uM")
            plt.errorbar(
                g["concentration_uM"],
                g["mean"],
                yerr=g["err"],
                marker="o",
                capsize=3,
                label=bacteria,
            )

        plt.title(f"{label}: {dv} vs Concentration")
        plt.xlabel("Concentration (uM)")
        plt.ylabel(dv)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Bacteria")
        plt.tight_layout()
        if save_dir:
            safe_name = label.lower().replace(" ", "_")
            out_path = os.path.join(save_dir, f"{safe_name}_{dv}.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved: {out_path}")
        else:
            plt.show()


def graph_3d(curves_df: pd.DataFrame, bacteria: str | None = None, micronutrient: str | None = None) -> None:
    sub = curves_df
    if bacteria:
        sub = sub[sub["bacteria"] == bacteria]
    if micronutrient is not None:
        sub = sub[sub["micronutrient"] == micronutrient]

    if sub.empty:
        print("No data to plot for the requested filters.")
        return

    # Average replicates so each (time, concentration) becomes one point.
    grouped = sub.groupby(["concentration_uM", "time_hr"], as_index=False)["od600"].mean()

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        grouped["time_hr"],
        grouped["concentration_uM"],
        grouped["od600"],
        cmap="viridis",
        linewidth=0.2,
        antialiased=True,
    )
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Concentration (uM)")
    ax.set_zlabel("OD600")
    title_bits = ["3D Growth Surface"]
    if bacteria:
        title_bits.append(f"Bacteria: {bacteria}")
    if micronutrient is not None and micronutrient != "":
        title_bits.append(f"Micronutrient: {micronutrient}")
    ax.set_title(" | ".join(title_bits))
    plt.tight_layout()
    plt.show()


def get_arg_value(flag: str) -> str | None:
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))

    params_path = os.path.join(here, "parameters.csv")
    scen_path = os.path.join(here, "scenarios.csv")
    eff_path = os.path.join(here, "effects.csv")

    params = read_csv_robust(params_path)
    scenarios = read_csv_robust(scen_path)
    effects = read_csv_robust(eff_path)

    # Normalize key columns
    for df in (params, scenarios, effects):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    # Ensure required columns exist
    required_scen = {"scenario_id", "bacteria", "micronutrient", "concentration_uM", "replicates"}
    missing = required_scen - set(scenarios.columns)
    if missing:
        raise ValueError(f"scenarios.csv missing columns: {sorted(missing)}")

    # Fill safe defaults for control rows
    scenarios["micronutrient"] = scenarios["micronutrient"].fillna("")
    scenarios["concentration_uM"] = scenarios["concentration_uM"].fillna(0).astype(float)
    scenarios["replicates"] = scenarios["replicates"].fillna(3).astype(int)

    # Effects normalization
    effects["parameter"] = effects["parameter"].map(lambda x: normalize_text(x))
    effects["direction"] = effects["direction"].map(lambda x: normalize_text(x))
    effects["magnitude_percent"] = effects["magnitude_percent"].astype(float)
    effects["concentration_uM"] = effects["concentration_uM"].astype(float)

    curve_rows = []
    metric_rows = []

    for _, sc in scenarios.iterrows():
        bacteria = normalize_text(sc["bacteria"])
        micronutrient = normalize_text(sc["micronutrient"])
        conc = float(sc["concentration_uM"])
        n_reps = int(sc["replicates"])

        base = get_baseline(params, bacteria)
        mods = get_modifiers(effects, bacteria, micronutrient, conc)

        for rep in range(1, n_reps + 1):
            # small replicate variability
            noise_mumax = np.random.normal(0.0, NOISE_SD)
            noise_K = np.random.normal(0.0, NOISE_SD)

            mumax = base["mumax"] * (1.0 + mods["mumax_pct"] / 100.0) * (1.0 + noise_mumax)
            lag = base["lag"] * (1.0 + mods["lag_pct"] / 100.0)  # lag_pct negative -> decrease
            K = base["K"] * (1.0 + mods["K_pct"] / 100.0) * (1.0 + noise_K)

            # Guardrails
            mumax = max(mumax, 1e-6)
            lag = max(lag, 0.0)
            K = max(K, base["od0"] + 1e-6)

            od = gompertz_od(TIME_POINTS, base["od0"], K, mumax, lag)

            for t, v in zip(TIME_POINTS, od):
                curve_rows.append({
                    "scenario_id": sc["scenario_id"],
                    "bacteria": bacteria,
                    "micronutrient": micronutrient,
                    "concentration_uM": conc,
                    "dopamine_uM": DOPAMINE_UM,
                    "acetylcholine_uM": ACETYLCHOLINE_UM,
                    "replicate": rep,
                    "time_hr": float(t),
                    "od600": float(v),
                })

            m = compute_metrics(TIME_POINTS, od, base["od0"])
            metric_rows.append({
                "scenario_id": sc["scenario_id"],
                "bacteria": bacteria,
                "micronutrient": micronutrient,
                "concentration_uM": conc,
                "dopamine_uM": DOPAMINE_UM,
                "acetylcholine_uM": ACETYLCHOLINE_UM,
                "replicate": rep,
                **m,
            })

    curves_df = pd.DataFrame(curve_rows)
    metrics_df = pd.DataFrame(metric_rows)

    curves_df.to_csv(os.path.join(here, "simulated_curves.csv"), index=False)
    metrics_df.to_csv(os.path.join(here, "growth_metrics.csv"), index=False)

    # Best concentration per bacteria & micronutrient (by mean AUC across replicates)
    best_rows = []
    non_controls = metrics_df[metrics_df["micronutrient"] != ""].copy()
    if not non_controls.empty:
        grouped = non_controls.groupby(["bacteria", "micronutrient", "concentration_uM"], as_index=False)["auc_0_24"].mean()
        for (b, m), g in grouped.groupby(["bacteria", "micronutrient"]):
            best = g.loc[g["auc_0_24"].idxmax()]
            best_rows.append(best)
        best_df = pd.DataFrame(best_rows)
    else:
        best_df = pd.DataFrame(columns=["bacteria", "micronutrient", "concentration_uM", "auc_0_24"])

    best_df.to_csv(os.path.join(here, "best_concentrations.csv"), index=False)

    if "--plot2d" in sys.argv:
        dv_arg = get_arg_value("--dv") or "auc_0_24"
        error_arg = get_arg_value("--error") or "sd"
        graph_interaction(metrics_df, dv=dv_arg, error=error_arg)

    if "--plot2d-micro" in sys.argv:
        dv_arg = get_arg_value("--dv") or "auc_0_24"
        error_arg = get_arg_value("--error") or "sd"
        outdir_arg = get_arg_value("--outdir") or "plots"
        graph_interaction_by_micronutrient(
            metrics_df,
            dv=dv_arg,
            error=error_arg,
            save_dir=outdir_arg,
        )

    if "--anova2" in sys.argv:
        dv_arg = get_arg_value("--dv") or "auc_0_24"
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
        except Exception:
            print("statsmodels is required for two-way ANOVA. Try: pip install statsmodels")
        else:
            formula = f"{dv_arg} ~ C(bacteria) * C(concentration_uM)"
            model = smf.ols(formula, data=metrics_df).fit()
            table = sm.stats.anova_lm(model, typ=2)
            print("\nTwo-way ANOVA (bacteria x concentration):")
            print(table.to_string())

    if "--plot3d" in sys.argv:
        bacteria_arg = get_arg_value("--bacteria")
        micronutrient_arg = get_arg_value("--micronutrient")
        graph_3d(curves_df, bacteria=bacteria_arg, micronutrient=micronutrient_arg)

    print("✅ Simulation complete.")
    print("Wrote: simulated_curves.csv, growth_metrics.csv, best_concentrations.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("\n❌ Error:", e)
        sys.exit(1)
