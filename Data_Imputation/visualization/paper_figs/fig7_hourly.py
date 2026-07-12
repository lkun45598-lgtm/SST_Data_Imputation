"""Figure 7: Per-hour analysis (2x2 layout).

(a) Per-hour MAE — 24 bars/line, errorbars = std
(b) Per-hour RMSE — same
(c) Mean diurnal SST cycle (2024-07) — FNO reconstruction vs raw observations,
    spatial mean over SCS ocean
(d) Mean diurnal cycle at point (19°N, 115°E) — FNO vs raw observations
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import apply_paper_style, save_fig, annotate_metric, PALETTE

CACHE = Path(__file__).parent / "cache" / "fig7_stats.npz"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def k_to_c(k):
    return k - 273.15


def main():
    apply_paper_style()
    data = np.load(CACHE, allow_pickle=True)
    per_hour = data["per_hour"].item()
    diurnal = data["diurnal"].item()

    hours = np.arange(24)
    mae = np.full(24, np.nan)
    mae_std = np.full(24, np.nan)
    rmse = np.full(24, np.nan)
    rmse_std = np.full(24, np.nan)
    for h in hours:
        if h in per_hour:
            mae[h] = per_hour[h]["mae_mean"]
            mae_std[h] = per_hour[h]["mae_std"]
            rmse[h] = per_hour[h]["rmse_mean"]
            rmse_std[h] = per_hour[h]["rmse_std"]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(
        2, 2,
        hspace=0.32, wspace=0.20,
        left=0.07, right=0.97, top=0.94, bottom=0.07,
    )

    # ===== (a) Per-hour MAE =====
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.errorbar(hours, mae, yerr=mae_std, fmt="o-",
                  color=PALETTE["model"], markerfacecolor=PALETTE["model"],
                  markeredgecolor="#1c4a72", markersize=7,
                  linewidth=1.8, capsize=3, capthick=1,
                  ecolor="#7fa7c4", elinewidth=1, alpha=0.95)
    ax_a.axhline(np.nanmean(mae), color="#a83333", linestyle="--", lw=1.2,
                 alpha=0.7, label=f"24-h mean = {np.nanmean(mae):.3f} K")
    ax_a.set_xlabel("Hour of day (UTC)", fontsize=12)
    ax_a.set_ylabel("MAE (K)", fontsize=12)
    ax_a.set_title("(a) Per-hour MAE", fontsize=14, pad=8, fontweight="bold")
    ax_a.set_xticks(np.arange(0, 24, 2))
    ax_a.set_xlim(-0.5, 23.5)
    ax_a.tick_params(labelsize=11)
    ax_a.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax_a.legend(loc="upper right", fontsize=10, frameon=True)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ===== (b) Per-hour RMSE =====
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.errorbar(hours, rmse, yerr=rmse_std, fmt="o-",
                  color="#d4956b", markerfacecolor="#d4956b",
                  markeredgecolor="#8a5a3a", markersize=7,
                  linewidth=1.8, capsize=3, capthick=1,
                  ecolor="#e0b48a", elinewidth=1, alpha=0.95)
    ax_b.axhline(np.nanmean(rmse), color="#a83333", linestyle="--", lw=1.2,
                 alpha=0.7, label=f"24-h mean = {np.nanmean(rmse):.3f} K")
    ax_b.set_xlabel("Hour of day (UTC)", fontsize=12)
    ax_b.set_ylabel("RMSE (K)", fontsize=12)
    ax_b.set_title("(b) Per-hour RMSE", fontsize=14, pad=8, fontweight="bold")
    ax_b.set_xticks(np.arange(0, 24, 2))
    ax_b.set_xlim(-0.5, 23.5)
    ax_b.tick_params(labelsize=11)
    ax_b.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax_b.legend(loc="upper right", fontsize=10, frameon=True)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # ===== (c) Mean diurnal SST cycle (spatial mean across SCS) =====
    ax_c = fig.add_subplot(gs[1, 0])
    pred_c = k_to_c(diurnal["hourly_mean_pred"])
    obs_c = k_to_c(diurnal["hourly_mean_obs"])
    ax_c.plot(hours, pred_c, "o-", color=PALETTE["model"],
              markeredgecolor="#1c4a72", markersize=7, linewidth=2.0,
              label="FNO-CBAM reconstruction")
    ax_c.plot(hours, obs_c, "s-", color=PALETTE["gt"],
              markeredgecolor="#7a1f1f", markersize=7, linewidth=2.0,
              label="Raw JAXA observations")
    # Amplitude annotation
    amp_pred = np.nanmax(pred_c) - np.nanmin(pred_c)
    amp_obs = np.nanmax(obs_c) - np.nanmin(obs_c)
    ax_c.set_xlabel("Hour of day (UTC)", fontsize=12)
    ax_c.set_ylabel("Spatial mean SST (°C)", fontsize=12)
    ax_c.set_title("(c) Diurnal cycle (SCS spatial mean, July 2024)",
                   fontsize=14, pad=8, fontweight="bold")
    ax_c.set_xticks(np.arange(0, 24, 2))
    ax_c.set_xlim(-0.5, 23.5)
    ax_c.tick_params(labelsize=11)
    ax_c.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax_c.legend(loc="lower right", fontsize=10, frameon=True)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    annotate_metric(ax_c, [
        f"Pred amp:  {amp_pred:.2f} K",
        f"Obs amp:   {amp_obs:.2f} K",
        f"n days = 31",
    ], loc="upper right", fontsize=11)

    # ===== (d) Point time series at (19°N, 115°E) =====
    ax_d = fig.add_subplot(gs[1, 1])
    pp_mean = k_to_c(diurnal["point_pred_mean"])
    pp_std = diurnal["point_pred_std"]  # already in K (= °C diff)
    po_mean = k_to_c(diurnal["point_obs_mean"])
    po_count = diurnal["point_obs_count"]

    # Plot pred with std-band
    ax_d.fill_between(hours, pp_mean - pp_std, pp_mean + pp_std,
                      color=PALETTE["model"], alpha=0.18,
                      label="FNO ± std across July days")
    ax_d.plot(hours, pp_mean, "o-", color=PALETTE["model"],
              markeredgecolor="#1c4a72", markersize=7, linewidth=2.0,
              label="FNO-CBAM mean")
    # Observations
    valid_obs = ~np.isnan(po_mean)
    ax_d.plot(hours[valid_obs], po_mean[valid_obs], "s",
              color=PALETTE["gt"], markeredgecolor="#7a1f1f",
              markersize=8, linewidth=0, label="Raw observation mean")
    # Diurnal amplitude
    amp_pp = np.nanmax(pp_mean) - np.nanmin(pp_mean)
    amp_po = np.nanmax(po_mean) - np.nanmin(po_mean)
    ax_d.set_xlabel("Hour of day (UTC)", fontsize=12)
    ax_d.set_ylabel("SST at (19°N, 115°E) (°C)", fontsize=12)
    ax_d.set_title("(d) Diurnal cycle at 19°N, 115°E (July 2024)",
                   fontsize=14, pad=8, fontweight="bold")
    ax_d.set_xticks(np.arange(0, 24, 2))
    ax_d.set_xlim(-0.5, 23.5)
    ax_d.tick_params(labelsize=11)
    ax_d.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax_d.legend(loc="lower right", fontsize=10, frameon=True)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    annotate_metric(ax_d, [
        f"FNO amp:  {amp_pp:.2f} K",
        f"Obs amp:  {amp_po:.2f} K",
    ], loc="upper right", fontsize=11)

    fig.suptitle("Per-hour evaluation (24 hourly models)",
                 fontsize=15, fontweight="bold", y=0.985)

    out = OUT_DIR / "fig7_hourly.png"
    save_fig(fig, out, also_pdf=True)
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Mean MAE  across 24 hours: {np.nanmean(mae):.4f} K  std: {np.nanstd(mae):.4f}")
    print(f"Mean RMSE across 24 hours: {np.nanmean(rmse):.4f} K")
    print(f"Diurnal amplitude — spatial: pred={amp_pred:.3f}K obs={amp_obs:.3f}K")
    print(f"Diurnal amplitude — point  : pred={amp_pp:.3f}K obs={amp_po:.3f}K")


if __name__ == "__main__":
    main()
