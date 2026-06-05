from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 180,
        "savefig.dpi": 300,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tsp50() -> None:
    epochs = [
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        800,
        850,
        900,
        950,
        970,
        990,
        1000,
    ]
    cado_gap = [
        4.771,
        4.627,
        4.364,
        4.789,
        4.503,
        4.943,
        4.649,
        4.988,
        4.561,
        4.328,
        4.737,
        4.339,
        4.599,
        4.362,
        4.505,
        4.545,
        4.948,
        4.489,
        4.813,
        3.955,
        3.959,
        4.031,
    ]
    sl_gap = 4.46

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(epochs, cado_gap, marker="o", linewidth=1.8, label="Exp. 5 CADO extended")
    ax.axhline(sl_gap, linestyle="--", color="tab:red", linewidth=1.3, label="Exp. 3 DIFUSCO SL")
    ax.scatter([970], [3.955], color="tab:green", zorder=3)
    ax.annotate(
        "best 3.955%",
        xy=(970, 3.955),
        xytext=(710, 4.05),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    ax.set_title("TSP-50: DIFUSCO SL vs. CADO Fine-Tuning")
    ax.set_xlabel("CADO epoch")
    ax.set_ylabel("Validation optimality gap (%)")
    ax.set_ylim(3.75, 5.35)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "paper_tsp50_gap")


def plot_cvrp50() -> None:
    epochs_sl = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    exp6_greedy = [29.98, 29.98, 31.17, 36.68, 32.74, 37.59, 39.72, 33.41, 33.04, 32.97, 33.66]
    exp7_2opt = [31.20, 27.39, 25.96, 27.36, 30.19, 30.82, 33.92, 34.75, 34.12, 35.09, 35.51]

    epochs_cado = [
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        470,
        500,
        550,
        600,
        650,
        700,
        750,
        800,
        850,
        900,
        950,
        1000,
    ]
    exp8_cado = [
        25.73,
        25.41,
        24.59,
        25.01,
        25.97,
        25.02,
        24.66,
        21.85,
        21.83,
        20.24,
        23.38,
        25.14,
        24.81,
        25.61,
        25.23,
        27.63,
        47.37,
        42.15,
        41.48,
        39.63,
        47.16,
    ]

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.plot(epochs_sl, exp6_greedy, marker="o", linewidth=1.7, label="Exp. 6 SL, greedy")
    ax.plot(epochs_sl, exp7_2opt, marker="s", linewidth=1.7, label="Exp. 7 SL, greedy + 2-opt")
    ax.plot(epochs_cado, exp8_cado, marker="^", linewidth=1.5, markersize=4, label="Exp. 8 CADO")
    ax.scatter([470], [20.24], color="tab:green", zorder=3)
    ax.annotate(
        "best CADO 20.24%",
        xy=(470, 20.24),
        xytext=(540, 21.2),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    ax.annotate("late collapse", xy=(800, 47.37), xytext=(655, 45.0), arrowprops={"arrowstyle": "->", "linewidth": 0.8})
    ax.set_title("CVRP-50: Decoder Choice and CADO Fine-Tuning")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation optimality gap (%)")
    ax.set_ylim(18, 50)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "paper_cvrp50_gap_curves")


def plot_best_final_summary() -> None:
    labels = [
        "TSP SL\nExp. 3",
        "TSP CADO\nExp. 5",
        "CVRP SL greedy\nExp. 6",
        "CVRP SL 2-opt\nExp. 7",
        "CVRP CADO\nExp. 8",
    ]
    best = [4.46, 3.955, 29.98, 25.96, 20.24]
    final = [4.46, 4.031, 33.66, 35.51, 47.16]

    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar([i - width / 2 for i in x], best, width=width, label="Best checkpoint")
    ax.bar([i + width / 2 for i in x], final, width=width, label="Final checkpoint")
    ax.set_title("Paper-Selected Experiments: Best vs. Final Gap")
    ax.set_ylabel("Validation optimality gap (%)")
    ax.set_xticks(list(x), labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "paper_best_final_gap_summary")


if __name__ == "__main__":
    plot_tsp50()
    plot_cvrp50()
    plot_best_final_summary()
    print(f"Saved paper result figures to {OUT_DIR}")
