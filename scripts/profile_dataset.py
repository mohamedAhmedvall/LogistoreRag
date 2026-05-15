"""Profilage du dataset Customer Support Tickets.

But : connaître la distribution réelle des champs (notamment numériques),
le taux d'unicité textuelle et les longueurs, pour adapter les validateurs
Pydantic et la stratégie d'ingestion.

Usage :
    python scripts/profile_dataset.py [--rows N]
    python scripts/profile_dataset.py --rows 200000 --save-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data/raw/customer_support_tickets.csv"
FIGURES_DIR = PROJECT_ROOT / "docs/figures"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=None, help="Échantillon (défaut: tout)")
    p.add_argument("--save-figures", action="store_true", help="Génère les PNG dans docs/figures/")
    return p.parse_args()


def _sep(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    args = _parse_args()
    df = pd.read_csv(CSV_PATH, nrows=args.rows)

    _sep("1. SHAPE & SCHEMA")
    print(f"Rows : {len(df):,}")
    print(f"Cols : {len(df.columns)}")
    print(df.dtypes.to_string())

    _sep("2. NaN par colonne")
    nulls = df.isna().sum()
    print(nulls[nulls > 0].to_string() or "(aucun)")

    _sep("3. Stats numériques (min / max / mean / std)")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    print(df[num_cols].describe().T[["min", "max", "mean", "std"]].round(2).to_string())

    _sep("4. Distributions des champs catégoriels")
    for col in [
        "priority",
        "status",
        "channel",
        "region",
        "language",
        "category",
        "product",
        "subscription_type",
        "customer_segment",
        "escalated",
        "sla_breached",
        "preferred_contact_time",
        "customer_gender",
    ]:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        print(f"\n[{col}] ({len(vc)} valeurs distinctes)")
        print(vc.head(15).to_string())

    _sep("5. Longueurs textuelles (issue_description, resolution_notes)")
    for col in ("issue_description", "resolution_notes"):
        s = df[col].fillna("").astype(str).str.len()
        print(
            f"\n[{col}] len min={s.min()} max={s.max()} median={s.median():.0f} mean={s.mean():.1f}"
        )
        print(s.quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())

    _sep("6. Unicité textuelle (clé de dédup = issue + resolution)")
    df["_iss"] = df["issue_description"].fillna("").astype(str).str.strip()
    df["_res"] = df["resolution_notes"].fillna("").astype(str).str.strip()
    df["_key"] = df["_iss"] + "\n" + df["_res"]
    n_total = len(df)
    n_unique_pairs = df["_key"].nunique()
    n_unique_issue = df["_iss"].nunique()
    n_unique_resolution = df["_res"].nunique()
    print(f"Lignes totales         : {n_total:,}")
    pct_iss = 100 * n_unique_issue / n_total
    pct_res = 100 * n_unique_resolution / n_total
    pct_pair = 100 * n_unique_pairs / n_total
    print(f"Issue uniques          : {n_unique_issue:,}  ({pct_iss:.1f}%)")
    print(f"Resolution uniques     : {n_unique_resolution:,}  ({pct_res:.1f}%)")
    print(f"Couples (iss,res) uniq.: {n_unique_pairs:,}  ({pct_pair:.1f}%)")
    dup_ratio = 1 - n_unique_pairs / n_total
    print(f"=> Taux de doublons    : {100*dup_ratio:.1f}%")

    _sep("7. Top 10 contenus les plus dupliqués")
    print(df["_key"].value_counts().head(10).to_string())

    _sep("8. Cohérence des dates")
    for col in ("ticket_created_date", "ticket_resolved_date"):
        d = pd.to_datetime(df[col], errors="coerce")
        print(f"[{col}] min={d.min()} max={d.max()} NaT={d.isna().sum()}")

    # Détection des champs hors-bornes du modèle Pydantic actuel
    _sep("9. Champs hors-bornes par rapport aux validateurs Ticket (avant ajustement)")
    constraints = {
        "customer_age": (0, 120),
        "customer_satisfaction_score": (1, 5),
        "issue_complexity_score": (1, 5),
    }
    for col, (lo, hi) in constraints.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        out = ((s < lo) | (s > hi)).sum()
        pct = 100 * out / len(s)
        rng = f"[{s.min()}..{s.max()}]"
        print(f"[{col}] in [{lo}..{hi}] -> outliers: {out} ({pct:.1f}%)  observed range={rng}")

    if args.save_figures:
        import matplotlib.pyplot as plt
        import seaborn as sns

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

        # 1) catégories
        fig, ax = plt.subplots(figsize=(8, 5))
        df["category"].value_counts().plot.barh(ax=ax)
        ax.set_title("Distribution des catégories de tickets")
        ax.set_xlabel("Nombre de tickets")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "categories.png", dpi=120)

        # 2) longueurs
        fig, ax = plt.subplots(figsize=(8, 5))
        df["issue_description"].fillna("").str.len().plot.hist(
            bins=60, ax=ax, alpha=0.6, label="issue"
        )
        df["resolution_notes"].fillna("").str.len().plot.hist(
            bins=60, ax=ax, alpha=0.6, label="resolution"
        )
        ax.legend()
        ax.set_title("Longueur des champs textuels (caractères)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "text_length.png", dpi=120)

        # 3) langues
        fig, ax = plt.subplots(figsize=(8, 5))
        df["language"].value_counts().plot.bar(ax=ax)
        ax.set_title("Langues")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "languages.png", dpi=120)

        # 4) complexity score
        fig, ax = plt.subplots(figsize=(8, 5))
        df["issue_complexity_score"].value_counts().sort_index().plot.bar(ax=ax)
        ax.set_title("Distribution du issue_complexity_score")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "complexity_score.png", dpi=120)

        print(f"\nFigures sauvées dans {FIGURES_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
