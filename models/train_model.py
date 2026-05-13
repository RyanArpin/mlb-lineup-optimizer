# models/train_model.py
"""
Hybrid statistical-ML model for predicting batter run contribution
against a specific opposing pitcher.

Approach:
  1. Load batter_pitcher_features from PostgreSQL
  2. Train a Gradient Boosting model to predict target_lw
  3. Evaluate ML model against the pure statistical baseline (target_lw)
  4. Save the trained model to disk for use by the optimizer
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from sql.run_features import load_features

# Where to save the trained model
MODEL_DIR  = os.path.join(os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "lineup_model.joblib")


# ── Feature definitions ───────────────────────────────────────

# These are the columns we pass to the model.
FEATURE_COLS = [
    # Batter overall — remove avg_lw (too close to target)
    "batter_k_rate",
    "batter_bb_rate",
    "batter_hr_rate",
    "batter_xwoba",
    "batter_exit_velo",
    "batter_launch_angle",
    "batter_hard_hit_rate",

    # Pitcher overall — remove avg_lw (too close to target)
    "pitcher_k_rate",
    "pitcher_bb_rate",
    "pitcher_hr_rate",
    "pitcher_xwoba",
    "pitcher_exit_velo",
    "pitcher_whiff_rate",

    # Matchup / interaction — remove split_avg_lw (directly encodes target)
    "platoon_advantage",
    "split_k_rate",
    "split_bb_rate",
]

TARGET_COL = "target_lw"


# ── Data preparation ──────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """
    Cleans and prepares the feature table for model training.

    Key decisions:
    - Drop rows where any feature is NULL (can't train on missing data)
    - For rows missing split features (batter never faced this pitcher's
      handedness), fill with batter's overall average — a reasonable
      fallback that preserves the row rather than dropping it
    """
    df = df.copy()

    # Fill missing split features with batter overall stats
    # (these are NULL when there's no handedness split data)
    df["split_avg_lw"]  = df["split_avg_lw"].fillna(df["batter_avg_lw"])
    df["split_k_rate"]  = df["split_k_rate"].fillna(df["batter_k_rate"])
    df["split_bb_rate"] = df["split_bb_rate"].fillna(df["batter_bb_rate"])

    # Drop rows still missing any feature or target
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    print(f"  Rows after cleaning: {len(df):,}")
    print(f"  Features:            {len(FEATURE_COLS)}")
    print(f"  Target range:        [{df[TARGET_COL].min():.4f}, "
          f"{df[TARGET_COL].max():.4f}]")
    print(f"  Target mean:         {df[TARGET_COL].mean():.4f}")

    return df


# ── Model training ────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Trains a Gradient Boosting model and evaluates it against
    the pure statistical baseline (target_lw used directly).

    Returns the trained pipeline and evaluation metrics.
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    baseline = df["batter_avg_lw"].values  # carry through the split

    # 80/20 train/test split — baseline carried alongside X so it
    # gets shuffled identically and corresponds to the correct test rows
    X_train, X_test, y_train, y_test, _, baseline_test = train_test_split(
        X, y, baseline, test_size=0.2, random_state=42
    )

    # ── Build the model pipeline ──────────────────────────────
    # StandardScaler normalizes features to zero mean, unit variance.
    # Gradient Boosting doesn't strictly require this, but it helps
    # with numerical stability and makes feature importances comparable.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=300,      # number of boosting stages
            learning_rate=0.05,    # shrinkage — lower = more robust
            max_depth=4,           # tree depth — controls complexity
            min_samples_leaf=20,   # prevents overfitting on small groups
            subsample=0.8,         # stochastic gradient boosting
            random_state=42
        ))
    ])

    print("\nTraining Gradient Boosting model...")
    pipeline.fit(X_train, y_train)

    # ── Evaluate ML model ─────────────────────────────────────
    y_pred_ml = pipeline.predict(X_test)
    ml_mae    = mean_absolute_error(y_test, y_pred_ml)
    ml_r2     = r2_score(y_test, y_pred_ml)

    # ── Evaluate statistical baseline ────────────────────────
    # baseline_test contains batter_avg_lw for the exact same rows
    # as y_test — correct alignment guaranteed by train_test_split
    baseline_pred = baseline_test
    baseline_mae  = mean_absolute_error(y_test, baseline_pred)
    baseline_r2   = r2_score(y_test, baseline_pred)

    # ── Cross validation ──────────────────────────────────────
    cv_scores = cross_val_score(
        pipeline, X, y,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    cv_mae = -cv_scores.mean()

    # ── Print results ─────────────────────────────────────────
    print("\n" + "="*50)
    print("  MODEL EVALUATION")
    print("="*50)
    print(f"\n  {'Metric':<25} {'Baseline':>10} {'ML Model':>10}")
    print(f"  {'-'*45}")
    print(f"  {'MAE (lower = better)':<25} {baseline_mae:>10.4f} {ml_mae:>10.4f}")
    print(f"  {'R² (higher = better)':<25} {baseline_r2:>10.4f} {ml_r2:>10.4f}")
    print(f"\n  5-Fold CV MAE: {cv_mae:.4f}")

    lift = (baseline_mae - ml_mae) / baseline_mae * 100
    print(f"\n  ML lift over baseline: {lift:+.1f}%")
    if lift > 0:
        print("  ✅ ML model improves over statistical baseline")
    else:
        print("  ⚠️  Baseline is competitive — hybrid weighting recommended")

    return pipeline, {
        "ml_mae":       ml_mae,
        "ml_r2":        ml_r2,
        "baseline_mae": baseline_mae,
        "baseline_r2":  baseline_r2,
        "cv_mae":       cv_mae,
        "lift_pct":     lift,
    }


# ── Feature importance ────────────────────────────────────────

def plot_feature_importance(pipeline, feature_cols):
    importances = pipeline.named_steps["model"].feature_importances_
    
    # Build a Series so we can filter by importance threshold
    importance_series = pd.Series(importances, index=feature_cols)
    
    # Remove features with near-zero importance (< 0.005)
    importance_series = importance_series[importance_series >= 0.005]
    importance_series = importance_series.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=importance_series.values,
        y=importance_series.index.tolist(),
        hue=importance_series.index.tolist(),  # fixes FutureWarning too
        palette="viridis",
        legend=False,
        ax=ax
    )
    ax.set_title("Feature Importances — Gradient Boosting Model")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    save_path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Feature importance plot saved to {save_path}")


# ── Prediction function ───────────────────────────────────────

def predict_lineup_scores(
    batter_ids: list,
    pitcher_id: int,
    pipeline,
    blend_weight: float = 0.5
) -> pd.DataFrame:
    """
    Predicts run contribution for each batter against a specific pitcher.

    Uses a hybrid score: weighted blend of ML prediction and target_lw
    baseline. blend_weight=0.5 means 50% ML, 50% statistical baseline.

    Args:
        batter_ids:    list of MLBAM batter IDs (your 14-player roster)
        pitcher_id:    MLBAM pitcher ID (opposing pitcher)
        pipeline:      trained sklearn pipeline
        blend_weight:  weight given to ML prediction (0=pure stats, 1=pure ML)

    Returns:
        DataFrame with batter_id, batter_name, ml_score, stat_score,
        hybrid_score — sorted by hybrid_score descending
    """
    from db.connection import get_engine
    engine = get_engine()

    # Pull matchup features for this specific roster vs pitcher
    query = """
        SELECT *
        FROM batter_pitcher_features
        WHERE batter_id = ANY(%(batter_ids)s)
          AND pitcher_id = %(pitcher_id)s
    """
    df = pd.read_sql(query, engine, params={
        "batter_ids": batter_ids,
        "pitcher_id": pitcher_id
    })

    if df.empty:
        # No direct matchup data — fall back to batter overall stats
        print("  ⚠️  No direct matchup data found, using batter overall stats")
        query_fallback = """
            SELECT
                bs.batter_id,
                bs.batter_name,
                bs.avg_linear_weight   AS batter_avg_lw,
                bs.k_rate              AS batter_k_rate,
                bs.bb_rate             AS batter_bb_rate,
                bs.hr_rate             AS batter_hr_rate,
                bs.avg_xwoba           AS batter_xwoba,
                bs.avg_exit_velo       AS batter_exit_velo,
                bs.avg_launch_angle    AS batter_launch_angle,
                bs.hard_hit_rate       AS batter_hard_hit_rate,
                ps.avg_lw_allowed      AS pitcher_avg_lw,
                ps.k_rate              AS pitcher_k_rate,
                ps.bb_rate             AS pitcher_bb_rate,
                ps.hr_rate_allowed     AS pitcher_hr_rate,
                ps.avg_xwoba_allowed   AS pitcher_xwoba,
                ps.avg_exit_velo_allowed AS pitcher_exit_velo,
                ps.whiff_rate          AS pitcher_whiff_rate,
                CASE WHEN bs.bats != ps.throws THEN 1 ELSE 0 END
                                       AS platoon_advantage,
                bs.avg_linear_weight   AS split_avg_lw,
                bs.k_rate              AS split_k_rate,
                bs.bb_rate             AS split_bb_rate,
                bs.avg_linear_weight   AS target_lw
            FROM batter_stats bs
            CROSS JOIN pitcher_stats ps
            WHERE bs.batter_id = ANY(%(batter_ids)s)
              AND ps.pitcher_id = %(pitcher_id)s
        """
        df = pd.read_sql(query_fallback, engine, params={
            "batter_ids": batter_ids,
            "pitcher_id": pitcher_id
        })

    if df.empty:
        raise ValueError(
            f"No data found for pitcher_id={pitcher_id}. "
            "Check that the pitcher exists in pitcher_stats."
        )

    # Fill missing split features
    df["split_avg_lw"]  = df["split_avg_lw"].fillna(df["batter_avg_lw"])
    df["split_k_rate"]  = df["split_k_rate"].fillna(df["batter_k_rate"])
    df["split_bb_rate"] = df["split_bb_rate"].fillna(df["batter_bb_rate"])

    df = df.dropna(subset=FEATURE_COLS)

    # ML prediction
    X = df[FEATURE_COLS].values
    ml_scores   = pipeline.predict(X)

    # Statistical baseline
    stat_scores = df["target_lw"].values

    # Hybrid blend
    hybrid_scores = (
        blend_weight * ml_scores +
        (1 - blend_weight) * stat_scores
    )

    results = pd.DataFrame({
        "batter_id":     df["batter_id"].values,
        "batter_name":   df["batter_name"].values,
        "ml_score":      ml_scores,
        "stat_score":    stat_scores,
        "hybrid_score":  hybrid_scores,
    }).sort_values("hybrid_score", ascending=False).reset_index(drop=True)

    return results


# ── Save / load model ─────────────────────────────────────────

def save_model(pipeline):
    joblib.dump(pipeline, MODEL_PATH)
    print(f"  ✅ Model saved to {MODEL_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH}. Run train_model.py first."
        )
    pipeline = joblib.load(MODEL_PATH)
    print(f"  ✅ Model loaded from {MODEL_PATH}")
    return pipeline


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("  MLB LINEUP OPTIMIZER — MODEL TRAINING")
    print("="*50)

    # Load features
    print("\n[1/4] Loading features...")
    df = load_features()

    # Prepare data
    print("\n[2/4] Preparing data...")
    df = prepare_data(df)

    # Train and evaluate
    print("\n[3/4] Training model...")
    pipeline, metrics = train_model(df)

    # Feature importance
    print("\n[4/4] Plotting feature importance...")
    plot_feature_importance(pipeline, FEATURE_COLS)

    # Save model
    save_model(pipeline)

    print("\n" + "="*50)
    print("  TRAINING COMPLETE")
    print("="*50)