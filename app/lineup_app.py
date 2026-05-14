# app/lineup_app.py
"""
MLB Lineup Optimizer — Streamlit Web App

Users select a roster and opposing pitcher from dropdowns.
The app runs the full pipeline:
  1. ML model scores each batter vs the pitcher
  2. Top 9 players selected by hybrid score
  3. Monte Carlo simulation optimizes the batting order
  4. Results displayed as an interactive table with expected runs
"""

import sys
import os

# Add project root to Python path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sqlalchemy import text

from db.connection import get_engine
from models.train_model import load_model, predict_lineup_scores
from optimizer.lineup_optimizer import find_optimal_lineup

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Lineup Optimizer",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e94560;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #aaa;
        margin-top: 0.3rem;
    }
    .slot-badge {
        background: #e94560;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
    }
    div[data-testid="stSidebar"] {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached data loaders ───────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Load the trained ML model — cached so it only loads once."""
    return load_model()


@st.cache_data(ttl=3600)
def load_batters():
    """
    Load all qualified batters from batter_stats.
    Cached for 1 hour — refreshes if new data is loaded.
    Returns a DataFrame with player_id and full_name.
    """
    engine = get_engine()
    query  = text("""
        SELECT bs.batter_id AS player_id, bs.batter_name AS full_name
        FROM batter_stats bs
        WHERE bs.pa >= 100
          AND bs.batter_id NOT IN (
              SELECT pitcher_id FROM pitcher_stats
              WHERE batters_faced >= 200
          )
        ORDER BY bs.avg_xwoba DESC NULLS LAST
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df = df.drop_duplicates(subset='player_id').reset_index(drop=True)
    df['player_label'] = df.apply(
        lambda row: f"{row['full_name']} ({int(row['player_id'])})",
        axis=1
    )
    return df


@st.cache_data(ttl=3600)
def load_pitchers():
    """
    Load all qualified pitchers from pitcher_stats.
    Cached for 1 hour.
    """
    engine = get_engine()
    query  = text("""
        SELECT ps.pitcher_id AS player_id, ps.pitcher_name AS full_name,
               ps.throws, ps.batters_faced
        FROM pitcher_stats ps
        WHERE ps.batters_faced >= 100
        ORDER BY ps.avg_xwoba_allowed ASC NULLS LAST
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df = df.drop_duplicates(subset='player_id').reset_index(drop=True)
    df['player_label'] = df.apply(
        lambda row: f"{row['full_name']} ({int(row['player_id'])})",
        axis=1
    )
    return df


# ── Helper functions ──────────────────────────────────────────

def plot_lineup_scores(result_df: pd.DataFrame,
                       pitcher_name: str) -> plt.Figure:
    """
    Creates a horizontal bar chart of hybrid scores by batting slot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.RdYlGn(
        np.linspace(0.3, 0.9, len(result_df))
    )[::-1]

    bars = ax.barh(
        [f"#{int(r['slot'])} {r['batter_name']}" for _, r in result_df.iterrows()],
        result_df['hybrid_score'],
        color=colors,
        edgecolor='white',
        linewidth=0.5
    )

    # Add value labels
    for bar, (_, row) in zip(bars, result_df.iterrows()):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{row['hybrid_score']:.3f}",
            va='center', ha='left',
            fontsize=9, color='#333'
        )

    ax.set_xlabel("Hybrid Score (Expected Run Contribution per PA)", fontsize=10)
    ax.set_title(f"Batting Order vs {pitcher_name}", fontsize=12, fontweight='bold')
    ax.set_xlim(0, result_df['hybrid_score'].max() * 1.2)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def plot_score_comparison(result_df: pd.DataFrame) -> plt.Figure:
    """
    Compares ML score vs statistical score for each batter.
    Shows where the ML model adds value over pure statistics.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x      = np.arange(len(result_df))
    width  = 0.35
    names  = [f"#{int(r['slot'])} {r['batter_name'].split(',')[0]}"
              for _, r in result_df.iterrows()]

    ax.bar(x - width/2, result_df['ml_score'],  width,
           label='ML Score',   color='#e94560', alpha=0.85)
    ax.bar(x + width/2, result_df['stat_score'], width,
           label='Stat Score', color='#0f3460', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Expected Run Contribution per PA")
    ax.set_title("ML Score vs Statistical Score by Batting Slot",
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


# ── Main app ──────────────────────────────────────────────────

def main():
    # Header
    st.markdown('<p class="main-header">⚾ MLB Lineup Optimizer</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Monte Carlo + Gradient Boosting lineup '
        'optimization using 2024-2026 Statcast data</p>',
        unsafe_allow_html=True
    )

    # Load data
    pipeline  = load_pipeline()
    batters   = load_batters()
    pitchers  = load_pitchers()

    if batters.empty or pitchers.empty:
        st.error("No player data found. Make sure the database is populated.")
        return

    batter_ids    = batters['player_id'].tolist()
    pitcher_ids   = pitchers['player_id'].tolist()
    batter_labels  = dict(zip(batters['player_id'], batters['player_label']))
    pitcher_labels = dict(zip(pitchers['player_id'], pitchers['player_label']))

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Lineup Configuration")
        st.markdown("---")

        # Pitcher selection
        st.subheader("Opposing Pitcher")
        pitcher_name = st.selectbox(
            "Select pitcher",
            options=pitcher_ids,
            index=0,
            format_func=lambda pid: pitcher_labels[pid],
            help="Pitchers sorted by toughness (lowest xwOBA allowed first)"
        )

        st.markdown("---")

        # Roster selection
        st.subheader("Your Roster (select 9-14 players)")
        st.caption("Players sorted by xwOBA — top of list = best hitters")

        selected_players = st.multiselect(
            "Select 9-14 players",
            options=batter_ids,
            default=None,
            format_func=lambda pid: batter_labels[pid],
            help="The optimizer will select the best 9 from your roster"
        )

        # Validation
        n_selected = len(selected_players)
        if n_selected == 0:
            st.info("Search and select players above")
        elif n_selected < 9:
            st.error(f"Need at least 9 players ({n_selected} selected)")
        elif n_selected > 14:
            st.error(f"Maximum 14 players ({n_selected} selected)")
        else:
            st.success(f"✅ Roster ready ({n_selected} players)")

        # Run button
        run_button = st.button(
            "🔄 Optimize Lineup",
            type="primary",
            disabled=(n_selected < 9 or n_selected > 14),
            width='stretch'
        )

        st.markdown("---")
        st.caption(
            "**How it works:**\n\n"
            "1. ML model scores each batter vs pitcher\n"
            "2. Top 9 selected by hybrid score\n"
            "3. Monte Carlo simulation optimizes batting order\n\n"
            "Runtime: ~60-90 seconds"
        )

    # ── Main panel ────────────────────────────────────────────
    if not run_button:
        # Show instructions when app first loads
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Step 1**\n\nSelect an opposing pitcher from the sidebar")
        with col2:
            st.info("**Step 2**\n\nChoose 9-14 players for your roster")
        with col3:
            st.info("**Step 3**\n\nClick 'Optimize Lineup' to run the model")

        st.markdown("---")
        st.subheader("About This Tool")
        st.markdown("""
        This optimizer uses a two-stage approach:

        **Stage 1 — Gradient Boosting Model**
        Predicts each batter's run contribution against the specific
        opposing pitcher using 16 Statcast features including
        handedness-specific xwOBA, exit velocity, HR rate, and whiff rate.
        Trained on 435,000 batter-pitcher matchups from 2024-2026.

        **Stage 2 — Monte Carlo Lineup Optimizer**
        Simulates 8,000 full 9-inning games per lineup evaluation.
        Each simulation draws discrete at-bat outcomes from each batter's
        empirical event distribution, adjusted for pitcher difficulty
        via the odds ratio method. Pairwise-swap local search finds
        the near-optimal batting order.

        **Note on variance:** Lineup ordering effects in baseball are
        small. Minor slot variations between runs
        are expected — player selection drives most of the value.
        """)
        return

    # ── Run optimization ──────────────────────────────────────
    # Get IDs for selected players and pitcher
    pitcher_row = pitchers[pitchers['player_id'] == pitcher_name].iloc[0]
    pitcher_id  = int(pitcher_row['player_id'])

    roster_df   = batters[batters['player_id'].isin(selected_players)].drop_duplicates('player_id')
    batter_ids  = roster_df['player_id'].tolist()

    # Progress display
    progress_bar = st.progress(0, text="Starting optimization...")
    status       = st.empty()

    try:
        # Stage 1 — ML scoring
        status.info("🤖 Stage 1/2: Scoring batters with ML model...")
        progress_bar.progress(20, text="Scoring batters...")
        time.sleep(0.3)

        all_scores = predict_lineup_scores(batter_ids, pitcher_id, pipeline)

        progress_bar.progress(50, text="Running Monte Carlo optimizer...")
        status.info("📊 Stage 2/2: Running Monte Carlo simulation...")

        # Stage 2 — Optimization
        result = find_optimal_lineup(
            batter_ids, pitcher_id, pipeline,
            verbose=False
        )

        progress_bar.progress(100, text="Complete!")
        status.empty()
        progress_bar.empty()

    except ValueError as e:
        progress_bar.empty()
        status.error(f"❌ Optimization failed: {e}")
        return
    except Exception as e:
        progress_bar.empty()
        status.error(f"❌ Unexpected error: {e}")
        st.exception(e)
        return

    # ── Display results ───────────────────────────────────────
    st.success(f"✅ Optimal lineup found vs {pitcher_row['full_name']}")
    st.markdown("---")

    # Top metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Expected Runs (9 innings)",
            value=f"{result['expected_runs'].iloc[0]:.1f}",
            help="Monte Carlo simulation (8,000 games). "
                 "Results may vary slightly between runs (±0.05) — "
                 "this reflects genuine uncertainty in lineup ordering."
        )
    best_batter = result.loc[result['hybrid_score'].idxmax(), 'batter_name']
    with col2:
        st.metric(
            label="Best Batter",
            value=best_batter.split(',')[0],
            help="Highest hybrid score in the lineup"
        )
    with col3:
        st.metric(
            label="Opposing Pitcher",
            value=pitcher_row['full_name'].split(',')[0],
            help=f"Batters faced: {int(pitcher_row['batters_faced']):,}"
        )

    st.markdown("---")

    # Results in two columns
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🏆 Optimal Batting Order")

        # Build styled table
        display_df = result[['slot', 'batter_name', 'hybrid_score']].copy()
        display_df.columns = ['Slot', 'Batter', 'Hybrid Score']
        display_df['Slot'] = display_df['Slot'].astype(int)
        display_df['Hybrid Score'] = display_df['Hybrid Score'].round(4)

        st.dataframe(
            display_df,
            hide_index=True,
            width='stretch',
            column_config={
                "Slot": st.column_config.NumberColumn(
                    "Slot", width="small"
                ),
                "Batter": st.column_config.TextColumn(
                    "Batter", width="medium"
                ),
                "Hybrid Score": st.column_config.ProgressColumn(
                    "Hybrid Score",
                    min_value=0,
                    max_value=float(display_df['Hybrid Score'].max() * 1.2),
                    format="%.4f"
                )
            }
        )

    with col_right:
        st.subheader("📊 Score Breakdown")
        # Merge with result to get slot ordering
        merged = result.merge(
            all_scores[['batter_id', 'ml_score', 'stat_score']],
            on='batter_id'
        )
        fig = plot_score_comparison(merged)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Full width chart
    st.subheader("📈 Lineup Hybrid Scores by Batting Slot")
    fig2 = plot_lineup_scores(result, pitcher_row['full_name'])
    st.pyplot(fig2)
    plt.close()

    # Raw data expander
    with st.expander("🔍 View Raw Data"):
        st.subheader("Selected Lineup Details")
        full_display = result.copy()
        full_display['slot'] = full_display['slot'].astype(int)
        full_display['hybrid_score'] = full_display['hybrid_score'].round(4)
        full_display['expected_runs'] = full_display['expected_runs'].round(3)
        st.dataframe(full_display, hide_index=True, width='stretch')

        st.subheader("All Roster Scores")
        all_scores['ml_score']    = all_scores['ml_score'].round(4)
        all_scores['stat_score']  = all_scores['stat_score'].round(4)
        all_scores['hybrid_score'] = all_scores['hybrid_score'].round(4)
        st.dataframe(
            all_scores[['batter_name', 'ml_score',
                        'stat_score', 'hybrid_score']],
            hide_index=True,
            width='stretch'
        )


if __name__ == "__main__":
    main()