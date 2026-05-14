# optimizer/lineup_optimizer.py
"""
MLB Lineup Optimizer.

Two-stage pipeline:
  Stage 1 — Player Selection (ML Model)
    Gradient Boosting model scores each batter's expected run
    contribution against the specific opposing pitcher using
    16 Statcast features. Top 9 players selected by hybrid score
    (50% ML prediction + 50% statistical baseline).

  Stage 2 — Batting Order Optimization (Monte Carlo Simulation)
    Simulates 8,000 full 9-inning games per lineup evaluation.
    Each simulation draws discrete at-bat outcomes from each batter's
    empirical event distribution, adjusted for pitcher difficulty
    via the odds ratio method. Pairwise-swap local search finds
    a near-optimal batting order.

    Note: Lineup ordering effects in baseball are small (~0.05 runs/game).
    Minor slot variations between runs are expected and acceptable.

Markov chain usage:
    The 24 base-out state Markov chain and Bellman equation solver are
    retained for computing per-batter run expectancy values (V[0]),
    which inform the transition probability matrices used in simulation.
    They are not used directly for lineup ordering.

State encoding:
    State = (first_base, second_base, third_base, outs)
    Encoded as integer: outs * 8 + bases
    where bases = 4*first + 2*second + third (binary)
    States 0-23: active, State 24: inning over (terminal)
"""

import numpy as np
import pandas as pd
from itertools import permutations
from db.connection import get_connection, get_engine
from models.train_model import load_model, predict_lineup_scores


# ── State space ───────────────────────────────────────────────

N_STATES  = 25   # 24 base-out states + 1 terminal
TERMINAL  = 24   # index of "3 outs / inning over" state
N_INNINGS = 9

# League average event probabilities (2024-2026 era)
LEAGUE_AVG_EVENT_PROBS = {
    'single':                      0.150,
    'double':                      0.048,
    'triple':                      0.005,
    'home_run':                    0.036,
    'walk':                        0.085,
    'intent_walk':                 0.004,
    'hit_by_pitch':                0.011,
    'strikeout':                   0.225,
    'strikeout_double_play':       0.002,
    'field_out':                   0.225,
    'grounded_into_double_play':   0.030,
    'force_out':                   0.050,
    'fielders_choice':             0.015,
    'fielders_choice_out':         0.020,
    'sac_fly':                     0.008,
    'sac_fly_double_play':         0.001,
    'sac_bunt':                    0.003,
    'double_play':                 0.010,
    'field_error':                 0.008,
    'catcher_interf':              0.001,
    'truncated_pa':                0.000,
}


def encode_state(first: bool, second: bool, third: bool, outs: int) -> int:
    """
    Encodes a base-out state as an integer 0-23.
    Terminal state (3 outs) = 24.

    Examples:
        encode_state(0,0,0,0) = 0   (bases empty, 0 outs)
        encode_state(1,0,0,0) = 8   (runner on first, 0 outs)
        encode_state(1,1,1,2) = 23  (bases loaded, 2 outs)
    """
    if outs == 3:
        return TERMINAL
    bases = 4 * int(first) + 2 * int(second) + int(third)
    return outs * 8 + bases


def decode_state(state: int) -> tuple:
    """
    Decodes an integer state back to (first, second, third, outs).
    """
    if state == TERMINAL:
        return (False, False, False, 3)
    outs   = state // 8
    bases  = state %  8
    first  = bool(bases & 4)
    second = bool(bases & 2)
    third  = bool(bases & 1)
    return (first, second, third, outs)


# ── Transition model ──────────────────────────────────────────

def build_transition_matrix(event_probs: dict) -> tuple:
    """
    Builds the transition matrix T and run matrix R for one batter.

    T[s, s'] = probability of transitioning from state s to state s'
    R[s]     = expected runs scored on this at-bat from state s

    Args:
        event_probs: dict mapping event name → probability
                     e.g. {'single': 0.18, 'strikeout': 0.22, ...}

    Returns:
        T: (N_STATES, N_STATES) transition probability matrix
        R: (N_STATES,) expected immediate runs vector
    """
    T = np.zeros((N_STATES, N_STATES))
    R = np.zeros(N_STATES)

    for s in range(N_STATES - 1):  # exclude terminal
        f, sc, th, outs = decode_state(s)

        for event, prob in event_probs.items():
            if prob <= 0:
                continue

            # Compute next state and runs scored for this event
            new_f, new_sc, new_th = f, sc, th
            new_outs = outs
            runs = 0

            if event in ('strikeout', 'field_out', 'force_out',
                         'fielders_choice_out'):
                new_outs += 1

            elif event == 'strikeout_double_play':
                # Strikeout + caught/erased runner = two outs.
                # Most commonly this removes a runner from first.
                new_outs += 2
                if f:
                    new_f = False
                elif sc:
                    new_sc = False
                elif th:
                    new_th = False

            elif event == 'grounded_into_double_play':
                new_outs += 2
                # Runner on first is erased
                if f:
                    new_f = False
                # Run scores if third base occupied
                if th:
                    runs += 1
                    new_th = False

            elif event == 'double_play':
                # Generic double play: batter out + one lead runner out.
                new_outs += 2
                if f:
                    new_f = False
                elif sc:
                    new_sc = False
                elif th:
                    new_th = False

            elif event == 'fielders_choice':
                # Batter reaches first, lead runner out
                new_outs += 1
                new_f = True
                if th:
                    runs += 1
                    new_th = False

            elif event == 'single':
                # Standard advancement:
                # 3rd → home (scores), 2nd → 3rd, 1st → 2nd, batter → 1st
                runs   = int(th)
                new_th = sc
                new_sc = f
                new_f  = True

            elif event == 'double':
                # 3rd scores, 2nd scores, 1st → 3rd, batter → 2nd
                runs   = int(th) + int(sc)
                new_th = f
                new_sc = True
                new_f  = False

            elif event == 'triple':
                # Everyone scores
                runs   = int(f) + int(sc) + int(th)
                new_f  = False
                new_sc = False
                new_th = True

            elif event == 'home_run':
                # Everyone scores including batter
                runs   = 1 + int(f) + int(sc) + int(th)
                new_f  = False
                new_sc = False
                new_th = False

            elif event in ('walk', 'hit_by_pitch', 'intent_walk'):
                # Only force-advance runners when the base directly behind
                # them is occupied (actual baseball force rule).
                runs = 0
                # new_f/new_sc/new_th were initialized from current state
                if f and sc and th:
                    # Bases loaded — runner from 3rd scores, bases remain loaded
                    runs = 1
                    new_f, new_sc, new_th = True, True, True
                elif f and sc:
                    # Runners on 1st and 2nd: 2nd forced to 3rd, batter to 1st
                    new_th = True
                    new_sc = True
                    new_f  = True
                elif f:
                    # Runner on 1st only: forced to 2nd
                    new_sc = True
                    new_f  = True
                else:
                    # No force on existing runners; batter reaches first
                    new_f = True

            elif event == 'sac_fly':
                new_outs += 1
                if th:
                    runs  += 1
                    new_th = False

            elif event == 'sac_bunt':
                new_outs += 1
                # All runners advance one base
                runs   = int(th)
                new_th = sc
                new_sc = f
                new_f  = False

            elif event == 'field_error':
                # Treat like a single
                runs   = int(th)
                new_th = sc
                new_sc = f
                new_f  = True

            elif event == 'catcher_interf':
                # Batter reaches first, no outs
                new_f = True

            elif event == 'truncated_pa':
                # Incomplete PA — no state change
                pass

            elif event in ('sac_fly_double_play',):
                new_outs += 2
                if th:
                    runs  += 1
                    new_th = False

            elif event == 'triple_play':
                new_outs = 3

            else:
                # Unknown event — treat as out
                new_outs += 1
            # Clamp outs to 3 — any excess outs just end the inning
            new_outs = min(new_outs, 3)

            next_state = encode_state(new_f, new_sc, new_th, new_outs)
            T[s, next_state] += prob
            R[s]             += prob * runs

        # Normalize row to sum to 1 (handle floating point)
        row_sum = T[s].sum()
        if row_sum > 0:
            T[s] /= row_sum

    # Terminal state is absorbing
    T[TERMINAL, TERMINAL] = 1.0

    return T, R


# ── Run expectancy ────────────────────────────────────────────

def compute_run_expectancy(T: np.ndarray, R: np.ndarray,
                            max_iter: int = 1000,
                            tol: float = 1e-9) -> np.ndarray:
    """
    Solves for run expectancy E[R|s] using value iteration.

    Bellman equation:
        V(s) = R(s) + sum_s' T(s,s') * V(s')

    For the terminal state: V(TERMINAL) = 0

    Converges when max change < tol.

    Returns:
        V: (N_STATES,) array of expected runs from each state
    """
    V = np.zeros(N_STATES)

    for iteration in range(max_iter):
        V_new      = R + T @ V
        V_new[TERMINAL] = 0.0   # terminal always has 0 future value

        delta = np.max(np.abs(V_new - V))
        V     = V_new

        if delta < tol:
            break

    return V


# ── Event probability extraction ──────────────────────────────

def get_event_probs(batter_id: int) -> dict:
    """
    Computes empirical event probability distribution for a batter
    from their historical at-bats in our Statcast database.

    Falls back to league-average probabilities if insufficient data.
    """
    league_avg = LEAGUE_AVG_EVENT_PROBS.copy()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT event, COUNT(*) as cnt
        FROM at_bats
        WHERE batter_id = %s
          AND event IS NOT NULL
        GROUP BY event
    """, (batter_id,))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        return league_avg

    total = sum(r[1] for r in rows)

    # Require at least 100 PA for reliable estimates
    if total < 100:
        # Blend with league average (Bayesian shrinkage)
        # Weight: batter data gets total/(total+200) weight
        w_batter = total / (total + 200)
        w_league = 1 - w_batter
        probs = {}
        batter_counts = {r[0]: r[1] for r in rows}
        for event in league_avg:
            batter_rate = batter_counts.get(event, 0) / total
            probs[event] = w_batter * batter_rate + w_league * league_avg[event]
        return probs

    # Sufficient data — use empirical distribution
    batter_counts = {r[0]: r[1] for r in rows}
    probs = {}
    for event in league_avg:
        probs[event] = batter_counts.get(event, 0) / total

    # Normalize to sum to 1
    total_prob = sum(probs.values())
    if total_prob > 0:
        probs = {k: v / total_prob for k, v in probs.items()}

    return probs


def adjust_probs_for_pitcher(batter_probs: dict,
                              pitcher_id: int,
                              batter_id: int) -> dict:
    """
    Adjusts batter event probabilities based on pitcher difficulty.

    Uses the odds ratio method — a standard sabermetric technique
    for combining batter and pitcher tendencies:

        P(event | batter, pitcher) =
            (P_batter * P_pitcher) / P_league

    This preserves the relative probability ordering while
    accounting for pitcher effects.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Get pitcher event distribution
    cursor.execute("""
        SELECT event, COUNT(*) as cnt
        FROM at_bats
        WHERE pitcher_id = %s
          AND event IS NOT NULL
        GROUP BY event
    """, (pitcher_id,))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        return batter_probs

    total = sum(r[1] for r in rows)
    if total < 100:
        return batter_probs

    pitcher_counts = {r[0]: r[1] for r in rows}

    # League average probabilities
    league_avg = LEAGUE_AVG_EVENT_PROBS.copy()

    # Odds ratio adjustment with dampening
    # Cap adjustment factor to prevent extreme suppression/inflation
    # from small pitcher sample sizes
    adjusted = {}
    for event, p_batter in batter_probs.items():
        p_pitcher = pitcher_counts.get(event, 0) / total
        p_league  = league_avg.get(event, 0.01)

        if p_league > 0 and p_pitcher > 0:
            odds_batter  = p_batter  / (1 - p_batter  + 1e-9)
            odds_pitcher = p_pitcher / (1 - p_pitcher + 1e-9)
            odds_league  = p_league  / (1 - p_league  + 1e-9)
            odds_adj     = (odds_batter * odds_pitcher) / odds_league

            # Dampen the adjustment — blend 50/50 with batter's raw odds
            # This prevents small pitcher samples from dominating
            odds_dampened = 0.5 * odds_adj + 0.5 * odds_batter
            adjusted[event] = odds_dampened / (1 + odds_dampened)
        else:
            adjusted[event] = p_batter

    # Normalize
    total_prob = sum(adjusted.values())
    if total_prob > 0:
        adjusted = {k: v / total_prob for k, v in adjusted.items()}

    return adjusted


# ── Lineup simulation ─────────────────────────────────────────

def simulate_lineup(batting_order: list,
                    pitcher_id: int,
                    n_innings: int = N_INNINGS) -> float:
    batter_matrices = {}
    for batter_id in batting_order:
        raw_probs = get_event_probs(batter_id)
        adj_probs = adjust_probs_for_pitcher(raw_probs, pitcher_id, batter_id)
        T, R      = build_transition_matrix(adj_probs)
        V         = compute_run_expectancy(T, R)
        batter_matrices[batter_id] = (T, R, V)

    total_runs         = 0.0
    current_batter_idx = 0

    for inning in range(n_innings):
        state_dist = np.zeros(N_STATES)
        state_dist[encode_state(False, False, False, 0)] = 1.0

        for pa in range(30):
            if state_dist[TERMINAL] > 0.9999:
                break

            batter_id   = batting_order[current_batter_idx % 9]
            T, R, V     = batter_matrices[batter_id]
            inning_runs = float(state_dist[:TERMINAL] @ R[:TERMINAL])
            total_runs += inning_runs
            state_dist  = state_dist @ T
            current_batter_idx += 1

    return total_runs


# ── Optimal lineup finder ─────────────────────────────────────

def find_optimal_lineup(batter_ids: list,
                         pitcher_id: int,
                         pipeline,
                         n_select: int = 9,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Finds a near-optimal batting order from a roster of players.

    Stage 1 — Player Selection:
        Scores all roster players using the ML hybrid model and selects
        the top n_select by expected run contribution vs this pitcher.

    Stage 2 — Order Optimization:
        Runs Monte Carlo simulation (8,000 games per evaluation) with
        pairwise-swap local search. Accepts swaps that improve expected
        runs by >0.015 (above simulation noise floor).

    Note: Lineup ordering effects are small (~0.05 runs/game). Minor
    slot variations between runs are expected — player selection drives
    most of the value.

    Args:
        batter_ids:  list of MLBAM IDs (9-14 player roster)
        pitcher_id:  opposing pitcher MLBAM ID
        pipeline:    trained sklearn model pipeline
        n_select:    number of batters to select (default 9)
        verbose:     print progress to stdout

    Returns:
        DataFrame with columns: slot, batter_id, batter_name,
        hybrid_score, expected_runs
    """
    if verbose:
        print(f"Scoring {len(batter_ids)} batters vs pitcher {pitcher_id}...")

    # Step 1 — Score all batters with ML model
    scores = predict_lineup_scores(batter_ids, pitcher_id, pipeline)

    # Filter to batters that have feature data
    scores = scores.dropna(subset=['hybrid_score'])
    scores = scores.sort_values('hybrid_score', ascending=False)
    scores = scores.drop_duplicates(subset=['batter_id'], keep='first').reset_index(drop=True)

    if len(scores) < n_select:
        raise ValueError(
            f"Only {len(scores)} batters have sufficient data. "
            f"Need at least {n_select}."
        )

    # Select top n_select by hybrid score
    top_batters = scores.head(n_select)
    selected_ids = top_batters['batter_id'].tolist()

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Duplicate batter IDs were selected for the lineup.")

    if verbose:
        print(f"Selected top {n_select} batters:")
        for i, row in top_batters.iterrows():
            print(f"  {row['batter_name']:<25} hybrid_score={row['hybrid_score']:.4f}")
        print(f"Running local search over lineup orderings (near-optimal)...")

    # Step 2 — Precompute all batter matrices and event probs once
    if verbose:
        print("\nPrecomputing batter transition matrices...")

    batter_matrices = {}
    cached_events   = {}  # batter_id → list of event names
    cached_probs    = {}  # batter_id → numpy prob array

    for batter_id in selected_ids:
        raw_probs = get_event_probs(batter_id)
        adj_probs = adjust_probs_for_pitcher(raw_probs, pitcher_id, batter_id)
        T, R      = build_transition_matrix(adj_probs)
        V         = compute_run_expectancy(T, R)
        batter_matrices[batter_id] = (T, R, V)

        # Cache for Monte Carlo sampling — avoid DB hits during simulation
        events = list(adj_probs.keys())
        probs  = np.array(list(adj_probs.values()), dtype=np.float64)
        probs  = probs / probs.sum()
        cached_events[batter_id] = events
        cached_probs[batter_id]  = probs

    def simulate_fast(order, n_sims=8000):
        """
        Monte Carlo simulation of a 9-inning game.
        Results may vary slightly between runs due to sampling noise
        (~±0.1 runs at 8,000 simulations). This is a known property
        of Monte Carlo methods for lineup optimization — lineup ordering
        effects in baseball are small relative to simulation variance.
        """
        batter_events    = {b: cached_events[b] for b in order}
        batter_probs_arr = {b: cached_probs[b]  for b in order}

        total_runs = 0

        for sim in range(n_sims):
            runs       = 0
            batter_idx = 0

            for inning in range(N_INNINGS):
                on_1b = on_2b = on_3b = False
                outs  = 0

                while outs < 3:
                    batter_id = order[batter_idx % 9]
                    events    = batter_events[batter_id]
                    probs     = batter_probs_arr[batter_id]

                    event = events[np.searchsorted(
                        np.cumsum(probs),
                        np.random.random()
                    )]

                    if event in ('strikeout', 'field_out', 'force_out',
                                 'fielders_choice_out'):
                        outs += 1

                    elif event == 'strikeout_double_play':
                        outs += 2
                        if on_1b:   on_1b = False
                        elif on_2b: on_2b = False

                    elif event == 'grounded_into_double_play':
                        outs += 2
                        if on_1b: on_1b = False
                        if on_3b:
                            runs += 1
                            on_3b = False

                    elif event == 'double_play':
                        outs += 2
                        if on_1b:   on_1b = False
                        elif on_2b: on_2b = False

                    elif event == 'fielders_choice':
                        outs += 1
                        if on_3b:
                            runs += 1
                            on_3b = False
                        on_1b = True

                    elif event == 'single':
                        if on_3b: runs += 1
                        on_3b = on_2b
                        on_2b = on_1b
                        on_1b = True

                    elif event == 'double':
                        if on_3b: runs += 1
                        if on_2b: runs += 1
                        on_3b = on_1b
                        on_2b = True
                        on_1b = False

                    elif event == 'triple':
                        runs += int(on_1b) + int(on_2b) + int(on_3b)
                        on_1b = on_2b = False
                        on_3b = True

                    elif event == 'home_run':
                        runs += 1 + int(on_1b) + int(on_2b) + int(on_3b)
                        on_1b = on_2b = on_3b = False

                    elif event in ('walk', 'hit_by_pitch', 'intent_walk'):
                        if on_1b and on_2b and on_3b:
                            runs += 1
                        elif on_1b and on_2b:
                            on_3b = True
                        elif on_1b:
                            on_2b = True
                        on_1b = True

                    elif event == 'sac_fly':
                        outs += 1
                        if on_3b:
                            runs += 1
                            on_3b = False

                    elif event == 'sac_bunt':
                        outs += 1
                        if on_3b: runs += 1
                        on_3b = on_2b
                        on_2b = on_1b
                        on_1b = False

                    elif event == 'field_error':
                        if on_3b: runs += 1
                        on_3b = on_2b
                        on_2b = on_1b
                        on_1b = True

                    elif event == 'sac_fly_double_play':
                        outs += 2
                        if on_3b:
                            runs += 1
                            on_3b = False

                    elif event == 'catcher_interf':
                        on_1b = True

                    elif event == 'triple_play':
                        outs = 3

                    else:
                        outs += 1

                    outs = min(outs, 3)
                    batter_idx += 1

            total_runs += runs

        return total_runs / n_sims

    # Step 3 — Local search optimization
    if verbose:
        print("Running local search optimization...")

    sorted_ids = top_batters.sort_values(
        'hybrid_score', ascending=False
    )['batter_id'].tolist()

    best_order = sorted_ids.copy()
    best_runs  = simulate_fast(best_order)
    improved   = True
    iteration  = 0

    while improved:
        improved  = False
        iteration += 1
        for i in range(n_select):
            for j in range(i + 1, n_select):
                candidate      = best_order.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]
                candidate_runs = simulate_fast(candidate)

                if candidate_runs > best_runs + 0.015:
                    best_runs  = candidate_runs
                    best_order = candidate
                    improved   = True

        if verbose:
            print(f"  Iteration {iteration}: best = {best_runs:.4f} runs")

    if verbose:
        print(f"\n✅ Optimization complete in {iteration} iterations")
        print(f"   Expected runs: {best_runs:.3f}")

    # Step 4 — Build results DataFrame
    id_to_name  = dict(zip(top_batters['batter_id'], top_batters['batter_name']))
    id_to_score = dict(zip(top_batters['batter_id'], top_batters['hybrid_score']))

    results = pd.DataFrame({
        'slot':          range(1, n_select + 1),
        'batter_id':     best_order,
        'batter_name':   [id_to_name[b]  for b in best_order],
        'hybrid_score':  [id_to_score[b] for b in best_order],
        'expected_runs': best_runs
    })

    if results['batter_id'].duplicated().any():
        raise ValueError("Optimized lineup contains duplicate batter IDs.")

    return results


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from sqlalchemy import text

    engine = get_engine()

    player_query = text("""
        SELECT player_id, full_name
        FROM players
        WHERE full_name IN (
            'Judge, Aaron', 'Soto, Juan', 'Ohtani, Shohei',
            'Freeman, Freddie', 'Betts, Mookie', 'Trout, Mike',
            'Goldschmidt, Paul', 'Alvarez, Yordan', 'Seager, Corey',
            'Ramirez, Jose', 'Devers, Rafael', 'Turner, Trea',
            'Arenado, Nolan', 'Bogaerts, Xander'
        )
    """)

    pitcher_query = text("""
        SELECT player_id, full_name
        FROM players
        WHERE full_name ILIKE '%skenes%'
        LIMIT 1
    """)

    with engine.connect() as conn:
        players = pd.read_sql(player_query, conn)
        pitcher = pd.read_sql(pitcher_query, conn)

    print(f"Roster: {len(players)} players found")
    print(f"Pitcher: {pitcher.iloc[0]['full_name']}")

    pipeline   = load_model()
    batter_ids = players['player_id'].tolist()
    pitcher_id = int(pitcher.iloc[0]['player_id'])

    result = find_optimal_lineup(batter_ids, pitcher_id, pipeline)

    print("\n" + "="*50)
    print("  OPTIMAL BATTING ORDER")
    print("="*50)
    print(f"\n  Opposing pitcher: {pitcher.iloc[0]['full_name']}")
    print(f"  Expected runs:    {result['expected_runs'].iloc[0]:.3f}\n")
    print(f"  {'Slot':<6} {'Batter':<25} {'Hybrid Score':>12}")
    print(f"  {'-'*45}")
    for _, row in result.iterrows():
        print(f"  {int(row['slot']):<6} {row['batter_name']:<25} "
              f"{row['hybrid_score']:>12.4f}")