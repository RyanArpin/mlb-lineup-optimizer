-- =============================================================
-- MLB Lineup Optimizer — Feature Engineering
-- =============================================================
-- Produces one flat feature table: batter_pitcher_features
-- Each row = one (batter_id, pitcher_id) matchup
-- with aggregated statistics from all their historical at-bats.
--
-- Run order matters — each step builds on the previous one:
--   1. linear_weights       (run value per outcome)
--   2. batter_stats         (batter overall profile)
--   3. pitcher_stats        (pitcher overall profile)
--   4. matchup_stats        (batter vs pitcher directly)
--   5. batter_pitcher_features (final joined feature table)
-- =============================================================


-- -------------------------------------------------------------
-- STEP 4A: Linear Weights
-- -------------------------------------------------------------
DROP TABLE IF EXISTS linear_weights;
CREATE TABLE linear_weights AS
SELECT
    event,
    AVG(lw) AS weight
FROM (
    SELECT
        event,
        CASE event
            WHEN 'home_run'                  THEN  1.397
            WHEN 'triple'                    THEN  1.070
            WHEN 'double'                    THEN  0.776
            WHEN 'single'                    THEN  0.474
            WHEN 'walk'                      THEN  0.323
            WHEN 'hit_by_pitch'              THEN  0.352
            WHEN 'intent_walk'               THEN  0.179
            WHEN 'strikeout'                 THEN -0.274
            WHEN 'field_out'                 THEN -0.270
            WHEN 'grounded_into_double_play' THEN -0.502
            WHEN 'force_out'                 THEN -0.270
            WHEN 'sac_fly'                   THEN  0.215
            WHEN 'sac_bunt'                  THEN -0.106
            WHEN 'double_play'               THEN -0.502
            WHEN 'field_error'               THEN  0.215
            WHEN 'fielders_choice'           THEN -0.270
            WHEN 'fielders_choice_out'       THEN -0.270
            WHEN 'caught_stealing_2b'        THEN -0.467
            WHEN 'caught_stealing_3b'        THEN -0.467
            WHEN 'caught_stealing_home'      THEN -0.467
            ELSE 0.0
        END AS lw
    FROM at_bats
    WHERE event IS NOT NULL
) subq
GROUP BY event;

-- Fill in linear weights on at_bats table
UPDATE at_bats ab
SET linear_weight = lw.weight
FROM linear_weights lw
WHERE ab.event = lw.event;


-- -------------------------------------------------------------
-- STEP 4B: Batter Overall Statistics
-- -------------------------------------------------------------
-- Now includes handedness-specific xwOBA (xwoba_vs_R, xwoba_vs_L)
-- so the model sees the correct quality metric for each matchup
-- rather than a career average that obscures platoon effects.
-- -------------------------------------------------------------
DROP TABLE IF EXISTS batter_stats;
CREATE TABLE batter_stats AS

WITH pitch_aggs AS (
    SELECT
        batter_id,
        AVG(estimated_woba_using_speedangle)                AS avg_xwoba,
        AVG(estimated_ba_using_speedangle)                  AS avg_xba,
        AVG(CASE WHEN launch_speed IS NOT NULL
            THEN launch_speed END)                          AS avg_exit_velo,
        AVG(CASE WHEN launch_angle IS NOT NULL
            THEN launch_angle END)                          AS avg_launch_angle,
        SUM(CASE WHEN launch_speed >= 95
            THEN 1.0 ELSE 0 END) /
        NULLIF(SUM(CASE WHEN launch_speed IS NOT NULL
            THEN 1 ELSE 0 END), 0)                          AS hard_hit_rate,

        -- Handedness-specific xwOBA
        -- These capture platoon effects properly:
        -- a R-handed batter's xwoba_vs_L will be higher than xwoba_vs_R
        -- if he has a platoon advantage, and the model will learn this
        AVG(CASE WHEN p_throws = 'R'
            AND estimated_woba_using_speedangle IS NOT NULL
            THEN estimated_woba_using_speedangle END)       AS xwoba_vs_R,
        AVG(CASE WHEN p_throws = 'L'
            AND estimated_woba_using_speedangle IS NOT NULL
            THEN estimated_woba_using_speedangle END)       AS xwoba_vs_L
    FROM pitches
    GROUP BY batter_id
)
SELECT
    ab.batter_id,
    p.full_name                                             AS batter_name,
    p.bats,
    COUNT(*)                                                AS pa,
    SUM(CASE WHEN ab.event IN (
        'strikeout','strikeout_double_play')
        THEN 1 ELSE 0 END)                                  AS k_count,
    SUM(CASE WHEN ab.event IN (
        'walk','intent_walk')
        THEN 1 ELSE 0 END)                                  AS bb_count,
    SUM(CASE WHEN ab.event = 'home_run'
        THEN 1 ELSE 0 END)                                  AS hr_count,
    ROUND(AVG(ab.linear_weight)::NUMERIC, 4)                AS avg_linear_weight,
    ROUND((SUM(CASE WHEN ab.event IN (
        'strikeout','strikeout_double_play')
        THEN 1.0 ELSE 0 END) / COUNT(*))::NUMERIC, 4)      AS k_rate,
    ROUND((SUM(CASE WHEN ab.event IN (
        'walk','intent_walk')
        THEN 1.0 ELSE 0 END) / COUNT(*))::NUMERIC, 4)      AS bb_rate,
    ROUND((SUM(CASE WHEN ab.event = 'home_run'
        THEN 1.0 ELSE 0 END) / COUNT(*))::NUMERIC, 4)      AS hr_rate,
    ROUND(pa.avg_xwoba::NUMERIC, 4)                         AS avg_xwoba,
    ROUND(pa.avg_xba::NUMERIC, 4)                           AS avg_xba,
    ROUND(pa.avg_exit_velo::NUMERIC, 4)                     AS avg_exit_velo,
    ROUND(pa.avg_launch_angle::NUMERIC, 4)                  AS avg_launch_angle,
    ROUND(pa.hard_hit_rate::NUMERIC, 4)                     AS hard_hit_rate,
    ROUND(pa.xwoba_vs_R::NUMERIC, 4)                        AS xwoba_vs_R,
    ROUND(pa.xwoba_vs_L::NUMERIC, 4)                        AS xwoba_vs_L

FROM at_bats ab
JOIN players p          ON ab.batter_id  = p.player_id
LEFT JOIN pitch_aggs pa ON ab.batter_id  = pa.batter_id
WHERE ab.event IS NOT NULL
GROUP BY ab.batter_id, p.full_name, p.bats,
         pa.avg_xwoba, pa.avg_xba, pa.avg_exit_velo,
         pa.avg_launch_angle, pa.hard_hit_rate,
         pa.xwoba_vs_R, pa.xwoba_vs_L
HAVING COUNT(*) >= 50;


-- -------------------------------------------------------------
-- STEP 4C: Pitcher Overall Statistics
-- -------------------------------------------------------------
-- Now includes handedness-specific xwOBA allowed
-- (xwoba_vs_R, xwoba_vs_L from the pitcher's perspective)
-- so we know how tough this pitcher is against each batter hand.
-- -------------------------------------------------------------
DROP TABLE IF EXISTS pitcher_stats;
CREATE TABLE pitcher_stats AS

WITH pitch_aggs AS (
    SELECT
        pitcher_id,
        AVG(estimated_woba_using_speedangle)                AS avg_xwoba_allowed,
        AVG(CASE WHEN launch_speed IS NOT NULL
            THEN launch_speed END)                          AS avg_exit_velo_allowed,
        SUM(CASE WHEN description = 'swinging_strike'
            THEN 1.0 ELSE 0 END) /
        NULLIF(COUNT(*), 0)                                 AS whiff_rate,

        -- How tough is this pitcher against R-handed batters?
        AVG(CASE WHEN stand = 'R'
            AND estimated_woba_using_speedangle IS NOT NULL
            THEN estimated_woba_using_speedangle END)       AS xwoba_vs_R,

        -- How tough is this pitcher against L-handed batters?
        AVG(CASE WHEN stand = 'L'
            AND estimated_woba_using_speedangle IS NOT NULL
            THEN estimated_woba_using_speedangle END)       AS xwoba_vs_L
    FROM pitches
    GROUP BY pitcher_id
)
SELECT
    ab.pitcher_id,
    p.full_name                                             AS pitcher_name,
    p.throws,
    COUNT(*)                                                AS batters_faced,
    ROUND(AVG(ab.linear_weight)::NUMERIC, 4)                AS avg_lw_allowed,
    ROUND((SUM(CASE WHEN ab.event IN (
        'strikeout','strikeout_double_play')
        THEN 1.0 ELSE 0 END) / COUNT(*))::NUMERIC, 4)      AS k_rate,
    ROUND((SUM(CASE WHEN ab.event IN (
        'walk','intent_walk')
        THEN 1.0 ELSE 0 END) / COUNT(*))::NUMERIC, 4)      AS bb_rate,
    ROUND((SUM(CASE WHEN ab.event = 'home_run'
        THEN 1.0 ELSE 0 END) / COUNT(*))::NUMERIC, 4)      AS hr_rate_allowed,
    ROUND(pa.avg_xwoba_allowed::NUMERIC, 4)                 AS avg_xwoba_allowed,
    ROUND(pa.avg_exit_velo_allowed::NUMERIC, 4)             AS avg_exit_velo_allowed,
    ROUND(pa.whiff_rate::NUMERIC, 4)                        AS whiff_rate,
    ROUND(pa.xwoba_vs_R::NUMERIC, 4)                        AS xwoba_vs_R,
    ROUND(pa.xwoba_vs_L::NUMERIC, 4)                        AS xwoba_vs_L

FROM at_bats ab
JOIN players p        ON ab.pitcher_id = p.player_id
LEFT JOIN pitch_aggs pa ON ab.pitcher_id = pa.pitcher_id
WHERE ab.event IS NOT NULL
GROUP BY ab.pitcher_id, p.full_name, p.throws,
         pa.avg_xwoba_allowed, pa.avg_exit_velo_allowed,
         pa.whiff_rate, pa.xwoba_vs_R, pa.xwoba_vs_L
HAVING COUNT(*) >= 50;


-- -------------------------------------------------------------
-- STEP 4D: Batter vs Pitcher Handedness Splits
-- -------------------------------------------------------------
DROP TABLE IF EXISTS handedness_splits;
CREATE TABLE handedness_splits AS
SELECT
    ab.batter_id,
    pit.stand                                               AS batter_hand,
    pit.p_throws                                            AS pitcher_hand,
    COUNT(*)                                                AS pa,
    ROUND(AVG(ab.linear_weight)::NUMERIC, 4)                AS avg_lw,
    ROUND(AVG(pit.estimated_woba_using_speedangle)
        ::NUMERIC, 4)                                       AS avg_xwoba,
    ROUND((SUM(CASE WHEN ab.event IN (
        'strikeout','strikeout_double_play')
        THEN 1.0 ELSE 0 END) / COUNT(*))::NUMERIC, 4)      AS k_rate,
    ROUND((SUM(CASE WHEN ab.event IN (
        'walk','intent_walk')
        THEN 1.0 ELSE 0 END) / COUNT(*))::NUMERIC, 4)      AS bb_rate
FROM at_bats ab
JOIN pitches pit ON ab.at_bat_id = pit.at_bat_id
WHERE ab.event IS NOT NULL
  AND pit.stand IS NOT NULL
  AND pit.p_throws IS NOT NULL
GROUP BY ab.batter_id, pit.stand, pit.p_throws
HAVING COUNT(*) >= 20;


-- -------------------------------------------------------------
-- STEP 4E: Final Feature Table
-- -------------------------------------------------------------
-- Key change: batter_xwoba and pitcher_xwoba are now
-- handedness-specific — the correct metric for this matchup.
-- This is what makes platoon effects show up properly in the model.
-- -------------------------------------------------------------
DROP TABLE IF EXISTS batter_pitcher_features;
CREATE TABLE batter_pitcher_features AS
SELECT
    -- Identity
    bs.batter_id,
    bs.batter_name,
    ps.pitcher_id,
    ps.pitcher_name,
    bs.bats,
    ps.throws,

    -- ── Batter overall features ──────────────────────────────
    bs.pa                                                   AS batter_pa,
    bs.avg_linear_weight                                    AS batter_avg_lw,
    bs.k_rate                                               AS batter_k_rate,
    bs.bb_rate                                              AS batter_bb_rate,
    bs.hr_rate                                              AS batter_hr_rate,

    -- Handedness-specific xwOBA: use the correct split
    -- based on the opposing pitcher's throwing arm
    COALESCE(
        CASE
            WHEN ps.throws = 'R' THEN bs.xwoba_vs_R
            WHEN ps.throws = 'L' THEN bs.xwoba_vs_L
        END,
        bs.avg_xwoba                                        -- fallback
    )                                                       AS batter_xwoba,

    bs.avg_exit_velo                                        AS batter_exit_velo,
    bs.avg_launch_angle                                     AS batter_launch_angle,
    bs.hard_hit_rate                                        AS batter_hard_hit_rate,

    -- ── Pitcher overall features ─────────────────────────────
    ps.batters_faced                                        AS pitcher_bf,
    ps.avg_lw_allowed                                       AS pitcher_avg_lw,
    ps.k_rate                                               AS pitcher_k_rate,
    ps.bb_rate                                              AS pitcher_bb_rate,
    ps.hr_rate_allowed                                      AS pitcher_hr_rate,

    -- Handedness-specific xwOBA allowed: how tough is this pitcher
    -- against THIS batter's handedness specifically
    COALESCE(
        CASE
            WHEN bs.bats = 'R' THEN ps.xwoba_vs_R
            WHEN bs.bats = 'L' THEN ps.xwoba_vs_L
        END,
        ps.avg_xwoba_allowed                                -- fallback
    )                                                       AS pitcher_xwoba,

    ps.avg_exit_velo_allowed                                AS pitcher_exit_velo,
    ps.whiff_rate                                           AS pitcher_whiff_rate,

    -- ── Handedness split features ────────────────────────────
    hs.avg_lw                                               AS split_avg_lw,
    hs.avg_xwoba                                            AS split_xwoba,
    hs.k_rate                                               AS split_k_rate,
    hs.bb_rate                                              AS split_bb_rate,
    hs.pa                                                   AS split_pa,

    -- ── Platoon indicator ────────────────────────────────────
    CASE
        WHEN bs.bats != ps.throws THEN 1
        ELSE 0
    END                                                     AS platoon_advantage,

    -- ── Target variable ──────────────────────────────────────
    ROUND(((bs.avg_linear_weight + (-ps.avg_lw_allowed)) / 2)
        ::NUMERIC, 4)                                       AS target_lw

FROM batter_stats bs
CROSS JOIN pitcher_stats ps
LEFT JOIN handedness_splits hs
    ON  hs.batter_id    = bs.batter_id
    AND hs.batter_hand  = bs.bats
    AND hs.pitcher_hand = ps.throws
WHERE ps.batters_faced >= 50;