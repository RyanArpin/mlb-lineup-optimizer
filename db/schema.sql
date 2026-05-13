-- =============================================================
-- MLB Lineup Optimizer — Database Schema
-- =============================================================
-- Hierarchy: players → games → at_bats → pitches
-- All Statcast data is ultimately at the pitch level.
-- Feature engineering (Step 4) will aggregate upward from pitches
-- into at_bats, then into player-vs-pitcher summary statistics.
-- =============================================================


-- Drop tables in reverse dependency order so we can re-run
-- this script cleanly during development without FK conflicts.
DROP TABLE IF EXISTS pitches CASCADE;
DROP TABLE IF EXISTS at_bats CASCADE;
DROP TABLE IF EXISTS games CASCADE;
DROP TABLE IF EXISTS players CASCADE;


-- -------------------------------------------------------------
-- PLAYERS
-- One row per unique player (both batters and pitchers).
-- player_id is the MLB MLBAM ID — a stable integer assigned by
-- MLB that pybaseball uses as its primary key.
-- -------------------------------------------------------------
CREATE TABLE players (
    player_id       INTEGER PRIMARY KEY,  -- MLBAM ID from Baseball Savant
    full_name       VARCHAR(100) NOT NULL,
    bats            CHAR(1),              -- 'L', 'R', or 'S' (switch)
    throws          CHAR(1),              -- 'L' or 'R'
    primary_pos     VARCHAR(5),           -- e.g. 'SP', 'RP', '1B', 'CF'
    mlb_debut       DATE,
    created_at      TIMESTAMP DEFAULT NOW()
);


-- -------------------------------------------------------------
-- GAMES
-- One row per game.
-- We store enough context to filter by season, home/away, etc.
-- -------------------------------------------------------------
CREATE TABLE games (
    game_pk         INTEGER PRIMARY KEY,  -- MLB game ID (from Statcast)
    game_date       DATE NOT NULL,
    season          SMALLINT NOT NULL,    -- e.g. 2023
    home_team       VARCHAR(10) NOT NULL, -- e.g. 'TOR', 'NYY'
    away_team       VARCHAR(10) NOT NULL,
    venue           VARCHAR(100),
    created_at      TIMESTAMP DEFAULT NOW()
);


-- -------------------------------------------------------------
-- AT_BATS
-- One row per plate appearance.
-- An at-bat belongs to one game and involves one batter and
-- one pitcher. The result (single, HR, strikeout, walk, etc.)
-- is recorded in `event`, which is what we'll ultimately
-- predict / aggregate for run expectancy.
-- -------------------------------------------------------------
CREATE TABLE at_bats (
    at_bat_id       SERIAL PRIMARY KEY,   -- synthetic PK, auto-incremented
    game_pk         INTEGER NOT NULL REFERENCES games(game_pk),
    batter_id       INTEGER NOT NULL REFERENCES players(player_id),
    pitcher_id      INTEGER NOT NULL REFERENCES players(player_id),
    at_bat_number   SMALLINT NOT NULL,
    inning          SMALLINT NOT NULL,
    inning_half     CHAR(3) NOT NULL,     -- 'top' or 'bot'
    outs_when_up    SMALLINT NOT NULL,    -- 0, 1, or 2
    on_1b           BOOLEAN DEFAULT FALSE,
    on_2b           BOOLEAN DEFAULT FALSE,
    on_3b           BOOLEAN DEFAULT FALSE,
    event           VARCHAR(50),          -- 'single','home_run','strikeout', etc.
    -- Run value of this at-bat outcome (engineered in Step 4)
    -- NULL until we compute it via linear weights SQL query
    linear_weight   NUMERIC(6, 4),
    created_at      TIMESTAMP DEFAULT NOW()
);

ALTER TABLE at_bats
    ADD CONSTRAINT uq_at_bats_natural
    UNIQUE (game_pk, batter_id, pitcher_id, at_bat_number);

-- Index for the most common query pattern:
-- "give me all at-bats for batter X against pitcher Y"
CREATE INDEX idx_at_bats_batter_pitcher
    ON at_bats(batter_id, pitcher_id);


-- -------------------------------------------------------------
-- PITCHES
-- One row per pitch — the most granular level of Statcast data.
-- This is the raw table that gets loaded directly from pybaseball.
-- Most columns map 1-to-1 with Statcast CSV column names to make
-- the data loading step (Step 3) straightforward.
-- -------------------------------------------------------------
CREATE TABLE pitches (
    pitch_id            SERIAL PRIMARY KEY,
    -- Linkage
    at_bat_id           INTEGER REFERENCES at_bats(at_bat_id),
    game_pk             INTEGER NOT NULL REFERENCES games(game_pk),
    batter_id           INTEGER NOT NULL REFERENCES players(player_id),
    pitcher_id          INTEGER NOT NULL REFERENCES players(player_id),
    at_bat_number       SMALLINT NOT NULL,
    pitch_number        SMALLINT NOT NULL,
    game_date           DATE NOT NULL,

    -- Pitch characteristics
    pitch_type          VARCHAR(5),    -- 'FF' (4-seam), 'SL' (slider), etc.
    release_speed       NUMERIC(5, 1), -- mph
    release_spin_rate   NUMERIC(7, 1), -- rpm
    pfx_x               NUMERIC(6, 3), -- horizontal movement (ft)
    pfx_z               NUMERIC(6, 3), -- vertical movement (ft)
    plate_x             NUMERIC(6, 3), -- horizontal location at plate
    plate_z             NUMERIC(6, 3), -- vertical location at plate
    zone                SMALLINT,      -- Statcast strike zone region 1-14

    -- Pitch outcome
    description         VARCHAR(50),   -- 'called_strike', 'ball', 'hit_into_play'
    type                CHAR(1),       -- 'S' strike, 'B' ball, 'X' in play
    balls               SMALLINT,      -- count before this pitch
    strikes             SMALLINT,

    -- Batted ball (NULL if not hit into play)
    launch_speed        NUMERIC(5, 1), -- exit velocity (mph)
    launch_angle        NUMERIC(5, 1), -- degrees
    hit_distance_sc     NUMERIC(6, 1), -- projected distance (ft)
    bb_type             VARCHAR(20),   -- 'ground_ball','fly_ball','line_drive','popup'
    estimated_ba_using_speedangle  NUMERIC(5, 3),  -- xBA
    estimated_woba_using_speedangle NUMERIC(5, 3), -- xwOBA

    -- Context
    stand               CHAR(1),       -- batter handedness this PA: 'L' or 'R'
    p_throws            CHAR(1),       -- pitcher handedness: 'L' or 'R'
    if_fielding_alignment VARCHAR(20),
    of_fielding_alignment VARCHAR(20),

    created_at          TIMESTAMP DEFAULT NOW()
);

ALTER TABLE pitches
    ADD CONSTRAINT uq_pitches_natural
    UNIQUE (game_pk, batter_id, pitcher_id, at_bat_number, pitch_number);

-- Two indexes: one for pitch-level batter analysis,
-- one for pitch-level pitcher analysis
CREATE INDEX idx_pitches_batter  ON pitches(batter_id, game_date);
CREATE INDEX idx_pitches_pitcher ON pitches(pitcher_id, game_date);