"""
sports_vocabulary.py — Sports domain vocabulary for column-level context detection.

Detection pipeline
------------------
1. Normalise each column name:
     - lowercase + strip
     - camelCase split  (teamName → team name)
     - underscore / dash / space split  (minutes_played → minutes played)
     - abbreviation expansion  (xG → expected goals, MP → minutes played)
2. Match the normalised tokens against synonym groups.
   Each synonym group maps a canonical sports term to a list of recognised
   aliases.  A column matches a group if any alias is an exact token-level
   substring of the normalised column name.
3. Count matched columns → compute sports_domain_confidence.
4. Return a SportsContext with per-category column lists, the flat list of
   canonical matched terms, a detected_domain label, and a flag for whether
   to ask the user to confirm the sports framing.

This two-level design (normalization + synonym groups) handles the naming
diversity of real sports data providers:
  StatsBomb → pass_x, shot_freeze_frame, under_pressure
  Wyscout   → teamName, matchId, xg
  NBA API   → PTS, REB, AST, PLUS_MINUS
  Opta      → total_scoring_att, won_contest
  Sofascore → minutesPlayed, attackingActions

Public API
----------
detect_sports_context(df)          -> SportsContext
is_sports_dataset(context)         -> bool
get_domain_question(context)       -> Optional[str]   # "is this sports data?"
get_leakage_question(col_name)     -> Optional[str]
get_missingness_question(col_name) -> Optional[str]
get_cardinality_question(col_name) -> Optional[str]
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Abbreviation expansion — applied before synonym matching
# ---------------------------------------------------------------------------

_ABBREV: Dict[str, str] = {
    # Expected goals
    "xg":   "expected goals",
    "xga":  "expected goals against",
    # Playing time
    "mp":   "minutes played",
    "mins": "minutes",
    "toi":  "time on ice",
    # Basketball box score
    "pts":  "points",
    "reb":  "rebounds",
    "ast":  "assists",
    "stl":  "steals",
    "blk":  "blocks",
    "tov":  "turnovers",
    "fg":   "field goals",
    "ft":   "free throws",
    "pf":   "personal fouls",
    "gp":   "games played",
    # Football / soccer shorthand
    "gls":  "goals",
    "sh":   "shots",
    "tkl":  "tackles",
    "apps": "appearances",
    "poss": "possession",
    "cs":   "clean sheet",
    # Tracking / load
    "hsr":  "high speed running",
    "hrv":  "heart rate variability",
    "hr":   "heart rate",
    "rpe":  "rate of perceived exertion",
    "acwr": "acute chronic workload ratio",
    "tl":   "training load",
    # Physical
    "bmi":  "body mass index",
    "ht":   "height",
    "wt":   "weight",
    "dob":  "date of birth",
    "vo2":  "oxygen uptake",
}


# ---------------------------------------------------------------------------
# Synonym groups by sports domain category
# canonical_term → list of recognised aliases (all lowercase, spaces as separator)
# ---------------------------------------------------------------------------

_ENTITY_SYNONYMS: Dict[str, List[str]] = {
    "player":      ["player", "athlete", "player id", "player name",
                    "athlete id", "athlete name", "first name", "last name",
                    "full name", "squad number"],
    "team":        ["team", "club", "squad", "team name", "club name",
                    "team id", "club id", "side", "franchise"],
    "match":       ["match", "game", "fixture", "contest",
                    "match id", "game id", "fixture id", "event id"],
    "season":      ["season", "season id", "campaign", "league", "competition",
                    "tournament"],
    "position":    ["position", "position code", "role",
                    "jersey number", "shirt number"],
    "nationality": ["nationality", "country", "nation", "citizenship"],
    "coach":       ["coach", "head coach", "trainer"],
    "opponent":    ["opponent", "opposition", "versus", "away team", "home team"],
}

_PERFORMANCE_SYNONYMS: Dict[str, List[str]] = {
    "goals":             ["goals", "goals scored", "goal difference",
                          "goals conceded", "total goals"],
    "expected goals":    ["expected goals", "expected goals against",
                          "xg", "xga"],
    "assists":           ["assists", "assist"],
    "shots":             ["shots", "shots total", "shots on target",
                          "shots off target", "shot attempts"],
    "possession":        ["possession", "ball possession", "total possession"],
    "pass accuracy":     ["pass accuracy", "pass completion",
                          "passes completed", "passes total"],
    "corners":           ["corners", "corner kicks"],
    "fouls":             ["fouls", "fouls committed"],
    "cards":             ["yellow cards", "red cards", "yellow card", "red card"],
    "offsides":          ["offsides", "offside"],
    "tackles":           ["tackles", "tackles won"],
    "interceptions":     ["interceptions"],
    "clearances":        ["clearances"],
    "saves":             ["saves", "clean sheet"],
    "match result":      ["win", "draw", "loss", "result", "match result",
                          "points earned", "points gained", "final score",
                          "total score", "goals scored", "goals conceded"],
    # Basketball
    "points":            ["points", "pts"],
    "rebounds":          ["rebounds"],
    "steals":            ["steals"],
    "blocks":            ["blocks"],
    "turnovers":         ["turnovers"],
    "field goals":       ["field goals", "fg percentage"],
    "free throws":       ["free throws", "ft percentage"],
    "plus minus":        ["plus minus", "game score", "win shares",
                          "box plus minus", "vorp"],
    # Tennis
    "aces":              ["aces", "double faults", "first serve percentage",
                          "break points won"],
    # Cricket
    "runs":              ["runs scored", "wickets taken",
                          "economy rate", "strike rate"],
}

_PLAYING_TIME_SYNONYMS: Dict[str, List[str]] = {
    "minutes":       ["minutes", "minutes played", "time on field",
                      "time on pitch", "time on ice", "time on court",
                      "playing time", "game time"],
    "appearances":   ["appearances", "starts", "games played",
                      "matches played", "contests", "games"],
    "substitution":  ["substitutions in", "substitutions out", "substituted"],
    "innings":       ["innings played", "sets played",
                      "quarters played", "periods played"],
}

_WORKLOAD_SYNONYMS: Dict[str, List[str]] = {
    "distance":       ["distance", "total distance", "distance training",
                       "km run"],
    "sprint":         ["sprint", "sprints", "sprint distance", "sprint speed",
                       "number of sprints", "high speed running",
                       "explosive distance"],
    "training load":  ["training load", "acute load", "chronic load",
                       "acute chronic workload ratio", "session load",
                       "monotony", "strain"],
    "session rpe":    ["rate of perceived exertion", "session rpe",
                       "training rpe"],
    "fitness":        ["fitness", "fatigue index"],
    "heart rate":     ["heart rate", "resting heart rate",
                       "heart rate variability"],
    "acceleration":   ["acceleration"],
}

_INJURY_SYNONYMS: Dict[str, List[str]] = {
    "injury":         ["injury", "injured", "injury type", "injury location",
                       "injury severity", "injury count", "injury history",
                       "acl", "hamstring", "groin", "concussion", "fracture"],
    "return to play": ["return to play", "days injured", "days out",
                       "absence days", "recovery time", "rehabilitation",
                       "re injury", "days missed"],
    "wellness":       ["wellness", "wellness score", "readiness", "fatigue",
                       "sleep quality", "soreness"],
    "availability":   ["availability", "available", "fit"],
}

_PHYSICAL_SYNONYMS: Dict[str, List[str]] = {
    "height":          ["height"],
    "weight":          ["weight", "body mass index"],
    "age":             ["date of birth", "birth date", "player age", "athlete age"],
    "speed":           ["speed", "sprint speed", "top speed", "peak speed"],
    "preferred foot":  ["dominant foot", "preferred foot"],
    "endurance":       ["endurance", "oxygen uptake", "lactate threshold"],
    "body composition":["body fat", "muscle mass", "lean body mass"],
    "agility":         ["agility", "strength", "wingspan"],
}

_SPATIAL_SYNONYMS: Dict[str, List[str]] = {
    "location x":  ["location x", "shot x", "pass x", "start x", "end x"],
    "location y":  ["location y", "shot y", "pass y", "start y", "end y"],
    "event type":  ["event type", "action type"],
    "home away":   ["home", "away", "venue", "stadium"],
}

# Master registry: category → synonym dict
_CATEGORIES: Dict[str, Dict[str, List[str]]] = {
    "performance":  _PERFORMANCE_SYNONYMS,
    "playing_time": _PLAYING_TIME_SYNONYMS,
    "injury":       _INJURY_SYNONYMS,
    "entity":       _ENTITY_SYNONYMS,
    "physical":     _PHYSICAL_SYNONYMS,
    "workload":     _WORKLOAD_SYNONYMS,
    "spatial":      _SPATIAL_SYNONYMS,
}

# Map category → SportsContext bucket attribute name
_CATEGORY_TO_BUCKET: Dict[str, str] = {
    "performance":  "post_match_cols",
    "playing_time": "playing_time_cols",
    "injury":       "injury_cols",
    "entity":       "identity_cols",
    "physical":     "physical_cols",
    "workload":     "workload_cols",
    "spatial":      "post_match_cols",
}

# Confidence thresholds for domain classification
_SPORTS_THRESHOLD    = 0.05   # ≥5% columns matched → call it sports
_AMBIGUOUS_THRESHOLD = 0.02   # 2–5% matched → ask the user to confirm

# Fuzzy matching: similarity ratio threshold (difflib SequenceMatcher).
# 0.75 catches common misspellings and abbreviation variants.
_FUZZY_THRESHOLD = 0.75

# Weight of fuzzy and value-based matches in the confidence calculation
# (full-weight exact matches = 1.0, partial-weight soft signals = 0.5).
_FUZZY_WEIGHT = 0.5

# Sports-related categorical values — scan top sample values of string columns.
_SPORTS_VALUE_TERMS: Dict[str, str] = {
    # Playing positions — football/soccer
    "goalkeeper": "entity", "defender": "entity", "midfielder": "entity",
    "forward": "entity", "striker": "entity", "winger": "entity",
    "fullback": "entity", "center back": "entity", "centre back": "entity",
    # Playing positions — basketball
    "point guard": "entity", "shooting guard": "entity",
    "power forward": "entity", "small forward": "entity",
    # Match outcomes
    "home win": "performance", "away win": "performance",
    "clean sheet": "performance",
    # Injury types
    "hamstring": "injury", "anterior cruciate": "injury",
    "groin strain": "injury", "ankle sprain": "injury",
    "muscle tear": "injury", "concussion": "injury",
    # Event types (Opta / StatsBomb)
    "shot on target": "performance", "key pass": "performance",
    "successful dribble": "performance", "tackle won": "performance",
    "aerial duel": "performance",
    # Venues
    "home": "spatial", "away": "spatial",
}

def _build_alias_list() -> List[Tuple[str, str, str]]:
    """Lazily build the flat (alias, category, canonical) list for fuzzy matching."""
    result = []
    for cat, synonyms in _CATEGORIES.items():
        for canonical, aliases in synonyms.items():
            for alias in aliases:
                if len(alias) >= 4:
                    result.append((alias, cat, canonical))
    return result


_ALL_ALIASES: Optional[List[Tuple[str, str, str]]] = None


def _get_aliases() -> List[Tuple[str, str, str]]:
    global _ALL_ALIASES
    if _ALL_ALIASES is None:
        _ALL_ALIASES = _build_alias_list()
    return _ALL_ALIASES


# ---------------------------------------------------------------------------
# SportsContext result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SportsContext:
    """
    Output of detect_sports_context() — a structured breakdown of the sports
    domain signals detected in a DataFrame's column names.

    Attributes
    ----------
    post_match_cols    : columns that likely record post-event measurements
                         (leakage risk if used to predict the same event).
    playing_time_cols  : columns recording participation / playing time.
    injury_cols        : columns related to injury or wellness.
    identity_cols      : player / team / match identifiers (should not be features).
    physical_cols      : physical / biometric attributes (safe model features).
    workload_cols      : training load / workload metrics.
    is_sports          : True when at least one column matches a sports synonym.
    confidence         : fraction of columns that matched any sports synonym (0–1).
    matched_terms      : canonical sports terms that were detected
                         (e.g. ["player", "minutes", "expected goals"]).
    detected_domain    : "sports" | "possible_sports" | "general_tabular"
    """
    post_match_cols:   List[str] = field(default_factory=list)
    playing_time_cols: List[str] = field(default_factory=list)
    injury_cols:       List[str] = field(default_factory=list)
    identity_cols:     List[str] = field(default_factory=list)
    physical_cols:     List[str] = field(default_factory=list)
    workload_cols:     List[str] = field(default_factory=list)
    is_sports:         bool = False
    confidence:        float = 0.0
    matched_terms:     List[str] = field(default_factory=list)
    detected_domain:   str = "general_tabular"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _split_camel(name: str) -> str:
    """Insert a space before each uppercase letter that follows a lowercase letter."""
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", name)


def _norm(name: str) -> str:
    """
    Normalise a column name into a space-separated lowercase token string.

    Steps:
      1. camelCase split  (teamName → team Name → team name)
      2. lowercase
      3. replace underscores / dashes with spaces
      4. collapse multiple spaces
      5. expand known abbreviations as whole tokens
    """
    s = _split_camel(name)
    s = s.lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Expand abbreviations that appear as standalone tokens
    tokens = s.split()
    expanded = [_ABBREV.get(t, t) for t in tokens]
    return " ".join(expanded)


def _matches_synonym_group(normalised: str, aliases: List[str]) -> bool:
    """
    Return True if any alias string appears as a token-boundary substring of
    the normalised column name.

    Whole-word boundary matching prevents "ast" (assists) from matching "assistant".
    """
    for alias in aliases:
        # Build a regex that requires word boundaries around the alias phrase
        pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
        if re.search(pattern, normalised):
            return True
    return False


def _match_column(col: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to match a column name against all synonym groups.

    Returns (category, canonical_term) of the first match found,
    or (None, None) if no match.

    Category priority order ensures that e.g. "goals_scored" is classified
    as "performance" (leakage risk) rather than a generic entity.
    """
    norm = _norm(col)
    for category, synonyms in _CATEGORIES.items():
        for canonical, aliases in synonyms.items():
            if _matches_synonym_group(norm, aliases):
                return category, canonical
    return None, None


def _fuzzy_match_column(col: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fuzzy-match a column name against all known sports alias strings.

    Uses difflib.SequenceMatcher to catch misspellings and non-standard
    abbreviations that slip through exact synonym matching.  Only aliases
    of length ≥ 4 are considered to avoid spurious short-token matches.

    Returns (category, canonical_term) for the best match above
    _FUZZY_THRESHOLD, or (None, None) if no close match found.
    """
    norm = _norm(col)
    best_ratio = 0.0
    best_cat: Optional[str] = None
    best_canon: Optional[str] = None

    for alias, cat, canonical in _get_aliases():
        ratio = difflib.SequenceMatcher(None, norm, alias).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_cat = cat
            best_canon = canonical

    if best_ratio >= _FUZZY_THRESHOLD:
        return best_cat, best_canon
    return None, None


def sample_categorical_values(df: pd.DataFrame, top_n: int = 15) -> List[str]:
    """
    Scan the top-N most frequent values of categorical columns for sports terms.

    Catches sports datasets where column names are generic (e.g. "type", "category",
    "status") but the values themselves reveal the domain ("goalkeeper", "tackle",
    "hamstring").

    Parameters
    ----------
    df     : DataFrame to scan (any columns; non-string columns are skipped).
    top_n  : maximum number of distinct values to check per column.

    Returns
    -------
    List of column names whose values contain at least one recognised sports term.
    """
    matched_cols: List[str] = []
    for col in df.columns:
        if not (df[col].dtype == object or str(df[col].dtype) == "category"):
            continue
        top_values = df[col].dropna().astype(str).value_counts().head(top_n).index
        for val in top_values:
            val_lower = val.lower().strip()
            for term in _SPORTS_VALUE_TERMS:
                if term in val_lower:
                    matched_cols.append(col)
                    break
            else:
                continue
            break
    return matched_cols


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_sports_context(df: pd.DataFrame) -> SportsContext:
    """
    Scan a DataFrame's column names (and optionally categorical values) and
    return a SportsContext describing which columns appear to be sports-related.

    Detection pipeline
    ------------------
    1. Exact synonym matching on normalised column names (full confidence weight).
    2. Fuzzy column-name matching via difflib for misspellings / abbreviations
       (half confidence weight).
    3. Categorical value sampling — scans the top-N values of string columns
       for sports domain terms such as position names or injury types
       (half confidence weight).

    Handles naming conventions from StatsBomb, Wyscout, Opta, NBA API,
    Sofascore, and other sports data providers.

    Parameters
    ----------
    df : any DataFrame (target column may be included or excluded).

    Returns
    -------
    SportsContext with categorised column lists, matched canonical terms,
    domain label, and confidence score.
    """
    buckets: Dict[str, List[str]] = {
        "post_match_cols":   [],
        "playing_time_cols": [],
        "injury_cols":       [],
        "identity_cols":     [],
        "physical_cols":     [],
        "workload_cols":     [],
    }
    matched_terms: List[str] = []
    exact_matched: Set[str] = set()
    weighted_sum = 0.0  # sum of per-column match weights for confidence

    # --- Pass 1: exact synonym matching ---
    for col in df.columns:
        category, canonical = _match_column(col)
        if category is not None:
            bucket = _CATEGORY_TO_BUCKET[category]
            buckets[bucket].append(col)
            if canonical not in matched_terms:
                matched_terms.append(canonical)
            exact_matched.add(col)
            weighted_sum += 1.0

    # --- Pass 2: fuzzy column-name matching (for unmatched columns only) ---
    fuzzy_matched: Set[str] = set()
    for col in df.columns:
        if col in exact_matched:
            continue
        category, canonical = _fuzzy_match_column(col)
        if category is not None:
            bucket = _CATEGORY_TO_BUCKET[category]
            if col not in buckets[bucket]:
                buckets[bucket].append(col)
            if canonical not in matched_terms:
                matched_terms.append(canonical)
            fuzzy_matched.add(col)
            weighted_sum += _FUZZY_WEIGHT

    # --- Pass 3: categorical value sampling (for still-unmatched columns) ---
    value_matched_cols = sample_categorical_values(df)
    for col in value_matched_cols:
        if col in exact_matched or col in fuzzy_matched:
            continue
        # Classify under "entity" (identity column) by default for value matches
        bucket = "identity_cols"
        if col not in buckets[bucket]:
            buckets[bucket].append(col)
        if "player" not in matched_terms:
            matched_terms.append("player")
        weighted_sum += _FUZZY_WEIGHT

    confidence = weighted_sum / max(len(df.columns), 1)

    if confidence >= _SPORTS_THRESHOLD:
        detected_domain = "sports"
    elif confidence >= _AMBIGUOUS_THRESHOLD:
        detected_domain = "possible_sports"
    else:
        detected_domain = "general_tabular"

    return SportsContext(
        post_match_cols=buckets["post_match_cols"],
        playing_time_cols=buckets["playing_time_cols"],
        injury_cols=buckets["injury_cols"],
        identity_cols=buckets["identity_cols"],
        physical_cols=buckets["physical_cols"],
        workload_cols=buckets["workload_cols"],
        is_sports=(detected_domain in ("sports", "possible_sports")),
        confidence=confidence,
        matched_terms=matched_terms,
        detected_domain=detected_domain,
    )


def is_sports_dataset(context: SportsContext, threshold: float = _SPORTS_THRESHOLD) -> bool:
    """
    Return True when the dataset's column vocabulary clearly suggests sports data.

    Parameters
    ----------
    context   : SportsContext from detect_sports_context().
    threshold : minimum confidence fraction required (default 0.05).
    """
    return context.confidence >= threshold


def get_domain_question(context: SportsContext) -> Optional[str]:
    """
    Return a clarification question when the dataset's domain is ambiguous —
    i.e. some sports signals were detected but below the confident threshold.

    Returns None when the domain is clearly sports (≥5% columns matched)
    or clearly non-sports (0 matches).

    Parameters
    ----------
    context : SportsContext from detect_sports_context().
    """
    if context.detected_domain != "possible_sports":
        return None

    terms_str = ", ".join(context.matched_terms[:5]) if context.matched_terms else "a few columns"
    return (
        f"This dataset does not clearly appear to be sports-related "
        f"({context.confidence:.0%} of columns matched sports vocabulary — "
        f"terms detected: {terms_str}). "
        f"Is it intended for sports analytics, or should the system continue "
        f"in general tabular ML mode? Answering 'sports' will enable sports-aware "
        f"issue detection (leakage warnings for post-match statistics, playing-time "
        f"imputation guidance, etc.)."
    )


def get_leakage_question(col_name: str) -> Optional[str]:
    """
    Return a sports-specific leakage clarification question if the column name
    matches a known performance (post-match) statistic, or None otherwise.

    Parameters
    ----------
    col_name : name of the leakage-candidate column.
    """
    norm = _norm(col_name)
    for canonical, aliases in _PERFORMANCE_SYNONYMS.items():
        if _matches_synonym_group(norm, aliases):
            return (
                f"Column '{col_name}' looks like a post-match performance statistic "
                f"(matched sports term: '{canonical}'). "
                f"Is this value recorded BEFORE the event or outcome you want to predict, "
                f"or AFTER it? If it is recorded after the event — for example, the final "
                f"score used to predict the match winner — it must be excluded to prevent "
                f"data leakage and inflated model performance."
            )
    # Also check spatial synonyms (event coordinates are also post-match)
    for canonical, aliases in _SPATIAL_SYNONYMS.items():
        if _matches_synonym_group(norm, aliases):
            return (
                f"Column '{col_name}' looks like a spatial or event-level measurement "
                f"(matched sports term: '{canonical}'). "
                f"Is this value available BEFORE the event you want to predict, "
                f"or is it recorded during / after it? Post-event spatial data can "
                f"cause data leakage if used to predict the same event's outcome."
            )
    return None


def get_missingness_question(col_name: str) -> Optional[str]:
    """
    Return a sports-specific missingness clarification question if the column
    matches a playing-time or injury/wellness indicator, or None otherwise.

    Parameters
    ----------
    col_name : name of the column with high missingness.
    """
    norm = _norm(col_name)
    for canonical, aliases in _PLAYING_TIME_SYNONYMS.items():
        if _matches_synonym_group(norm, aliases):
            return (
                f"Column '{col_name}' tracks playing time or participation "
                f"(matched sports term: '{canonical}'). "
                f"Does a missing value here mean the athlete DID NOT PLAY "
                f"(in which case missing = 0 is the right imputation), or that "
                f"the data was simply not collected for administrative reasons "
                f"(in which case mean or median imputation is more appropriate)?"
            )
    for canonical, aliases in _INJURY_SYNONYMS.items():
        if _matches_synonym_group(norm, aliases):
            return (
                f"Column '{col_name}' appears to record injury or wellness information "
                f"(matched sports term: '{canonical}'). "
                f"Does a missing value here mean NO INJURY / NO ISSUE was recorded "
                f"(in which case missing = 0 or a 'none' category is appropriate), "
                f"or that the data was not collected at all?"
            )
    return None


def get_cardinality_question(col_name: str) -> Optional[str]:
    """
    Return a sports-specific cardinality clarification question if the column
    matches a known entity/identity pattern, or None otherwise.

    Parameters
    ----------
    col_name : name of the high-cardinality column.
    """
    norm = _norm(col_name)
    for canonical, aliases in _ENTITY_SYNONYMS.items():
        if _matches_synonym_group(norm, aliases):
            return (
                f"Column '{col_name}' looks like a player name, team name, or match "
                f"identifier (matched sports term: '{canonical}'). "
                f"These columns typically have too many unique values to be used "
                f"directly as a model feature. Should this column be EXCLUDED from "
                f"the model, or does it contain meaningful information you want to keep "
                f"(e.g. as a grouping variable for aggregation)?"
            )
    return None
