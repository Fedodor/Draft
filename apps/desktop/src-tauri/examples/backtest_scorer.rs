//! Offline backtest of the deterministic draft scorer.
//!
//! Replays historical Captains Mode drafts (exact pick/ban order) through the
//! shipping engine with ML disabled, and measures two things the unit tests
//! cannot:
//!
//! 1. **Agreement**: how often the engine's top-ranked available hero is the
//!    hero the human team actually picked, overall and per draft phase, against
//!    a chronological popularity baseline and the random-choice expectation.
//! 2. **Outcome proxy**: whether the engine's rating of the heroes a team
//!    actually drafted is associated with that team winning, using per-pick
//!    scores (ban-aware, because every ban is replayed before the pick) and
//!    per-side percentile sums.
//!
//! It also records whether a ban removes the banning team's top recommendation,
//! which is a ban-aware signal that the ranking surfaces heroes opponents fear.
//!
//! Read-only by construction: the SQLite database is copied (with its WAL and
//! SHM siblings) into a temporary directory before opening, so the app's data is
//! never migrated, checkpointed, or written to.
//!
//! Run:
//!   cargo run --release --example backtest_scorer -- \
//!     --drafts ml/backtest/data/drafts_cm.jsonl \
//!     --db "%APPDATA%\com.smartdraftassistant.app\smart_draft.db" \
//!     --out ml/backtest/results/backtest.json \
//!     --picks-csv ml/backtest/results/picks.csv

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use smart_draft_assistant::database::Database;
use smart_draft_assistant::engine::DraftEngine;
use smart_draft_assistant::models::{
    DraftBan, DraftPick, HeroId, PlayerId, ScoreBreakdown, TeamSide,
};

#[derive(Debug, Deserialize, Clone)]
struct DraftRecord {
    match_id: i64,
    #[serde(default)]
    start_time: Option<i64>,
    #[serde(default)]
    patch: Option<String>,
    radiant_win: bool,
    actions: Vec<Action>,
}

#[derive(Debug, Deserialize, Clone)]
struct Action {
    order: i32,
    pick: bool,
    radiant: bool,
    hero_id: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    /// Picks 1-2, the opening pair.
    Opening,
    /// Picks 3-4.
    Early,
    /// Picks 5-6.
    Mid,
    /// Picks 7-8.
    Late,
    /// Picks 9-10, the last two.
    Final,
}

impl Phase {
    fn of(pick_ordinal: i32) -> Self {
        match pick_ordinal {
            1 | 2 => Phase::Opening,
            3 | 4 => Phase::Early,
            5 | 6 => Phase::Mid,
            7 | 8 => Phase::Late,
            _ => Phase::Final,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize)]
struct Counts {
    events: u64,
    top1: u64,
    top3: u64,
    top5: u64,
    /// Sum of the random-choice top-1 expectation (1 / available heroes).
    random_top1_expected: f64,
    rank_sum: f64,
    percentile_sum: f64,
}

impl Counts {
    fn record(&mut self, rank: usize, available: usize) {
        self.events += 1;
        self.rank_sum += rank as f64;
        if available > 1 {
            self.percentile_sum += 1.0 - (rank - 1) as f64 / (available - 1) as f64;
        } else {
            self.percentile_sum += 0.5;
        }
        if rank == 1 {
            self.top1 += 1;
        }
        if rank <= 3 {
            self.top3 += 1;
        }
        if rank <= 5 {
            self.top5 += 1;
        }
        if available > 0 {
            self.random_top1_expected += 1.0 / available as f64;
        }
    }

    fn agreement(&self) -> Agreement {
        Agreement {
            events: self.events,
            top1: ratio(self.top1, self.events),
            top3: ratio(self.top3, self.events),
            top5: ratio(self.top5, self.events),
            random_top1: ratio_f64(self.random_top1_expected, self.events),
            mean_rank: ratio_f64(self.rank_sum, self.events),
            mean_percentile: ratio_f64(self.percentile_sum, self.events),
        }
    }
}

#[derive(Debug, Serialize)]
struct Agreement {
    events: u64,
    top1: f64,
    top3: f64,
    top5: f64,
    random_top1: f64,
    /// Mean rank of the human pick among the available heroes (1 = top).
    mean_rank: f64,
    /// Mean percentile of the human pick; 0.5 is random ordering.
    mean_percentile: f64,
}

#[derive(Debug, Default, Clone, Serialize)]
struct OutcomeBucket {
    picks: u64,
    wins: u64,
}

impl OutcomeBucket {
    fn record(&mut self, won: bool) {
        self.picks += 1;
        if won {
            self.wins += 1;
        }
    }

    fn summary(&self) -> OutcomeSummary {
        let (low, high) = wilson(self.wins, self.picks);
        OutcomeSummary {
            picks: self.picks,
            win_rate: ratio(self.wins, self.picks),
            ci95_low: low,
            ci95_high: high,
        }
    }
}

#[derive(Debug, Serialize)]
struct OutcomeSummary {
    picks: u64,
    win_rate: f64,
    ci95_low: f64,
    ci95_high: f64,
}

#[derive(Debug, Default, Clone, Serialize)]
struct BanCounts {
    bans: u64,
    removes_top1: u64,
    removes_top3: u64,
}

#[derive(Debug, Clone, Serialize)]
struct PickRow {
    match_id: i64,
    patch: String,
    start_time: i64,
    pick_ordinal: i32,
    side: String,
    hero_id: i32,
    rank: usize,
    available: usize,
    score: f32,
    percentile: f64,
    won: bool,
    human_freq_rank: Option<usize>,
}

#[derive(Debug, Serialize)]
struct Discordant {
    blend_only: u64,
    baseline_only: u64,
}

#[derive(Debug, Serialize)]
struct BlendDiscordance {
    weight: f64,
    vs_engine: Discordant,
    vs_popularity: Discordant,
}

#[derive(Debug, Serialize)]
struct GateDiscordance {
    k: usize,
    gate_only: u64,
    prior_only: u64,
}

#[derive(Debug, Serialize)]
struct BanPhase {
    engine_eval: Agreement,
    prior_global_eval: Agreement,
    prior_per_patch_eval: Agreement,
}

#[derive(Debug, Serialize)]
struct MlModelInfo {
    path: String,
    features: usize,
}

#[derive(Debug, Serialize)]
struct VerifySummary {
    checked: usize,
    top5_mismatches: usize,
}

#[derive(Debug, Serialize)]
struct DbInfo {
    path: String,
    heroes: usize,
    matchups: usize,
    synergies: usize,
    patch: String,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at: String,
    db: DbInfo,
    ml_enabled: bool,
    /// Which popularity prior the baseline and the blend used.
    prior: String,
    /// `None` when the run used the deterministic path only.
    ml_model: Option<MlModelInfo>,
    dataset: DatasetInfo,
    engine_agreement: Agreement,
    engine_agreement_eval: Agreement,
    engine_by_pick_ordinal: Vec<(i32, Agreement)>,
    engine_by_phase: Vec<(Phase, Agreement)>,
    frequency_baseline_eval: Agreement,
    frequency_by_phase_eval: Vec<(Phase, Agreement)>,
    skipped: Skipped,
    rank_outcome: Vec<(String, OutcomeSummary)>,
    rank_outcome_by_phase: Vec<(Phase, Vec<(String, OutcomeSummary)>)>,
    ban_threat: BansSummary,
    team_score_auc: Option<f64>,
    team_percentile_auc: Option<f64>,
    matches_for_auc: u64,
    /// Same event stream and availability, ranked by single components.
    ablations: Vec<(String, Agreement)>,
    /// Engine/popularity percentile blend, keyed by the popularity weight.
    blend_sweep: Vec<(f64, Agreement)>,
    /// Paired top-1 discordance for the best blend weight (McNemar inputs).
    blend_top1_discordance: BlendDiscordance,
    /// The same sweep on each evaluation half, in case the weight is overfit.
    blend_halves: Vec<(f64, Agreement, Agreement)>,
    /// Prior top-K then engine order, keyed by K.
    gate_sweep: Vec<(usize, Agreement)>,
    /// Paired top-1 discordance for the gating size above (McNemar inputs).
    gate_top1_discordance: GateDiscordance,
    /// How the ban phase scores under the engine and under ban prevalence.
    ban_phase: BanPhase,
    verify: VerifySummary,
}

#[derive(Debug, Default, Serialize)]
struct Skipped {
    picks: u64,
    bans: u64,
    matches_missing_heroes: u64,
}

#[derive(Debug, Serialize)]
struct DatasetInfo {
    drafts: String,
    matches_total: usize,
    matches_evaluated: usize,
    matches_available: usize,
    warmup_matches: usize,
    warmup_fraction: f64,
    first_start_time: Option<i64>,
    last_start_time: Option<i64>,
    patches: Vec<(String, usize)>,
}

#[derive(Debug, Serialize)]
struct BansSummary {
    overall: BanRate,
    /// Bans taken after at least one pick, where pick preferences are known.
    post_pick: BanRate,
    /// Keyed by how many picks had already happened when the ban was made.
    by_picks_before: Vec<(i32, BanRate)>,
}

#[derive(Debug, Serialize)]
struct BanRate {
    bans: u64,
    top1: f64,
    top3: f64,
}

fn ban_rate(counts: &BanCounts) -> BanRate {
    BanRate {
        bans: counts.bans,
        top1: ratio(counts.removes_top1, counts.bans),
        top3: ratio(counts.removes_top3, counts.bans),
    }
}

fn ratio(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

fn ratio_f64(part: f64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part / whole as f64
    }
}

/// 95% Wilson score interval for a binomial proportion.
fn wilson(successes: u64, trials: u64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 0.0);
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.96_f64;
    let denom = 1.0 + z * z / n;
    let centre = p + z * z / (2.0 * n);
    let margin = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt();
    (
        ((centre - margin) / denom).max(0.0),
        ((centre + margin) / denom).min(1.0),
    )
}

/// Rank-based AUC with ties counted as half a win.
fn auc(scores: &[(f64, bool)]) -> Option<f64> {
    let positives = scores.iter().filter(|(_, won)| *won).count();
    let negatives = scores.len() - positives;
    if positives == 0 || negatives == 0 {
        return None;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Process ties in blocks.
    let mut index = 0;
    let mut rank_sum = 0.0_f64;
    while index < sorted.len() {
        let mut end = index + 1;
        while end < sorted.len() && sorted[end].0 == sorted[index].0 {
            end += 1;
        }
        let average_rank = (index + 1 + end) as f64 / 2.0;
        for entry in &sorted[index..end] {
            if entry.1 {
                rank_sum += average_rank;
            }
        }
        index = end;
    }
    let positives = positives as f64;
    let negatives = negatives as f64;
    Some((rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives))
}

/// Which number the ablation ranks heroes by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Ranking {
    /// The shipped heuristic total (what the app sorts by).
    Engine,
    /// Composition scorer alone.
    Composition,
    /// The two draft-interaction components, equally weighted.
    MatchupSynergy,
    /// Seeded meta score alone.
    Meta,
}

impl Ranking {
    fn key(&self, breakdown: &ScoreBreakdown) -> f32 {
        match self {
            Ranking::Engine => breakdown.total,
            Ranking::Composition => breakdown.composition,
            Ranking::MatchupSynergy => 0.5 * breakdown.matchup + 0.5 * breakdown.synergy,
            Ranking::Meta => breakdown.meta,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Ranking::Engine => "engine",
            Ranking::Composition => "composition_only",
            Ranking::MatchupSynergy => "matchup_synergy_only",
            Ranking::Meta => "meta_only",
        }
    }
}

/// Every available hero scored once for `team`, with all components kept so the
/// ablation can re-rank the same pool without re-scoring.
fn score_pool(engine: &DraftEngine, team: TeamSide) -> Vec<(i32, ScoreBreakdown)> {
    let available = engine.get_state().available_heroes;
    available
        .iter()
        .filter_map(|hero| {
            engine
                .score_breakdown(hero, team)
                .map(|breakdown| (hero.0, breakdown))
        })
        .collect()
}

fn ranked_in(pool: &[(i32, ScoreBreakdown)], ranking: Ranking) -> Vec<(i32, f32)> {
    let mut scored: Vec<(i32, f32)> = pool
        .iter()
        .map(|(id, breakdown)| (*id, ranking.key(breakdown)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored
}

/// Where the popularity prior comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PriorMode {
    /// One prior fitted on the chronological warm-up window, used everywhere.
    Global,
    /// Expanding counts per patch from the drafts already replayed, falling back
    /// to the global warm-up prior until a patch has enough history.
    PerPatch,
}

impl PriorMode {
    fn label(&self) -> &'static str {
        match self {
            PriorMode::Global => "global_warmup",
            PriorMode::PerPatch => "per_patch_rolling",
        }
    }
}

/// A patch needs this many replayed picks before its own counts are used.
const PATCH_PRIOR_MIN_PICKS: u64 = 200;
/// A patch needs this many replayed bans before its own ban counts are used.
const PATCH_BAN_PRIOR_MIN_BANS: u64 = 200;

fn prior_for_draft(
    mode: PriorMode,
    patch: &str,
    per_patch: &HashMap<String, HashMap<i32, u64>>,
    global: &HashMap<i32, u64>,
    min_total: u64,
) -> HashMap<i32, u64> {
    match mode {
        PriorMode::Global => global.clone(),
        PriorMode::PerPatch => per_patch
            .get(patch)
            .filter(|counts| counts.values().sum::<u64>() >= min_total)
            .cloned()
            .unwrap_or_else(|| global.clone()),
    }
}

/// Heroes ordered by a count table (bans use the same shape as picks).
fn prior_ranking(ids: impl Iterator<Item = i32>, counts: &HashMap<i32, u64>) -> Vec<(i32, u64)> {
    let mut ranked: Vec<(i32, u64)> = ids
        .map(|id| (id, counts.get(&id).copied().unwrap_or(0)))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked
}

/// Prior top-K candidates in engine order, then everything else in engine order.
/// The front of the list is what a gated recommendation would show.
fn gated_ranking(
    engine_order: &[(i32, f32)],
    prior_order: &[(i32, u64)],
    k: usize,
) -> Vec<(i32, f32)> {
    let allowed: std::collections::HashSet<i32> =
        prior_order.iter().take(k).map(|(id, _)| *id).collect();
    let mut out: Vec<(i32, f32)> = Vec::with_capacity(engine_order.len());
    out.extend(
        engine_order
            .iter()
            .filter(|(id, _)| allowed.contains(id))
            .copied(),
    );
    out.extend(
        engine_order
            .iter()
            .filter(|(id, _)| !allowed.contains(id))
            .copied(),
    );
    out
}

/// Prior sizes tried by the gating test.
const GATE_KS: [usize; 5] = [5, 10, 20, 40, 80];
/// Gating size used for the paired top-1 comparison against the plain prior.
const GATE_DISCORDANCE_K: usize = 5;

/// Blend weights as percentages, so the sweep is exact and hashable.
/// 0 reproduces the engine ranking, 100 reproduces the popularity ranking.
const BLEND_WEIGHTS: [i32; 11] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

/// Rank heroes by `(1 - weight) * engine_percentile + weight * popularity_percentile`,
/// both percentiles taken within the current available pool so the blend is
/// scale-free. The popularity prior itself is fitted on the warm-up drafts only.
fn blend_ranking(
    pool: &[(i32, ScoreBreakdown)],
    popularity: &HashMap<i32, u64>,
    weight: f64,
) -> Vec<(i32, f32)> {
    let count = pool.len();
    let percentile = |rank: usize| {
        if count > 1 {
            1.0 - rank as f64 / (count - 1) as f64
        } else {
            0.5
        }
    };

    let mut engine_order: Vec<(i32, f32)> = pool
        .iter()
        .map(|(id, breakdown)| (*id, breakdown.total))
        .collect();
    engine_order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let engine_percentile: HashMap<i32, f64> = engine_order
        .iter()
        .enumerate()
        .map(|(rank, (id, _))| (*id, percentile(rank)))
        .collect();

    let mut freq_order: Vec<(i32, u64)> = pool
        .iter()
        .map(|(id, _)| (*id, popularity.get(id).copied().unwrap_or(0)))
        .collect();
    freq_order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let freq_percentile: HashMap<i32, f64> = freq_order
        .iter()
        .enumerate()
        .map(|(rank, (id, _))| (*id, percentile(rank)))
        .collect();

    let mut blended: Vec<(i32, f32)> = pool
        .iter()
        .map(|(id, _)| {
            let engine = engine_percentile.get(id).copied().unwrap_or(0.5);
            let freq = freq_percentile.get(id).copied().unwrap_or(0.5);
            (*id, ((1.0 - weight) * engine + weight * freq) as f32)
        })
        .collect();
    blended.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    blended
}

fn position_of_counts(ranked: &[(i32, u64)], hero_id: i32) -> Option<usize> {
    ranked
        .iter()
        .position(|(id, _)| *id == hero_id)
        .map(|index| index + 1)
}

fn rank_of(scored: &[(i32, f32)], hero_id: i32) -> Option<(usize, f32)> {
    scored
        .iter()
        .position(|(id, _)| *id == hero_id)
        .map(|index| (index + 1, scored[index].1))
}

fn parse_args() -> Result<HashMap<String, String>, String> {
    let mut args = HashMap::new();
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        let flag = flag.trim_start_matches('-').to_string();
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for --{flag}"))?;
        args.insert(flag, value);
    }
    Ok(args)
}

fn github_commit() -> String {
    std::env::var("GITHUB_SHA").unwrap_or_else(|_| {
        std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    })
}

/// Copy the SQLite database and its WAL sidecars so the original is untouched.
fn copy_database(source: &Path, destination_dir: &Path) -> Result<PathBuf, String> {
    if !source.is_file() {
        return Err(format!(
            "database not found at {} (pass --db <path to smart_draft.db>)",
            source.display()
        ));
    }
    fs::create_dir_all(destination_dir).map_err(|error| error.to_string())?;
    let target = destination_dir.join("smart_draft.db");
    fs::copy(source, &target).map_err(|error| format!("copy database: {error}"))?;
    for suffix in ["-wal", "-shm"] {
        let sidecar = PathBuf::from(format!("{}{suffix}", source.display()));
        if sidecar.is_file() {
            let sidecar_target = PathBuf::from(format!("{}{suffix}", target.display()));
            fs::copy(&sidecar, &sidecar_target).map_err(|error| error.to_string())?;
        }
    }
    Ok(target)
}

fn load_drafts(path: &Path) -> Result<Vec<DraftRecord>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut drafts = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: DraftRecord = serde_json::from_str(line)
            .map_err(|error| format!("{}:{}: {error}", path.display(), line_number + 1))?;
        drafts.push(record);
    }
    Ok(drafts)
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let drafts_path = PathBuf::from(args.get("drafts").ok_or("--drafts <jsonl> is required")?);
    let db_path = PathBuf::from(args.get("db").ok_or("--db <smart_draft.db> is required")?);
    let out_path = PathBuf::from(
        args.get("out")
            .map(String::as_str)
            .unwrap_or("ml/backtest/results/backtest.json"),
    );
    let picks_csv = args.get("picks-csv").map(PathBuf::from);
    let warmup_fraction: f64 = args
        .get("warmup-fraction")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.6);
    let limit: Option<usize> = args.get("limit").and_then(|value| value.parse().ok());
    let sample: Option<usize> = args.get("sample").and_then(|value| value.parse().ok());
    let patch = args.get("patch").cloned();
    let verify: usize = args
        .get("verify")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let ml_on = args
        .get("ml")
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);
    let model_path = args
        .get("model")
        .cloned()
        .unwrap_or_else(|| "ml/models/draft_model_xgboost_full.onnx".to_string());
    let prior_mode = match args.get("prior").map(String::as_str) {
        Some("per-patch") | Some("per_patch") => PriorMode::PerPatch,
        _ => PriorMode::Global,
    };

    let mut drafts = load_drafts(&drafts_path)?;
    drafts.sort_by_key(|draft| (draft.start_time.unwrap_or(0), draft.match_id));
    if let Some(patch) = &patch {
        drafts.retain(|draft| draft.patch.as_deref() == Some(patch.as_str()));
    }
    let matches_available = drafts.len();
    if let Some(sample) = sample {
        if sample > 0 && sample < drafts.len() {
            // Even spacing keeps every patch and both eras represented, unlike
            // truncating to the first N matches.
            let stride = drafts.len() as f64 / sample as f64;
            drafts = (0..sample)
                .map(|index| drafts[(index as f64 * stride) as usize].clone())
                .collect();
        }
    }
    if let Some(limit) = limit {
        drafts.truncate(limit);
    }
    if drafts.is_empty() {
        return Err("no drafts loaded".to_string());
    }

    let temp_dir = std::env::temp_dir().join(format!(
        "sda-backtest-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    ));
    let copied = copy_database(&db_path, &temp_dir)?;

    let database = Arc::new(Database::new(temp_dir.clone()).map_err(|error| error.to_string())?);
    let mut engine = DraftEngine::new();
    engine.set_database(database);
    engine.initialize().map_err(|error| error.to_string())?;

    // ML is an enhancement layer: load the real bundled model when asked for the
    // ML path, and fail loudly rather than silently scoring a neutral 0.5.
    let ml_model = if ml_on {
        let shared = smart_draft_assistant::ml::get_ml_engine();
        let mut guard = shared.write();
        guard
            .load_model(&model_path)
            .map_err(|error| format!("ML model load failed ({model_path}): {error}"))?;
        if !guard.is_loaded() {
            return Err(format!("ML model at {model_path} did not load"));
        }
        Some(MlModelInfo {
            path: model_path.clone(),
            features: guard.feature_count(),
        })
    } else {
        None
    };
    engine.set_ml_enabled(ml_on);

    let state = engine.get_state();
    if state.available_heroes.is_empty() || engine.scorer.matchup_hero_count() == 0 {
        return Err(format!(
            "database at {} has no usable data (heroes={}, matchup heroes={}); seed it first",
            db_path.display(),
            state.available_heroes.len(),
            engine.scorer.matchup_hero_count()
        ));
    }
    let db_info = DbInfo {
        path: db_path.display().to_string(),
        heroes: engine.hero_db.len(),
        matchups: engine.scorer.matchup_hero_count(),
        synergies: engine.scorer.synergy_hero_count(),
        patch: state.patch_version.clone(),
    };

    let warmup_matches = ((drafts.len() as f64) * warmup_fraction).round() as usize;
    let warmup_matches = warmup_matches.min(drafts.len());

    // Popularity baseline is fitted on the chronological warm-up window only.
    let mut popularity: HashMap<i32, u64> = HashMap::new();
    for draft in drafts.iter().take(warmup_matches) {
        for action in draft.actions.iter().filter(|action| action.pick) {
            *popularity.entry(action.hero_id).or_default() += 1;
        }
    }

    let mut engine_counts = Counts::default();
    let mut engine_eval_counts = Counts::default();
    let mut by_ordinal: HashMap<i32, Counts> = HashMap::new();
    let mut by_phase: HashMap<Phase, Counts> = HashMap::new();
    let mut freq_eval = Counts::default();
    let mut freq_by_phase: HashMap<Phase, Counts> = HashMap::new();
    let mut rank_buckets: HashMap<String, OutcomeBucket> = HashMap::new();
    let mut rank_buckets_by_phase: HashMap<Phase, HashMap<String, OutcomeBucket>> = HashMap::new();
    let mut bans_overall = BanCounts::default();
    let mut bans_by_picks_before: HashMap<i32, BanCounts> = HashMap::new();
    let mut bans_post_pick = BanCounts::default();
    let mut skipped = Skipped::default();
    let mut pick_rows: Vec<PickRow> = Vec::new();
    let mut score_auc_rows: Vec<(f64, bool)> = Vec::new();
    let mut percentile_auc_rows: Vec<(f64, bool)> = Vec::new();
    let mut auc_matches = 0_u64;
    let mut verified = 0_usize;
    let mut verify_mismatches = 0_usize;
    let mut ablation_counts: HashMap<Ranking, Counts> = HashMap::new();
    let ablation_modes = [Ranking::Composition, Ranking::MatchupSynergy, Ranking::Meta];
    let mut per_patch_prior: HashMap<String, HashMap<i32, u64>> = HashMap::new();
    let mut per_patch_bans: HashMap<String, HashMap<i32, u64>> = HashMap::new();
    // Ban prevalence fitted on the same warm-up window, plus per-patch counts.
    let mut ban_popularity: HashMap<i32, u64> = HashMap::new();
    for draft in drafts.iter().take(warmup_matches) {
        for action in draft.actions.iter().filter(|action| !action.pick) {
            *ban_popularity.entry(action.hero_id).or_default() += 1;
        }
    }
    let mut ban_engine_eval = Counts::default();
    let mut ban_prior_global_eval = Counts::default();
    let mut ban_prior_patch_eval = Counts::default();
    let mut gate_counts: HashMap<usize, Counts> = HashMap::new();
    let mut only_gate = 0_u64;
    let mut only_prior = 0_u64;
    let mut blend_counts: HashMap<i32, Counts> = HashMap::new();
    // Same sweep split into two evaluation halves, so a weight can be chosen on
    // one half and checked on the other instead of picked in-sample.
    let eval_half_split = warmup_matches + (drafts.len() - warmup_matches) / 2;
    let mut blend_halves: [HashMap<i32, Counts>; 2] = std::array::from_fn(|_| HashMap::new());
    // Paired top-1 comparisons for the best blend weight, counted on the same
    // events: discordant counts feed a McNemar test, not two independent rates.
    // Equal weight is the defensible a-priori choice; the sweep below shows how
    // sensitive the result is to it.
    let best_weight = 50_i32;
    let mut only_blend_vs_engine = 0_u64;
    let mut only_engine = 0_u64;
    let mut only_blend_vs_popularity = 0_u64;
    let mut only_popularity = 0_u64;

    for (index, draft) in drafts.iter().enumerate() {
        let is_eval = index >= warmup_matches;
        engine.reset();
        // Prior for this draft: strictly earlier drafts of the same patch when
        // asked for, otherwise the global warm-up window. Never the future.
        let patch_key = draft.patch.clone().unwrap_or_default();
        let prior_counts = prior_for_draft(
            prior_mode,
            &patch_key,
            &per_patch_prior,
            &popularity,
            PATCH_PRIOR_MIN_PICKS,
        );
        let ban_global = &ban_popularity;
        let ban_patch = prior_for_draft(
            PriorMode::PerPatch,
            &patch_key,
            &per_patch_bans,
            ban_global,
            PATCH_BAN_PRIOR_MIN_BANS,
        );
        let mut pick_ordinal = 0_i32;
        let mut picks_before_ban = 0_i32;
        let mut side_scores: HashMap<TeamSide, f64> = HashMap::new();
        let mut side_percentiles: HashMap<TeamSide, f64> = HashMap::new();
        let mut usable = true;

        for action in &draft.actions {
            let team = if action.radiant {
                TeamSide::Radiant
            } else {
                TeamSide::Dire
            };
            let pool = score_pool(&engine, team);
            let scored = ranked_in(&pool, Ranking::Engine);
            let available = scored.len();
            // Set while sweeping the blend weights, read by the popularity block.
            let mut blend_top1: Option<bool> = None;

            if action.pick {
                pick_ordinal += 1;
                match rank_of(&scored, action.hero_id) {
                    Some((rank, score)) => {
                        let percentile = if available > 1 {
                            1.0 - (rank - 1) as f64 / (available - 1) as f64
                        } else {
                            0.5
                        };
                        let side_won = if action.radiant {
                            draft.radiant_win
                        } else {
                            !draft.radiant_win
                        };
                        // Sanity check: the ranking used here must match the
                        // order the app itself shows on screen.
                        if verified < verify {
                            let top: Vec<i32> = engine
                                .get_recommendations(team, 5)
                                .top_picks
                                .iter()
                                .map(|recommendation| recommendation.hero_id.0)
                                .collect();
                            let ours: Vec<i32> = scored.iter().take(5).map(|(id, _)| *id).collect();
                            if top != ours {
                                verify_mismatches += 1;
                            }
                            verified += 1;
                        }

                        if is_eval {
                            for mode in ablation_modes {
                                let ranked = ranked_in(&pool, mode);
                                if let Some((mode_rank, _)) = rank_of(&ranked, action.hero_id) {
                                    ablation_counts
                                        .entry(mode)
                                        .or_default()
                                        .record(mode_rank, available);
                                }
                            }
                            for weight in BLEND_WEIGHTS {
                                let ranked =
                                    blend_ranking(&pool, &prior_counts, f64::from(weight) / 100.0);
                                if let Some((blend_rank, _)) = rank_of(&ranked, action.hero_id) {
                                    blend_counts
                                        .entry(weight)
                                        .or_default()
                                        .record(blend_rank, available);
                                    let half = usize::from(index >= eval_half_split);
                                    blend_halves[half]
                                        .entry(weight)
                                        .or_default()
                                        .record(blend_rank, available);
                                    if weight == best_weight {
                                        blend_top1 = Some(blend_rank == 1);
                                        if blend_rank == 1 && rank != 1 {
                                            only_blend_vs_engine += 1;
                                        } else if blend_rank != 1 && rank == 1 {
                                            only_engine += 1;
                                        }
                                    }
                                }
                            }
                        }

                        engine_counts.record(rank, available);
                        by_ordinal
                            .entry(pick_ordinal)
                            .or_default()
                            .record(rank, available);
                        by_phase
                            .entry(Phase::of(pick_ordinal))
                            .or_default()
                            .record(rank, available);
                        if is_eval {
                            engine_eval_counts.record(rank, available);
                        }

                        let bucket = match rank {
                            1 => "rank_1",
                            2..=3 => "rank_2_3",
                            4..=10 => "rank_4_10",
                            _ => "rank_gt_10",
                        };
                        rank_buckets
                            .entry(bucket.to_string())
                            .or_default()
                            .record(side_won);
                        rank_buckets_by_phase
                            .entry(Phase::of(pick_ordinal))
                            .or_default()
                            .entry(bucket.to_string())
                            .or_default()
                            .record(side_won);

                        // Popularity baseline over the same available set.
                        if is_eval {
                            let mut freq: Vec<(i32, u64)> = scored
                                .iter()
                                .map(|(id, _)| (*id, prior_counts.get(id).copied().unwrap_or(0)))
                                .collect();
                            freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                            // Gating: the prior's top-K first, engine order inside.
                            let mut gate_top1: Option<bool> = None;
                            for k in GATE_KS {
                                let gated = gated_ranking(&scored, &freq, k);
                                if let Some((gate_rank, _)) = rank_of(&gated, action.hero_id) {
                                    gate_counts
                                        .entry(k)
                                        .or_default()
                                        .record(gate_rank, available);
                                    if k == GATE_DISCORDANCE_K {
                                        gate_top1 = Some(gate_rank == 1);
                                    }
                                }
                            }
                            if let Some(freq_rank) =
                                freq.iter().position(|(id, _)| *id == action.hero_id)
                            {
                                if let Some(gate_hit) = gate_top1 {
                                    let prior_hit = freq_rank == 0;
                                    if gate_hit && !prior_hit {
                                        only_gate += 1;
                                    } else if !gate_hit && prior_hit {
                                        only_prior += 1;
                                    }
                                }
                                if let Some(blend_hit) = blend_top1 {
                                    let freq_hit = freq_rank == 0;
                                    if blend_hit && !freq_hit {
                                        only_blend_vs_popularity += 1;
                                    } else if !blend_hit && freq_hit {
                                        only_popularity += 1;
                                    }
                                }
                                freq_eval.record(freq_rank + 1, available);
                                freq_by_phase
                                    .entry(Phase::of(pick_ordinal))
                                    .or_default()
                                    .record(freq_rank + 1, available);
                            }
                        }

                        *side_scores.entry(team).or_default() += score as f64;
                        *side_percentiles.entry(team).or_default() += percentile;

                        if is_eval && picks_csv.is_some() {
                            let human_freq_rank = {
                                let mut freq: Vec<(i32, u64)> = scored
                                    .iter()
                                    .map(|(id, _)| {
                                        (*id, prior_counts.get(id).copied().unwrap_or(0))
                                    })
                                    .collect();
                                freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                                freq.iter().position(|(id, _)| *id == action.hero_id)
                            };
                            pick_rows.push(PickRow {
                                match_id: draft.match_id,
                                patch: draft.patch.clone().unwrap_or_else(|| "unknown".into()),
                                start_time: draft.start_time.unwrap_or(0),
                                pick_ordinal,
                                side: if action.radiant { "radiant" } else { "dire" }.to_string(),
                                hero_id: action.hero_id,
                                rank,
                                available,
                                score,
                                percentile,
                                won: side_won,
                                human_freq_rank,
                            });
                        }
                    }
                    None => {
                        // The human picked a hero the engine cannot rank (not in
                        // the seeded pool). Keep replaying, but drop the match
                        // from outcome aggregates it cannot contribute to fairly.
                        skipped.picks += 1;
                        usable = false;
                    }
                }
                picks_before_ban += 1;
                engine.add_pick(DraftPick {
                    hero_id: HeroId(action.hero_id),
                    player_id: PlayerId(0),
                    order: action.order,
                    team,
                });
            } else {
                match rank_of(&scored, action.hero_id) {
                    Some((rank, _)) => {
                        bans_overall.bans += 1;
                        if is_eval {
                            ban_engine_eval.record(rank, available);
                            let ids: Vec<i32> = scored.iter().map(|(id, _)| *id).collect();
                            let global_ranked = prior_ranking(ids.iter().copied(), ban_global);
                            if let Some(prior_rank) =
                                position_of_counts(&global_ranked, action.hero_id)
                            {
                                ban_prior_global_eval.record(prior_rank, available);
                            }
                            let patch_ranked = prior_ranking(ids.iter().copied(), &ban_patch);
                            if let Some(prior_rank) =
                                position_of_counts(&patch_ranked, action.hero_id)
                            {
                                ban_prior_patch_eval.record(prior_rank, available);
                            }
                        }
                        if rank == 1 {
                            bans_overall.removes_top1 += 1;
                        }
                        if rank <= 3 {
                            bans_overall.removes_top3 += 1;
                        }
                        let entry = bans_by_picks_before.entry(picks_before_ban).or_default();
                        entry.bans += 1;
                        if rank == 1 {
                            entry.removes_top1 += 1;
                        }
                        if rank <= 3 {
                            entry.removes_top3 += 1;
                        }
                        if picks_before_ban > 0 {
                            bans_post_pick.bans += 1;
                            if rank == 1 {
                                bans_post_pick.removes_top1 += 1;
                            }
                            if rank <= 3 {
                                bans_post_pick.removes_top3 += 1;
                            }
                        }
                    }
                    None => {
                        skipped.bans += 1;
                        usable = false;
                    }
                }
                engine.add_ban(DraftBan {
                    hero_id: HeroId(action.hero_id),
                    order: action.order,
                    team,
                });
            }
        }

        // This draft becomes history for later drafts of the same patch.
        let patch_counts = per_patch_prior.entry(patch_key).or_default();
        for action in draft.actions.iter().filter(|action| action.pick) {
            *patch_counts.entry(action.hero_id).or_default() += 1;
        }
        let patch_ban_counts = per_patch_bans
            .entry(draft.patch.clone().unwrap_or_default())
            .or_default();
        for action in draft.actions.iter().filter(|action| !action.pick) {
            *patch_ban_counts.entry(action.hero_id).or_default() += 1;
        }

        if usable {
            if let (Some(radiant), Some(dire)) = (
                side_scores.get(&TeamSide::Radiant),
                side_scores.get(&TeamSide::Dire),
            ) {
                score_auc_rows.push((radiant - dire, draft.radiant_win));
                auc_matches += 1;
            }
            if let (Some(radiant), Some(dire)) = (
                side_percentiles.get(&TeamSide::Radiant),
                side_percentiles.get(&TeamSide::Dire),
            ) {
                percentile_auc_rows.push((radiant - dire, draft.radiant_win));
            }
        } else {
            skipped.matches_missing_heroes += 1;
        }
    }

    let patches: HashMap<String, usize> = drafts.iter().fold(HashMap::new(), |mut acc, draft| {
        *acc.entry(draft.patch.clone().unwrap_or_else(|| "unknown".to_string()))
            .or_default() += 1;
        acc
    });
    let mut patches: Vec<(String, usize)> = patches.into_iter().collect();
    patches.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let mut by_ordinal: Vec<(i32, Agreement)> = by_ordinal
        .into_iter()
        .map(|(ordinal, counts)| (ordinal, counts.agreement()))
        .collect();
    by_ordinal.sort_by_key(|(ordinal, _)| *ordinal);
    let mut by_phase: Vec<(Phase, Agreement)> = by_phase
        .into_iter()
        .map(|(phase, counts)| (phase, counts.agreement()))
        .collect();
    by_phase.sort_by_key(|(phase, _)| *phase as u8);

    let order = ["rank_1", "rank_2_3", "rank_4_10", "rank_gt_10"];
    let rank_outcome: Vec<(String, OutcomeSummary)> = order
        .iter()
        .filter_map(|name| {
            rank_buckets
                .get(*name)
                .map(|bucket| (name.to_string(), bucket.summary()))
        })
        .collect();
    let mut phases: Vec<Phase> = rank_buckets_by_phase.keys().copied().collect();
    phases.sort_by_key(|phase| *phase as u8);
    let rank_outcome_by_phase: Vec<(Phase, Vec<(String, OutcomeSummary)>)> = phases
        .into_iter()
        .map(|phase| {
            let buckets = rank_buckets_by_phase.get(&phase).expect("phase present");
            let rows = order
                .iter()
                .filter_map(|name| {
                    buckets
                        .get(*name)
                        .map(|bucket| (name.to_string(), bucket.summary()))
                })
                .collect();
            (phase, rows)
        })
        .collect();

    let mut bans_by_picks_before: Vec<(i32, BanRate)> = bans_by_picks_before
        .into_iter()
        .map(|(index, counts)| (index, ban_rate(&counts)))
        .collect();
    bans_by_picks_before.sort_by_key(|(index, _)| *index);

    let report = Report {
        generated_at: chrono::Utc::now().to_rfc3339(),
        db: db_info,
        ml_enabled: engine.ml_enabled,
        prior: prior_mode.label().to_string(),
        ml_model,
        dataset: DatasetInfo {
            drafts: drafts_path.display().to_string(),
            matches_total: drafts.len(),
            matches_evaluated: drafts.len() - warmup_matches,
            matches_available,
            warmup_matches,
            warmup_fraction,
            first_start_time: drafts.first().and_then(|draft| draft.start_time),
            last_start_time: drafts.last().and_then(|draft| draft.start_time),
            patches,
        },
        blend_halves: BLEND_WEIGHTS
            .iter()
            .map(|weight| {
                let half = |index: usize| {
                    blend_halves[index]
                        .get(weight)
                        .cloned()
                        .unwrap_or_default()
                        .agreement()
                };
                (f64::from(*weight) / 100.0, half(0), half(1))
            })
            .collect(),
        gate_sweep: {
            let mut rows: Vec<(usize, Agreement)> = gate_counts
                .iter()
                .map(|(k, counts)| (*k, counts.agreement()))
                .collect();
            rows.sort_by_key(|(k, _)| *k);
            rows
        },
        gate_top1_discordance: GateDiscordance {
            k: GATE_DISCORDANCE_K,
            gate_only: only_gate,
            prior_only: only_prior,
        },
        ban_phase: BanPhase {
            engine_eval: ban_engine_eval.agreement(),
            prior_global_eval: ban_prior_global_eval.agreement(),
            prior_per_patch_eval: ban_prior_patch_eval.agreement(),
        },
        blend_top1_discordance: BlendDiscordance {
            weight: f64::from(best_weight) / 100.0,
            vs_engine: Discordant {
                blend_only: only_blend_vs_engine,
                baseline_only: only_engine,
            },
            vs_popularity: Discordant {
                blend_only: only_blend_vs_popularity,
                baseline_only: only_popularity,
            },
        },
        blend_sweep: {
            let mut rows: Vec<(f64, Agreement)> = blend_counts
                .iter()
                .map(|(weight, counts)| (f64::from(*weight) / 100.0, counts.agreement()))
                .collect();
            rows.sort_by(|a, b| a.0.total_cmp(&b.0));
            rows
        },
        engine_agreement: engine_counts.agreement(),
        engine_agreement_eval: engine_eval_counts.agreement(),
        engine_by_pick_ordinal: by_ordinal,
        engine_by_phase: by_phase,
        frequency_baseline_eval: freq_eval.agreement(),
        frequency_by_phase_eval: {
            let mut rows: Vec<(Phase, Agreement)> = freq_by_phase
                .into_iter()
                .map(|(phase, counts)| (phase, counts.agreement()))
                .collect();
            rows.sort_by_key(|(phase, _)| *phase as u8);
            rows
        },
        skipped,
        rank_outcome,
        rank_outcome_by_phase,
        ban_threat: BansSummary {
            overall: ban_rate(&bans_overall),
            post_pick: ban_rate(&bans_post_pick),
            by_picks_before: bans_by_picks_before,
        },
        team_score_auc: auc(&score_auc_rows),
        team_percentile_auc: auc(&percentile_auc_rows),
        matches_for_auc: auc_matches,
        ablations: ablation_modes
            .iter()
            .map(|mode| {
                (
                    mode.label().to_string(),
                    ablation_counts
                        .get(mode)
                        .cloned()
                        .unwrap_or_default()
                        .agreement(),
                )
            })
            .collect(),
        verify: VerifySummary {
            checked: verified,
            top5_mismatches: verify_mismatches,
        },
    };

    if let Some(path) = picks_csv {
        let mut csv = String::from(
            "match_id,patch,start_time,pick_ordinal,side,hero_id,rank,available,score,percentile,won,human_freq_rank\n",
        );
        for row in &pick_rows {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{:.6},{:.6},{},{}\n",
                row.match_id,
                row.patch,
                row.start_time,
                row.pick_ordinal,
                row.side,
                row.hero_id,
                row.rank,
                row.available,
                row.score,
                row.percentile,
                row.won,
                row.human_freq_rank
                    .map(|rank| rank.to_string())
                    .unwrap_or_default()
            ));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        fs::write(&path, csv).map_err(|error| error.to_string())?;
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let json = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    fs::write(&out_path, format!("{json}\n")).map_err(|error| error.to_string())?;

    println!("{}", json);
    println!("commit: {}", github_commit());
    println!("database copy: {}", copied.display());
    // The copy only exists so the original is never opened for writing.
    let _ = fs::remove_dir_all(&temp_dir);
    Ok(())
}
