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
use smart_draft_assistant::models::{DraftBan, DraftPick, HeroId, PlayerId, TeamSide};

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

/// Every available hero scored for `team`, best first.
fn ranked_scores(engine: &DraftEngine, team: TeamSide) -> Vec<(i32, f32)> {
    let available = engine.get_state().available_heroes;
    let mut scored: Vec<(i32, f32)> = available
        .iter()
        .filter_map(|hero| {
            engine
                .score_breakdown(hero, team)
                .map(|breakdown| (hero.0, breakdown.total))
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored
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
    engine.set_ml_enabled(false);

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

    for (index, draft) in drafts.iter().enumerate() {
        let is_eval = index >= warmup_matches;
        engine.reset();
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
            let scored = ranked_scores(&engine, team);
            let available = scored.len();

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
                                .map(|(id, _)| (*id, popularity.get(id).copied().unwrap_or(0)))
                                .collect();
                            freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                            if let Some(freq_rank) =
                                freq.iter().position(|(id, _)| *id == action.hero_id)
                            {
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
                                    .map(|(id, _)| (*id, popularity.get(id).copied().unwrap_or(0)))
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
