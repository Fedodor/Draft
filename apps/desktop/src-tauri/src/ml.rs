//! ML Inference Module
//!
//! ONNX Runtime integration for fast local model inference.
//!
//! The trained XGBoost model is exported to ONNX alongside a `_features.json`
//! file describing the ordered input feature names (896 features in the full
//! pipeline). We reconstruct that feature vector at runtime from the live
//! draft state using the same logic as `ml/feature-engineering/processor.py`.
//!
//! ML is strictly an enhancement layer: if the model is missing, unreadable,
//! or inference fails, every prediction falls back to a neutral 0.5 and the
//! deterministic draft engine continues to work unchanged.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use parking_lot::RwLock;
use tracing::{error, info, warn};

use crate::models::*;

/// Accessor for one scalar semantic attribute of a hero.
type SemanticGetter = fn(&HeroSemantics) -> f32;

/// Feature name prefix helpers (mirror `processor.py` `_safe_names`).
fn safe_hero_name(hero: &Hero) -> String {
    let raw = hero.localized_name.trim();
    if raw.is_empty() {
        format!("hero_{}", hero.id.0)
    } else {
        raw.to_lowercase()
            .replace(' ', "_")
            .replace('\'', "")
            .replace('-', "_")
    }
}

fn safe_name_for(id: HeroId, heroes: &[Hero]) -> String {
    heroes
        .iter()
        .find(|h| h.id == id)
        .map(safe_hero_name)
        .unwrap_or_else(|| format!("hero_{}", id.0))
}

/// Runtime contract exported next to an ONNX model.
///
/// `categoricals` maps a feature name to the exact category order used by the
/// training data. `defaults` provides a value for any feature missing from the
/// live draft instead of silently using 0.0.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct FeatureSchema {
    #[serde(default)]
    pub feature_names: Vec<String>,
    #[serde(default)]
    pub categoricals: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub defaults: HashMap<String, f32>,
    #[serde(default = "default_positive_class_index")]
    pub positive_class_index: usize,
    #[serde(default = "default_input_name")]
    pub input_name: String,
    #[serde(default = "default_probability_output")]
    pub probability_output: String,
}

fn default_positive_class_index() -> usize {
    1
}

fn default_input_name() -> String {
    "float_input".to_string()
}

fn default_probability_output() -> String {
    "probabilities".to_string()
}

impl Default for FeatureSchema {
    fn default() -> Self {
        Self {
            feature_names: Vec::new(),
            categoricals: HashMap::new(),
            defaults: HashMap::new(),
            positive_class_index: default_positive_class_index(),
            input_name: default_input_name(),
            probability_output: default_probability_output(),
        }
    }
}

impl FeatureSchema {
    fn encode_category(&self, feature: &str, value: &str) -> Option<f32> {
        let categories = self.categoricals.get(feature)?;
        categories
            .iter()
            .position(|category| category == value)
            .or_else(|| categories.iter().position(|category| category == "unknown"))
            .map(|index| index as f32)
    }

    fn default_for(&self, feature: &str) -> Option<f32> {
        self.defaults.get(feature).copied()
    }

    fn validate(&self, feature_names: &[String]) -> Result<()> {
        if !self.feature_names.is_empty() && self.feature_names.as_slice() != feature_names {
            return Err(anyhow::anyhow!(
                "Feature schema does not match the model: schema has {} features, model has {}",
                self.feature_names.len(),
                feature_names.len()
            ));
        }
        Ok(())
    }
}

/// ONNX model wrapper. Session is wrapped in a `Mutex` because `ort::Session::run`
/// requires `&mut self` and predictions may be requested concurrently.
pub struct OnnxModel {
    session: Mutex<ort::session::Session>,
    feature_names: Vec<String>,
    schema: Arc<FeatureSchema>,
}

impl OnnxModel {
    /// Load an ONNX model and its accompanying `<stem>_features.json`.
    ///
    /// The ONNX Runtime loader panics if the runtime library is unavailable,
    /// so the body is wrapped in `catch_unwind` to degrade gracefully.
    pub fn load(model_path: &str) -> Result<Self> {
        match std::panic::catch_unwind(|| Self::load_inner(model_path)) {
            Ok(result) => result,
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic while loading ONNX Runtime".to_string());
                Err(anyhow::anyhow!("ONNX Runtime unavailable: {}", msg))
            }
        }
    }

    fn load_inner(model_path: &str) -> Result<Self> {
        let model_file = std::path::Path::new(model_path);
        if !model_file.exists() {
            return Err(anyhow::anyhow!("Model file not found: {}", model_path));
        }

        // Locate the ONNX Runtime DLL BEFORE touching ort: ort's loader aborts
        // the process if the dylib is missing or has an incompatible version,
        // so we only call into it once a compatible runtime is confirmed.
        if let Some(dll) = find_onnx_runtime(model_path) {
            if std::env::var("ORT_DYLIB_PATH")
                .map(|v| v.is_empty())
                .unwrap_or(true)
            {
                std::env::set_var("ORT_DYLIB_PATH", &dll);
            }
        } else {
            return Err(anyhow::anyhow!(
                "ONNX Runtime (onnxruntime.dll 1.28.0) not found near {}",
                model_path
            ));
        }

        let schema_path = derive_schema_path(model_path);
        let schema: FeatureSchema = if schema_path.is_file() {
            let raw = std::fs::read_to_string(&schema_path).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to read schema file {}: {}",
                    schema_path.display(),
                    e
                )
            })?;
            serde_json::from_str(&raw).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to parse schema file {}: {}",
                    schema_path.display(),
                    e
                )
            })?
        } else {
            FeatureSchema::default()
        };

        let features_path = derive_features_path(model_path);
        let feature_names: Vec<String> = if schema.feature_names.is_empty() {
            serde_json::from_str(&std::fs::read_to_string(&features_path).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to read features file {}: {}",
                    features_path.display(),
                    e
                )
            })?)?
        } else {
            schema.feature_names.clone()
        };
        schema.validate(&feature_names)?;

        let session = ort::session::Session::builder()
            .map_err(ort_error)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(ort_error)?
            .with_intra_threads(2)
            .map_err(ort_error)?
            .commit_from_file(model_path)
            .map_err(ort_error)?;

        info!(
            "Loaded ONNX model {} with {} features (features: {}, schema: {})",
            model_path,
            feature_names.len(),
            features_path.display(),
            if schema_path.is_file() {
                schema_path.display().to_string()
            } else {
                "legacy defaults".to_string()
            }
        );

        Ok(Self {
            session: Mutex::new(session),
            feature_names,
            schema: Arc::new(schema),
        })
    }

    pub fn feature_count(&self) -> usize {
        self.feature_names.len()
    }

    pub fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    /// Run inference for a single row built from the given feature map.
    /// Missing features use the schema's training median, then 0.0 only for
    /// legacy bundles without a schema.
    pub fn predict(&self, features: &HashMap<String, f32>) -> Result<f32> {
        let mut vector = Vec::with_capacity(self.feature_names.len());
        for name in &self.feature_names {
            let value = features
                .get(name)
                .copied()
                .or_else(|| self.schema.default_for(name))
                .unwrap_or(0.0);
            vector.push(value);
        }

        let array = ndarray::Array2::from_shape_vec((1, self.feature_names.len()), vector)?;
        let value = ort::value::Value::from_array(array).map_err(ort_error)?;

        let mut session = self.session.lock().unwrap();
        let input_name = if session
            .inputs()
            .iter()
            .any(|input| input.name() == self.schema.input_name)
        {
            self.schema.input_name.clone()
        } else {
            session
                .inputs()
                .first()
                .map(|input| input.name().to_string())
                .unwrap_or_else(|| "input".to_string())
        };
        let outputs = session.run(vec![(input_name, value)]).map_err(ort_error)?;

        // `probabilities` is [batch, class]. The positive class is P(Radiant
        // wins); using the first f32 would return P(Dire wins) for binary models.
        let mut prediction = outputs
            .get(&self.schema.probability_output)
            .and_then(|output| output.try_extract_tensor::<f32>().ok())
            .and_then(|(_shape, data)| select_probability(data, self.schema.positive_class_index));

        if prediction.is_none() {
            prediction = outputs.values().find_map(|output| {
                output
                    .try_extract_tensor::<f32>()
                    .ok()
                    .and_then(|(_shape, data)| {
                        select_probability(data, self.schema.positive_class_index)
                    })
            });
        }

        let pred = prediction.unwrap_or(0.5);
        Ok(pred.clamp(0.0, 1.0))
    }
}

/// Select the positive-class probability from a flat output tensor.
///
/// Falls back to the first value for single-class or legacy outputs, but never
/// suggests the negative class just because it appears first.
fn select_probability(data: &[f32], positive_class_index: usize) -> Option<f32> {
    data.get(positive_class_index)
        .copied()
        .or_else(|| data.first().copied())
}

/// Convert an `ort` error into a `Send + Sync` anyhow error.
fn ort_error<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("{}", e)
}

/// Derive the `<stem>_features.json` path from an `.onnx` model path.
fn derive_features_path(model_path: &str) -> std::path::PathBuf {
    derive_sidecar_path(model_path, "features.json")
}

/// Derive the `<stem>_schema.json` path from an `.onnx` model path.
fn derive_schema_path(model_path: &str) -> std::path::PathBuf {
    derive_sidecar_path(model_path, "schema.json")
}

fn derive_sidecar_path(model_path: &str, suffix: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(model_path);
    let stem = p
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    match p.parent() {
        Some(dir) => dir.join(format!("{stem}_{suffix}")),
        None => std::path::PathBuf::from(format!("{stem}_{suffix}")),
    }
}

/// Locate a compatible ONNX Runtime DLL without invoking ort.
/// Respects an existing `ORT_DYLIB_PATH`, then searches next to the model.
fn find_onnx_runtime(model_path: &str) -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        let path = std::path::PathBuf::from(&p);
        if !p.is_empty() && path.exists() {
            return Some(path);
        }
    }

    let model_file = std::path::Path::new(model_path);
    let model_dir = model_file.parent()?;
    let mut candidates = vec![
        model_dir.join("onnxruntime.dll"),
        model_dir.join("resources").join("onnxruntime.dll"),
    ];
    if let Some(parent) = model_dir.parent() {
        candidates.push(parent.join("onnxruntime.dll"));
        candidates.push(parent.join("resources").join("onnxruntime.dll"));
    }
    candidates.into_iter().find(|p| p.exists())
}

/// ML Inference Engine for draft value prediction.
///
/// Holds an optional loaded `OnnxModel`; without one, all predictions are the
/// neutral 0.5 (deterministic engine remains fully functional).
pub struct MLInferenceEngine {
    model: Arc<RwLock<Option<Arc<OnnxModel>>>>,
    model_path: Option<String>,
}

impl Default for MLInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MLInferenceEngine {
    pub fn new() -> Self {
        Self {
            model: Arc::new(RwLock::new(None)),
            model_path: None,
        }
    }

    pub fn model_path(&self) -> Option<&str> {
        self.model_path.as_deref()
    }

    pub fn is_loaded(&self) -> bool {
        self.model.read().is_some()
    }

    pub fn feature_count(&self) -> usize {
        self.model
            .read()
            .as_ref()
            .map(|m| m.feature_count())
            .unwrap_or(0)
    }

    pub fn feature_schema(&self) -> Option<Arc<FeatureSchema>> {
        self.model.read().as_ref().map(|model| model.schema.clone())
    }

    pub fn status(&self) -> MLStatus {
        MLStatus {
            loaded: self.is_loaded(),
            feature_count: self.feature_count(),
            model_path: self.model_path.clone(),
        }
    }

    /// Attempt to load an ONNX model. On any failure the engine logs a warning
    /// and continues in neutral mode (no hard dependency on the model).
    pub fn load_model(&mut self, model_path: &str) -> Result<()> {
        match OnnxModel::load(model_path) {
            Ok(model) => {
                *self.model.write() = Some(Arc::new(model));
                self.model_path = Some(model_path.to_string());
                info!("ML model loaded: {}", model_path);
                Ok(())
            }
            Err(e) => {
                warn!(
                    "ML model load failed ({}); continuing with neutral predictions",
                    e
                );
                Err(e)
            }
        }
    }

    /// Predict the probability that the Radiant team wins given a feature map.
    /// Returns the neutral 0.5 if no model is loaded or inference fails.
    pub fn predict(&self, features: &HashMap<String, f32>) -> Result<f32> {
        let model = match self.model.read().as_ref() {
            Some(m) => m.clone(),
            None => return Ok(0.5),
        };

        match model.predict(features) {
            Ok(p) => Ok(p),
            Err(e) => {
                error!("ML inference failed: {}", e);
                Ok(0.5)
            }
        }
    }
}

/// Build the ML feature map from the current draft state, mirroring
/// `processor.py` `FeatureProcessor::process_match` (minus match-only fields
/// such as `duration`/`game_mode`, which default to their training medians).
pub fn build_feature_map(
    state: &DraftState,
    heroes: &[Hero],
    semantics: &HashMap<HeroId, HeroSemantics>,
    meta: &MetaData,
    matchups: &HashMap<HeroId, Vec<HeroMatchup>>,
    schema: Option<&FeatureSchema>,
) -> HashMap<String, f32> {
    let mut f = HashMap::new();

    let rad_pick_ids: std::collections::HashSet<i32> =
        state.radiant_picks.iter().map(|p| p.hero_id.0).collect();
    let dire_pick_ids: std::collections::HashSet<i32> =
        state.dire_picks.iter().map(|p| p.hero_id.0).collect();
    let rad_ban_ids: std::collections::HashSet<i32> = state
        .bans
        .iter()
        .filter(|b| b.team == TeamSide::Radiant)
        .map(|b| b.hero_id.0)
        .collect();
    let dire_ban_ids: std::collections::HashSet<i32> = state
        .bans
        .iter()
        .filter(|b| b.team == TeamSide::Dire)
        .map(|b| b.hero_id.0)
        .collect();

    // Presence + bans (one-hot per known hero).
    for hero in heroes {
        let sn = safe_hero_name(hero);
        f.insert(
            format!("rad_has_{sn}"),
            if rad_pick_ids.contains(&hero.id.0) {
                1.0
            } else {
                0.0
            },
        );
        f.insert(
            format!("dire_has_{sn}"),
            if dire_pick_ids.contains(&hero.id.0) {
                1.0
            } else {
                0.0
            },
        );
        f.insert(
            format!("rad_ban_{sn}"),
            if rad_ban_ids.contains(&hero.id.0) {
                1.0
            } else {
                0.0
            },
        );
        f.insert(
            format!("dire_ban_{sn}"),
            if dire_ban_ids.contains(&hero.id.0) {
                1.0
            } else {
                0.0
            },
        );
    }

    // Pick order.
    for (i, pick) in state.radiant_picks.iter().take(5).enumerate() {
        f.insert(format!("rad_pick_{i}"), pick.hero_id.0 as f32);
    }
    for (i, pick) in state.dire_picks.iter().take(5).enumerate() {
        f.insert(format!("dire_pick_{i}"), pick.hero_id.0 as f32);
    }

    // Semantic team aggregates.
    let sem_keys: [(&str, SemanticGetter); 17] = [
        ("stun", |s| s.stun),
        ("initiation", |s| s.initiation),
        ("save", |s| s.save),
        ("push", |s| s.push),
        ("scaling", |s| s.scaling),
        ("mobility", |s| s.mobility),
        ("durability", |s| s.durability),
        ("teamfight", |s| s.teamfight),
        ("splitpush", |s| s.splitpush),
        ("vision", |s| s.vision),
        ("magic_damage", |s| s.magic_damage),
        ("physical_damage", |s| s.physical_damage),
        ("pure_damage", |s| s.pure_damage),
        ("disable_count", |s| s.disable_count),
        ("lane_early", |s| s.lane_strength.early),
        ("lane_mid", |s| s.lane_strength.mid),
        ("lane_late", |s| s.lane_strength.late),
    ];

    for (team_name, picks) in [
        ("radiant", &state.radiant_picks),
        ("dire", &state.dire_picks),
    ] {
        let n = picks.len().max(1) as f32;
        for (key, getter) in sem_keys {
            let total: f32 = picks
                .iter()
                .map(|p| semantics.get(&p.hero_id).map(&getter).unwrap_or(0.5))
                .sum();
            f.insert(format!("{team_name}_avg_{key}"), total / n);
        }
    }

    // Semantic differentials.
    for key in [
        "stun",
        "initiation",
        "save",
        "push",
        "scaling",
        "durability",
        "teamfight",
        "magic_damage",
        "physical_damage",
        "disable_count",
    ] {
        let rad = f.get(&format!("radiant_avg_{key}")).copied().unwrap_or(0.5);
        let dire = f.get(&format!("dire_avg_{key}")).copied().unwrap_or(0.5);
        f.insert(format!("{key}_diff"), rad - dire);
    }

    // Per-hero winrates + aggregates.
    let mut rad_wrs = Vec::new();
    let mut dire_wrs = Vec::new();
    for pick in &state.radiant_picks {
        let sn = safe_name_for(pick.hero_id.clone(), heroes);
        let wr = meta
            .hero_winrates
            .get(&pick.hero_id)
            .copied()
            .unwrap_or(0.5);
        f.insert(format!("rad_wr_{sn}"), wr);
        rad_wrs.push(wr);
    }
    for pick in &state.dire_picks {
        let sn = safe_name_for(pick.hero_id.clone(), heroes);
        let wr = meta
            .hero_winrates
            .get(&pick.hero_id)
            .copied()
            .unwrap_or(0.5);
        f.insert(format!("dire_wr_{sn}"), wr);
        dire_wrs.push(wr);
    }

    if !rad_wrs.is_empty() {
        let sum: f32 = rad_wrs.iter().sum();
        let n = rad_wrs.len() as f32;
        f.insert("radiant_avg_winrate".into(), sum / n);
        f.insert(
            "radiant_max_winrate".into(),
            rad_wrs.iter().copied().fold(0.0_f32, f32::max),
        );
        f.insert(
            "radiant_min_winrate".into(),
            rad_wrs.iter().copied().fold(1.0_f32, f32::min),
        );
    }
    if !dire_wrs.is_empty() {
        let sum: f32 = dire_wrs.iter().sum();
        let n = dire_wrs.len() as f32;
        f.insert("dire_avg_winrate".into(), sum / n);
        f.insert(
            "dire_max_winrate".into(),
            dire_wrs.iter().copied().fold(0.0_f32, f32::max),
        );
        f.insert(
            "dire_min_winrate".into(),
            dire_wrs.iter().copied().fold(1.0_f32, f32::min),
        );
    }
    if !rad_wrs.is_empty() && !dire_wrs.is_empty() {
        let rad_avg = rad_wrs.iter().sum::<f32>() / rad_wrs.len() as f32;
        let dire_avg = dire_wrs.iter().sum::<f32>() / dire_wrs.len() as f32;
        f.insert("winrate_diff".into(), rad_avg - dire_avg);
    }

    let rad_ban_wrs: Vec<f32> = rad_ban_ids
        .iter()
        .map(|id| meta.hero_winrates.get(&HeroId(*id)).copied().unwrap_or(0.5))
        .collect();
    let dire_ban_wrs: Vec<f32> = dire_ban_ids
        .iter()
        .map(|id| meta.hero_winrates.get(&HeroId(*id)).copied().unwrap_or(0.5))
        .collect();
    if !rad_ban_wrs.is_empty() {
        f.insert(
            "radiant_ban_avg_winrate".into(),
            rad_ban_wrs.iter().sum::<f32>() / rad_ban_wrs.len() as f32,
        );
    }
    if !dire_ban_wrs.is_empty() {
        f.insert(
            "dire_ban_avg_winrate".into(),
            dire_ban_wrs.iter().sum::<f32>() / dire_ban_wrs.len() as f32,
        );
    }

    // Hero-vs-hero matchups.
    let mut rad_matchup_wrs = Vec::new();
    let mut dire_matchup_wrs = Vec::new();
    for rad_pick in &state.radiant_picks {
        let mut vs_dire = Vec::new();
        for dire_pick in &state.dire_picks {
            let wr = matchup_winrate(matchups, &rad_pick.hero_id, &dire_pick.hero_id);
            vs_dire.push(wr);
        }
        if !vs_dire.is_empty() {
            rad_matchup_wrs.push(vs_dire.iter().sum::<f32>() / vs_dire.len() as f32);
        }
    }
    for dire_pick in &state.dire_picks {
        let mut vs_rad = Vec::new();
        for rad_pick in &state.radiant_picks {
            let wr = matchup_winrate(matchups, &dire_pick.hero_id, &rad_pick.hero_id);
            vs_rad.push(wr);
        }
        if !vs_rad.is_empty() {
            dire_matchup_wrs.push(vs_rad.iter().sum::<f32>() / vs_rad.len() as f32);
        }
    }

    if !rad_matchup_wrs.is_empty() {
        let sum: f32 = rad_matchup_wrs.iter().sum();
        let n = rad_matchup_wrs.len() as f32;
        f.insert("rad_matchup_avg".into(), sum / n);
        f.insert(
            "rad_matchup_best".into(),
            rad_matchup_wrs.iter().copied().fold(0.0_f32, f32::max),
        );
        f.insert(
            "rad_matchup_worst".into(),
            rad_matchup_wrs.iter().copied().fold(1.0_f32, f32::min),
        );
    }
    if !dire_matchup_wrs.is_empty() {
        let sum: f32 = dire_matchup_wrs.iter().sum();
        let n = dire_matchup_wrs.len() as f32;
        f.insert("dire_matchup_avg".into(), sum / n);
        f.insert(
            "dire_matchup_best".into(),
            dire_matchup_wrs.iter().copied().fold(0.0_f32, f32::max),
        );
        f.insert(
            "dire_matchup_worst".into(),
            dire_matchup_wrs.iter().copied().fold(1.0_f32, f32::min),
        );
    }
    if !rad_matchup_wrs.is_empty() && !dire_matchup_wrs.is_empty() {
        let rad_avg = rad_matchup_wrs.iter().sum::<f32>() / rad_matchup_wrs.len() as f32;
        let dire_avg = dire_matchup_wrs.iter().sum::<f32>() / dire_matchup_wrs.len() as f32;
        f.insert("matchup_advantage".into(), rad_avg - dire_avg);
    }

    // Synergies (product of teammate winrates).
    let rad_hero_ids: Vec<i32> = state.radiant_picks.iter().map(|p| p.hero_id.0).collect();
    if rad_hero_ids.len() >= 2 {
        let mut scores = Vec::new();
        for i in 0..rad_hero_ids.len() {
            for j in (i + 1)..rad_hero_ids.len() {
                let w1 = meta
                    .hero_winrates
                    .get(&HeroId(rad_hero_ids[i]))
                    .copied()
                    .unwrap_or(0.5);
                let w2 = meta
                    .hero_winrates
                    .get(&HeroId(rad_hero_ids[j]))
                    .copied()
                    .unwrap_or(0.5);
                scores.push(w1 * w2);
            }
        }
        if !scores.is_empty() {
            f.insert(
                "rad_synergy_avg".into(),
                scores.iter().sum::<f32>() / scores.len() as f32,
            );
            f.insert(
                "rad_synergy_max".into(),
                scores.iter().copied().fold(0.0_f32, f32::max),
            );
            f.insert(
                "rad_synergy_min".into(),
                scores.iter().copied().fold(1.0_f32, f32::min),
            );
        }
    }
    let dire_hero_ids: Vec<i32> = state.dire_picks.iter().map(|p| p.hero_id.0).collect();
    if dire_hero_ids.len() >= 2 {
        let mut scores = Vec::new();
        for i in 0..dire_hero_ids.len() {
            for j in (i + 1)..dire_hero_ids.len() {
                let w1 = meta
                    .hero_winrates
                    .get(&HeroId(dire_hero_ids[i]))
                    .copied()
                    .unwrap_or(0.5);
                let w2 = meta
                    .hero_winrates
                    .get(&HeroId(dire_hero_ids[j]))
                    .copied()
                    .unwrap_or(0.5);
                scores.push(w1 * w2);
            }
        }
        if !scores.is_empty() {
            f.insert(
                "dire_synergy_avg".into(),
                scores.iter().sum::<f32>() / scores.len() as f32,
            );
            f.insert(
                "dire_synergy_max".into(),
                scores.iter().copied().fold(0.0_f32, f32::max),
            );
            f.insert(
                "dire_synergy_min".into(),
                scores.iter().copied().fold(1.0_f32, f32::min),
            );
        }
    }

    // Damage diversity (max - min of magic/physical per team).
    if state.radiant_picks.len() >= 2 {
        let mut attrs = Vec::new();
        for p in &state.radiant_picks {
            if let Some(sem) = semantics.get(&p.hero_id) {
                attrs.push(sem.magic_damage);
                attrs.push(sem.physical_damage);
            }
        }
        if !attrs.is_empty() {
            let max = attrs.iter().copied().fold(0.0_f32, f32::max);
            let min = attrs.iter().copied().fold(1.0_f32, f32::min);
            f.insert("radiant_damage_diversity".into(), max - min);
        }
    }
    if state.dire_picks.len() >= 2 {
        let mut attrs = Vec::new();
        for p in &state.dire_picks {
            if let Some(sem) = semantics.get(&p.hero_id) {
                attrs.push(sem.magic_damage);
                attrs.push(sem.physical_damage);
            }
        }
        if !attrs.is_empty() {
            let max = attrs.iter().copied().fold(0.0_f32, f32::max);
            let min = attrs.iter().copied().fold(1.0_f32, f32::min);
            f.insert("dire_damage_diversity".into(), max - min);
        }
    }

    // Match-only fields use the model schema's training medians. The
    // hardcoded values remain only as a fallback for legacy model bundles.
    let default_value = |name: &str, fallback: f32| {
        schema
            .and_then(|schema| schema.default_for(name))
            .unwrap_or(fallback)
    };
    f.insert("duration".into(), default_value("duration", 2163.0));
    f.insert("avg_mmr".into(), default_value("avg_mmr", 0.0));
    f.insert("game_mode".into(), default_value("game_mode", 22.0));
    f.insert("lobby_type".into(), default_value("lobby_type", 0.0));

    let patch = if state.patch_version.trim().is_empty() {
        meta.patch_version.trim()
    } else {
        state.patch_version.trim()
    };
    let patch_code = schema
        .and_then(|schema| schema.encode_category("patch", patch))
        .or_else(|| schema.and_then(|schema| schema.default_for("patch")))
        .unwrap_or(0.0);
    let patch_version_code = schema
        .and_then(|schema| schema.encode_category("patch_version", patch))
        .or_else(|| schema.and_then(|schema| schema.default_for("patch_version")))
        .unwrap_or(0.0);
    f.insert("patch".into(), patch_code);
    f.insert("patch_version".into(), patch_version_code);

    f
}

fn matchup_winrate(
    matchups: &HashMap<HeroId, Vec<HeroMatchup>>,
    hero: &HeroId,
    vs: &HeroId,
) -> f32 {
    matchups
        .get(hero)
        .and_then(|list| list.iter().find(|m| m.vs_hero_id == *vs))
        .map(|m| m.winrate)
        .unwrap_or(0.5)
}

/// ML engine status exposed to commands/frontend
#[derive(Debug, Clone, serde::Serialize)]
pub struct MLStatus {
    pub loaded: bool,
    pub feature_count: usize,
    pub model_path: Option<String>,
}

/// Global ML inference instance
static ML_ENGINE: once_cell::sync::OnceCell<Arc<RwLock<MLInferenceEngine>>> =
    once_cell::sync::OnceCell::new();

pub fn get_ml_engine() -> Arc<RwLock<MLInferenceEngine>> {
    ML_ENGINE
        .get_or_init(|| Arc::new(RwLock::new(MLInferenceEngine::new())))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_heroes() -> Vec<Hero> {
        let mut h = Hero {
            id: HeroId(1),
            name: "npc_dota_hero_axe".to_string(),
            localized_name: "Axe".to_string(),
            primary_attr: PrimaryAttribute::Str,
            attack_type: AttackType::Melee,
            roles: vec!["initiator".to_string(), "durable".to_string()],
            legs: 2,
            img: String::new(),
            icon: String::new(),
            base_health: 200,
            base_mana: 75,
            base_armor: 2.0,
            base_attack_min: 27,
            base_attack_max: 33,
            base_str: 25.0,
            base_agi: 20.0,
            base_int: 18.0,
            str_gain: 3.2,
            agi_gain: 2.0,
            int_gain: 1.8,
            attack_range: 150,
            projectile_speed: 0,
            attack_rate: 1.7,
            move_speed: 310,
            turn_rate: 0.6,
            cm_enabled: true,
        };
        // Shadow Fiend id=11, Vengeful Spirit id=20
        h.id = HeroId(11);
        h.name = "npc_dota_hero_nevermore".to_string();
        h.localized_name = "Shadow Fiend".to_string();
        h.primary_attr = PrimaryAttribute::Agi;
        h.attack_type = AttackType::Ranged;
        let sf = h.clone();

        h.id = HeroId(20);
        h.name = "npc_dota_hero_vengefulspirit".to_string();
        h.localized_name = "Vengeful Spirit".to_string();
        h.primary_attr = PrimaryAttribute::Agi;
        let vs = h;

        vec![
            Hero {
                id: HeroId(1),
                name: "npc_dota_hero_axe".to_string(),
                localized_name: "Axe".to_string(),
                ..sf.clone()
            },
            sf,
            vs,
        ]
    }

    fn sample_semantics() -> HashMap<HeroId, HeroSemantics> {
        let sem = |id: i32| HeroSemantics {
            hero_id: HeroId(id),
            initiation: 0.5,
            stun: 0.5,
            save: 0.5,
            push: 0.5,
            scaling: 0.5,
            mobility: 0.5,
            durability: 0.5,
            teamfight: 0.5,
            splitpush: 0.5,
            vision: 0.5,
            magic_damage: 0.5,
            physical_damage: 0.5,
            pure_damage: 0.0,
            disable_count: 1.0,
            lane_strength: LaneStrength {
                early: 0.5,
                mid: 0.5,
                late: 0.5,
            },
            tempo_profile: TempoProfile::Mid,
            roles_probability: HashMap::new(),
        };
        let mut m = HashMap::new();
        m.insert(HeroId(1), sem(1));
        m.insert(HeroId(11), sem(11));
        m.insert(HeroId(20), sem(20));
        m
    }

    #[test]
    fn test_safe_hero_names_match_model_convention() {
        let heroes = sample_heroes();
        assert_eq!(safe_hero_name(&heroes[1]), "shadow_fiend");
        assert_eq!(safe_hero_name(&heroes[2]), "vengeful_spirit");
    }

    #[test]
    fn test_build_feature_map_produces_expected_keys() {
        let heroes = sample_heroes();
        let semantics = sample_semantics();

        let mut state = DraftState::default();
        state.radiant_picks.push(DraftPick {
            hero_id: HeroId(1),
            player_id: PlayerId(0),
            order: 1,
            team: TeamSide::Radiant,
        });
        state.radiant_picks.push(DraftPick {
            hero_id: HeroId(11),
            player_id: PlayerId(0),
            order: 3,
            team: TeamSide::Radiant,
        });
        state.dire_picks.push(DraftPick {
            hero_id: HeroId(20),
            player_id: PlayerId(0),
            order: 2,
            team: TeamSide::Dire,
        });
        state.bans.push(DraftBan {
            hero_id: HeroId(20),
            order: 1,
            team: TeamSide::Dire,
        });

        let meta = MetaData {
            patch_version: "7.41".to_string(),
            hero_meta_scores: HashMap::new(),
            hero_pickrates: HashMap::new(),
            hero_winrates: [(HeroId(1), 0.53), (HeroId(11), 0.51), (HeroId(20), 0.5)]
                .into_iter()
                .collect(),
            updated_at: 0,
        };
        let matchups = HashMap::new();

        let map = build_feature_map(&state, &heroes, &semantics, &meta, &matchups, None);

        assert_eq!(map.get("rad_has_axe"), Some(&1.0));
        assert_eq!(map.get("dire_has_axe"), Some(&0.0));
        assert_eq!(map.get("dire_has_vengeful_spirit"), Some(&1.0));
        assert_eq!(map.get("dire_ban_vengeful_spirit"), Some(&1.0));
        assert_eq!(map.get("rad_pick_0"), Some(&1.0));
        assert_eq!(map.get("rad_pick_1"), Some(&11.0));
        assert_eq!(map.get("dire_pick_0"), Some(&20.0));
        assert_eq!(map.get("rad_wr_axe"), Some(&0.53));
        assert_eq!(map.get("radiant_avg_winrate"), Some(&0.52));
        assert!(map.contains_key("matchup_advantage"));
        assert!(map.contains_key("radiant_avg_stun"));
        assert!(map.contains_key("stun_diff"));
        assert!(map.contains_key("duration"));
    }

    #[test]
    fn test_predict_returns_neutral_without_model() {
        let engine = MLInferenceEngine::new();
        let map = HashMap::new();
        assert!(!engine.is_loaded());
        assert_eq!(engine.predict(&map).unwrap(), 0.5);
    }

    #[test]
    fn test_select_probability_uses_positive_class() {
        assert_eq!(select_probability(&[0.425, 0.575], 1), Some(0.575));
        assert_eq!(select_probability(&[0.575], 1), Some(0.575));
        assert_eq!(select_probability(&[], 1), None);
    }

    #[test]
    fn test_schema_encodes_categories_and_defaults() {
        let schema = FeatureSchema {
            categoricals: [
                (
                    "patch".to_string(),
                    vec!["7.40".to_string(), "7.41".to_string()],
                ),
                (
                    "patch_version".to_string(),
                    vec![
                        "7.40".to_string(),
                        "unknown".to_string(),
                        "7.41".to_string(),
                    ],
                ),
            ]
            .into_iter()
            .collect(),
            defaults: [
                ("duration".to_string(), 2163.0),
                ("patch".to_string(), 1.0),
                ("patch_version".to_string(), 2.0),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };

        let state = DraftState {
            patch_version: "7.41".to_string(),
            ..Default::default()
        };
        let meta = MetaData {
            patch_version: "7.40".to_string(),
            hero_meta_scores: HashMap::new(),
            hero_pickrates: HashMap::new(),
            hero_winrates: HashMap::new(),
            updated_at: 0,
        };

        let map = build_feature_map(
            &state,
            &[],
            &HashMap::new(),
            &meta,
            &HashMap::new(),
            Some(&schema),
        );

        assert_eq!(map.get("patch"), Some(&1.0));
        assert_eq!(map.get("patch_version"), Some(&2.0));
        assert_eq!(map.get("duration"), Some(&2163.0));
    }

    #[test]
    fn test_schema_uses_unknown_category_when_patch_is_missing() {
        let schema = FeatureSchema {
            categoricals: [(
                "patch_version".to_string(),
                vec![
                    "7.40".to_string(),
                    "unknown".to_string(),
                    "7.41".to_string(),
                ],
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        };

        assert_eq!(schema.encode_category("patch_version", "7.99"), Some(1.0));
        assert_eq!(schema.encode_category("patch", "7.99"), None);
    }

    #[test]
    fn test_schema_rejects_feature_order_mismatch() {
        let schema = FeatureSchema {
            feature_names: vec!["patch".to_string(), "duration".to_string()],
            ..Default::default()
        };

        assert!(schema
            .validate(&["duration".to_string(), "patch".to_string()])
            .is_err());
        assert!(schema.validate(&["patch".to_string()]).is_err());
        assert!(schema
            .validate(&[
                "patch".to_string(),
                "duration".to_string(),
                "extra".to_string()
            ])
            .is_err());
    }

    #[test]
    fn test_exported_schema_matches_model_contract() {
        let schema_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../ml/models/draft_model_xgboost_full_schema.json");
        let raw = std::fs::read_to_string(&schema_path).unwrap();
        let schema: FeatureSchema = serde_json::from_str(&raw).unwrap();

        assert_eq!(schema.feature_names.len(), 887);
        assert_eq!(schema.encode_category("patch", "7.41"), Some(24.0));
        assert_eq!(schema.encode_category("patch_version", "7.41"), Some(1.0));
        assert_eq!(schema.default_for("duration"), Some(2163.0));
        assert_eq!(schema.positive_class_index, 1);
    }

    #[test]
    fn test_unknown_hero_uses_stable_feature_placeholder() {
        let state = DraftState {
            radiant_picks: vec![DraftPick {
                hero_id: HeroId(999),
                player_id: PlayerId(0),
                order: 1,
                team: TeamSide::Radiant,
            }],
            ..Default::default()
        };
        let meta = MetaData {
            patch_version: "7.41".to_string(),
            hero_meta_scores: HashMap::new(),
            hero_pickrates: HashMap::new(),
            hero_winrates: HashMap::new(),
            updated_at: 0,
        };

        let map = build_feature_map(&state, &[], &HashMap::new(), &meta, &HashMap::new(), None);

        assert_eq!(map.get("rad_wr_hero_999"), Some(&0.5));
        assert_eq!(safe_name_for(HeroId(999), &[]), "hero_999");
    }

    #[test]
    fn test_golden_vector_matches_bundled_onnx_model() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let dll = manifest.join("resources/onnxruntime.dll");
        if !dll.exists() {
            eprintln!("skipping ONNX golden test: onnxruntime.dll not bundled");
            return;
        }
        std::env::set_var("ORT_DYLIB_PATH", &dll);

        let model_path = manifest.join("../../../ml/models/draft_model_xgboost_full.onnx");
        let fixture_path = manifest.join("../../../ml/tests/fixtures/golden_draft_features.json");
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&fixture_path).unwrap()).unwrap();
        let feature_names: Vec<String> =
            serde_json::from_value(fixture["feature_names"].clone()).unwrap();
        let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).unwrap();
        assert_eq!(feature_names.len(), input.len());

        let features: HashMap<String, f32> = feature_names.into_iter().zip(input).collect();
        let expected = fixture["expected"]["positive_probability"]
            .as_f64()
            .unwrap() as f32;

        let model = OnnxModel::load(model_path.to_str().unwrap()).unwrap();
        let prediction = model.predict(&features).unwrap();
        assert!(
            (prediction - expected).abs() < 1e-5,
            "expected {expected}, got {prediction}"
        );
    }
}
