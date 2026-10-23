//! End-to-end inference check: loads the real ONNX model and runs a prediction.
//!
//! Usage:
//!   $env:ORT_DYLIB_PATH = "C:\path\to\onnxruntime.dll"
//!   cargo run --example inference_check -- <path-to-model.onnx>
//!
//! The schema must sit next to the model as `<stem>_schema.json`.

use smart_draft_assistant::ml::{build_feature_map, OnnxModel};
use smart_draft_assistant::models::*;
use std::collections::HashMap;

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../../../ml/models/draft_model_xgboost_full.onnx".to_string());

    // If ORT_DYLIB_PATH isn't set, point it at the bundled runtime next to the app.
    if std::env::var("ORT_DYLIB_PATH")
        .map(|v| v.is_empty())
        .unwrap_or(true)
    {
        for candidate in [
            "resources/onnxruntime.dll",
            "../../../apps/desktop/src-tauri/resources/onnxruntime.dll",
        ] {
            if std::path::Path::new(candidate).exists() {
                std::env::set_var("ORT_DYLIB_PATH", candidate);
                break;
            }
        }
    }

    match OnnxModel::load(&model_path) {
        Ok(model) => {
            println!("Model loaded: {} features", model.feature_count());

            let heroes = vec![
                Hero {
                    id: HeroId(1),
                    name: "npc_dota_hero_axe".to_string(),
                    localized_name: "Axe".to_string(),
                    primary_attr: PrimaryAttribute::Str,
                    attack_type: AttackType::Melee,
                    roles: vec!["initiator".into(), "durable".into()],
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
                },
                Hero {
                    id: HeroId(11),
                    name: "npc_dota_hero_nevermore".to_string(),
                    localized_name: "Shadow Fiend".to_string(),
                    primary_attr: PrimaryAttribute::Agi,
                    attack_type: AttackType::Ranged,
                    roles: vec!["carry".into(), "nuker".into()],
                    legs: 0,
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
                },
                Hero {
                    id: HeroId(20),
                    name: "npc_dota_hero_vengefulspirit".to_string(),
                    localized_name: "Vengeful Spirit".to_string(),
                    primary_attr: PrimaryAttribute::Agi,
                    attack_type: AttackType::Ranged,
                    roles: vec!["support".into(), "initiator".into()],
                    legs: 0,
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
                },
            ];

            let semantics = HashMap::new();
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

            let map = build_feature_map(
                &state,
                &heroes,
                &semantics,
                &meta,
                &matchups,
                Some(model.schema()),
            );
            println!("Built feature map with {} values", map.len());

            let prediction = model.predict(&map).expect("prediction failed");
            println!("Radiant win probability: {:.4}", prediction);
        }
        Err(e) => {
            eprintln!("Model load failed: {}", e);
            eprintln!("(set ORT_DYLIB_PATH to an onnxruntime.dll and retry)");
        }
    }
}
