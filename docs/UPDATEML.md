FULL DESIGN: REAL-TIME DRAFT AI SYSTEM (Dota 2)

Система для Dota 2 должна состоять из 3 уровней:

🧩 1. FEATURE + STATE LAYER

Ты превращаешь драфт в состояние:

State S =
(radiant_picks,
 dire_picks,
 bans,
 available_heroes,
 patch,
 lane context)
Feature generation включает:
hero embeddings (dense vectors)
synergy scores
counter scores
lane strength
tempo estimation
meta strength
⚙️ 2. VALUE MODEL (ML LAYER)

Это твоя XGBoost / ONNX модель:

Value(S) → probability of winning

Но важно:

👉 она НЕ выбирает героев
👉 она только оценивает состояние

🌳 3. TREE SEARCH LAYER (DECISION ENGINE)

Теперь ты строишь поиск:

for each candidate hero:
    simulate pick
    evaluate new state
    simulate opponent response
    compute expected value
Финальная формула:
Score(hero) =
Σ P(opponent_response) * Value(S_after_sequence)
Оптимизация:
top-K candidates (10–20 героев)
beam search
1–2 steps lookahead
caching embeddings
🧠 ИТОГОВАЯ АРХИТЕКТУРА
                 CURRENT DRAFT STATE
                           ↓
               CANDIDATE GENERATION
                           ↓
                SIMULATE PICK(HERO)
                           ↓
                FEATURE UPDATE LAYER
                           ↓
                 VALUE MODEL (ML)
                           ↓
              OPPONENT RESPONSE MODEL
                           ↓
               EXPECTED VALUE SCORE
                           ↓
                 BEST HERO PICK
🚀 ЧАСТЬ 1 — КАК ОБУЧАТЬ ML ПОД TREE SEARCH

Это КЛЮЧЕВОЙ момент.

❌ Ошибка большинства систем

Они учат:

predict radiant_win from full draft

И всё.

👉 это НЕ подходит для tree search

✔️ ПРАВИЛЬНЫЙ ПОДХОД

Ты должен обучать:

👉 “value of action” model
🧠 1. Переформулировка задачи

Вместо:

P(win | full draft)

Ты учишь:

Δ Value = Value(S_after_pick) - Value(S_before)
📊 2. Генерация training data (самое важное)

Для каждого матча ты делаешь:

Step 1 — берёшь реальный draft sequence
S0 → S1 → S2 → S3 → ... → final
Step 2 — создаёшь “counterfactual states”

Ты симулируешь:

“а что если вместо этого героя был другой?”
Step 3 — label = outcome influence

Пример label:

Value(S_k) = win probability from that state forward
Step 4 — training samples:

Каждый sample:

(state S, action A) → future win probability
🧠 3. Final training format
X = (state features + candidate hero)
y = win probability after full game
⚠️ ВАЖНО

Ты НЕ учишь:

“good hero”
“bad hero”

Ты учишь:

👉 “good move in this state”
🌳 4. Training for tree search (critical upgrade)

Ты добавляешь:

multi-step targets
Value(S, depth=1)
Value(S, depth=2)
Value(S, depth=3)
why:

Tree search uses:

1-step evaluation
2-step foresight
opponent reaction
🧠 5. SELF-DISTILLATION LOOP

Очень важный трюк:

model plays draft simulation
generates best picks
logs results
retrains on its own decisions
🚀 ЧАСТЬ 2 — SELF-IMPROVING SYSTEM

Это делает систему “живой”.

🔁 1. Основная идея

Система учится не только на реальных матчах:

👉 но и на своих симуляциях

⚙️ 2. Loop architecture
        REAL MATCH DATA
               ↓
        TRAIN INITIAL MODEL
               ↓
        TREE SEARCH SIMULATION
               ↓
     LOG AI DECISIONS + SCORES
               ↓
     CREATE NEW TRAINING DATA
               ↓
        RETRAIN MODEL
               ↓
            REPEAT
🧠 3. Self-play style improvement

Ты создаёшь:

“AI vs AI draft simulation”
AI Radiant vs AI Dire
оба используют tree search
оба оптимизируют win probability
📊 4. Что ты получаешь
новые unusual drafts
edge-case scenarios
meta discovery
⚠️ 5. КРИТИЧНО: prevent model collapse

Если просто self-train:

❌ модель начнёт зацикливаться

✔️ решения:
1. inject real data ratio
70% real matches
30% self-play
2. exploration noise
epsilon-greedy picks
random hero injection
3. periodic reset
re-train from base checkpoint
🧬 6. ADVANCED SELF-IMPROVEMENT (level 2)

Добавляешь:

“critic model”
Critic(S) → detect bad drafts
pipeline:
Actor (tree search) → makes pick
Critic → evaluates mistakes
→ feedback loop
🔥 7. FINAL SYSTEM (WHAT YOU ARE BUILDING)

Ты фактически строишь:

👉 hybrid game AI system

состоит из:

1. State encoder
2. Value model (XGBoost / ONNX)
3. Tree search planner
4. Opponent model
5. Self-play trainer
6. Critic network