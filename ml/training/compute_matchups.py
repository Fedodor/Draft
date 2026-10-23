import pandas as pd
import numpy as np
from pathlib import Path
import json
import time

output_dir = Path('ml/datasets/processed')
df = pd.read_parquet(output_dir / 'draft_features.parquet')
print('Loaded %d samples' % len(df))

rad_has_cols = [c for c in df.columns if c.startswith('rad_has_')]
dire_has_cols = [c for c in df.columns if c.startswith('dire_has_')]

print('Computing hero matchup winrates from feature matrix...')

# For each pair of heroes (rad_hero, dire_hero), compute how often rad_hero's team wins
# Using the feature matrix directly (vectorized)

def safe_name(localized_name):
    return (
        localized_name.strip().lower().replace(' ', '_').replace("'", '').replace('-', '_')
    )

heroes = json.loads(Path('infrastructure/seeders/heroes.json').read_text(encoding='utf-8'))
hero_id_by_name = {safe_name(h['localized_name']): int(h['id']) for h in heroes}
hero_names = [c[8:] for c in rad_has_cols]  # strip 'rad_has_'

matchup_rows = []
t0 = time.time()

for i, rad_col in enumerate(rad_has_cols):
    rad_name = hero_names[i]
    if rad_name not in hero_id_by_name:
        continue
    rad_mask = df[rad_col] == 1
    rad_count = rad_mask.sum()
    if rad_count < 50:
        continue
    
    for j, dire_col in enumerate(dire_has_cols):
        dire_name = hero_names[j]
        if rad_name == dire_name or dire_name not in hero_id_by_name:
            continue
        
        dire_mask = df[dire_col] == 1
        
        # Both heroes present in same match
        both = rad_mask & dire_mask
        games = both.sum()
        if games < 20:
            continue
        
        # Radiant wins when both heroes are present
        wins = (both & (df['radiant_win'] == 1)).sum()
        winrate = float(wins / games)
        
        matchup_rows.append({
            'hero_a': hero_id_by_name[rad_name],
            'hero_b': hero_id_by_name[dire_name],
            'games': int(games),
            'hero_a_winrate': winrate,
            'hero_b_winrate': 1.0 - winrate,
        })

elapsed = time.time() - t0
print('Computed %d matchups in %.1fs' % (len(matchup_rows), elapsed))

matchup_df = pd.DataFrame(matchup_rows)
matchup_df = matchup_df.sort_values(['hero_a', 'hero_b'])
matchup_df.to_parquet(output_dir / 'hero_matchups.parquet', index=False)
print('Saved to hero_matchups.parquet')

# Summary
print('\nStrongest counter pairs (hero_a beats hero_b, >=50 games):')
strong = matchup_df[matchup_df['games'] >= 50].nlargest(15, 'hero_a_winrate')
for _, r in strong.iterrows():
    print('  %s vs %s: %.3f (%d games)' % (r['hero_a'], r['hero_b'], r['hero_a_winrate'], r['games']))

print('\nWeakest matchups (hero_a loses to hero_b, >=50 games):')
weak = matchup_df[matchup_df['games'] >= 50].nsmallest(10, 'hero_a_winrate')
for _, r in weak.iterrows():
    print('  %s vs %s: %.3f (%d games)' % (r['hero_a'], r['hero_b'], r['hero_a_winrate'], r['games']))

# Hero overall winrate (from matchup data, as cross-check)
hero_stats = {}
for _, r in matchup_df.iterrows():
    h = r['hero_a']
    if h not in hero_stats:
        hero_stats[h] = {'wins': 0, 'games': 0}
    hero_stats[h]['wins'] += r['hero_a_winrate'] * r['games']
    hero_stats[h]['games'] += r['games']

print('\nHero overall winrates (from matchup data):')
sorted_h = sorted(hero_stats.items(), key=lambda x: x[1]['wins']/max(1,x[1]['games']), reverse=True)
for name, s in sorted_h[:10]:
    print('  %s: %.3f (%d games)' % (name, s['wins']/max(1,s['games']), s['games']))
print('...')
for name, s in sorted_h[-5:]:
    print('  %s: %.3f (%d games)' % (name, s['wins']/max(1,s['games']), s['games']))

