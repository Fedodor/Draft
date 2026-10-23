import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

raw_dir = Path('ml/datasets/raw')
output_dir = Path('ml/datasets/processed')

match_files = [f for f in raw_dir.glob('*.json') if f.name not in ('metadata.json', 'all_pick_matches.json', 'allpick_ids.json')]
print('Computing hero win rates from %d matches...' % len(match_files))

hero_stats = defaultdict(lambda: [0, 0])

for i, f in enumerate(match_files):
    if i % 20000 == 0:
        print('  %d/%d...' % (i, len(match_files)))
    try:
        with open(f, 'r') as fh:
            match = json.load(fh)
        
        players = match.get('players')
        if not players:
            continue
        
        rw = match.get('radiant_win', match.get('didRadiantWin'))
        if rw is None:
            continue
        
        for player in players:
            hid = player.get('hero_id') or player.get('heroId')
            if hid is None:
                continue
            ir = player.get('isRadiant')
            if ir is None:
                ir = player.get('player_slot', 128) < 128
            hero_stats[hid][0] += 1
            if (ir and rw) or (not ir and not rw):
                hero_stats[hid][1] += 1
    except Exception:
        pass

rows = []
for hid, (games, wins) in hero_stats.items():
    if games >= 10:
        rows.append({'hero_id': hid, 'games': games, 'wins': wins, 'winrate': wins / games})

df = pd.DataFrame(rows).sort_values('winrate', ascending=False)
df.to_parquet(output_dir / 'hero_winrates.parquet', index=False)
print('Saved %d hero win rates' % len(df))
for _, r in df.head(10).iterrows():
    print('  Hero %d: %.3f (%d games)' % (r['hero_id'], r['winrate'], r['games']))
