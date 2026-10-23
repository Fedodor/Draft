import json
import pandas as pd
from pathlib import Path

heroes = json.loads(Path('ml/datasets/heroes.json').read_text())
name_to_id = {}
for hid_str, hero in heroes.items():
    name = hero.get('localized_name', '').lower().replace(' ', '_').replace("'", '').replace('-', '_')
    name_to_id[name] = int(hid_str)

wdf = pd.read_parquet('ml/datasets/processed/hero_winrates_named.parquet')
rows = []
for _, r in wdf.iterrows():
    name = r['hero_name'].lstrip('_')
    hid = name_to_id.get(name)
    if hid:
        rows.append({'hero_id': hid, 'games': r['games'], 'wins': r['wins'], 'winrate': r['winrate']})

out = pd.DataFrame(rows).sort_values('hero_id')
out.to_parquet('ml/datasets/processed/hero_winrates.parquet', index=False)
print('Saved %d hero winrates with hero_id' % len(out))
for _, r in out.head(10).iterrows():
    print('  Hero %d: %.3f (%d games)' % (r['hero_id'], r['winrate'], r['games']))
