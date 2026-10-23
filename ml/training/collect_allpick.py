import requests
import json
import time
from pathlib import Path
from datetime import datetime

raw_dir = Path('ml/datasets/raw')
raw_dir.mkdir(parents=True, exist_ok=True)


# Check existing
existing_ids = set()
for f in raw_dir.glob('*.json'):
    if f.name not in ('metadata.json', 'all_pick_matches.json', 'allpick_ids.json'):
        try:
            existing_ids.add(int(f.stem))
        except:
            pass

print('Existing matches: %d' % len(existing_ids))
print('Target: collect as many All Pick (non-CM) matches as possible')

downloaded = 0
rate_limit_hits = 0

# Fetch public matches (all modes), extract non-CM ones
# publicMatches gives basic info (no players array), but we get match_id
# Then fetch /matches/{id} for full data including heroes

for page in range(0, 5000):
    try:
        # publicMatches returns recent matches with basic info
        resp = requests.get(
            'https://api.opendota.com/api/publicMatches',
            params={'offset': page * 100, 'limit': 100},
            timeout=30
        )
        
        if resp.status_code == 429:
            rate_limit_hits += 1
            if rate_limit_hits > 5:
                print('Too many rate limits, stopping at page %d' % page)
                break
            print('  Rate limited, waiting 120s...')
            time.sleep(120)
            continue
        
        if resp.status_code != 200:
            print('  Error %d at page %d' % (resp.status_code, page))
            break
        
        matches = resp.json()
        if not matches:
            print('  No matches at page %d' % page)
            break
        
        # Filter for non-CM matches (All Pick = game_mode 22, others are interesting too)
        # We want matches that have players info
        new_count = 0
        for m in matches:
            mid = m['match_id']
            if mid in existing_ids:
                continue
            
            # Skip CM (we have plenty)
            if m.get('game_mode') in (22, 23):
                continue
            
            time.sleep(0.3)
            
            try:
                detail_resp = requests.get(
                    'https://api.opendota.com/api/matches/%d' % mid,
                    timeout=30
                )
                
                if detail_resp.status_code == 429:
                    rate_limit_hits += 1
                    time.sleep(60)
                    continue
                
                if detail_resp.status_code == 200:
                    detail = detail_resp.json()
                    if detail.get('players') and len(detail['players']) == 10:
                        with open(raw_dir / ('%d.json' % mid), 'w') as f:
                            json.dump(detail, f)
                        existing_ids.add(mid)
                        downloaded += 1
                        new_count += 1
            except:
                pass
        
        if page % 50 == 0:
            print('[%s] Page %d: +%d new (total: %d, rate limits: %d)' % (
                datetime.now().strftime('%H:%M:%S'), page, new_count, len(existing_ids), rate_limit_hits))
        
        time.sleep(1)
        
    except Exception as e:
        print('Error at page %d: %s' % (page, e))
        break

print('\nDone! Downloaded %d new matches, total: %d' % (downloaded, len(existing_ids)))
