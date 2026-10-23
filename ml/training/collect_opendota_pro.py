import requests
import json
import time
from pathlib import Path
from datetime import datetime

output_dir = Path('ml/datasets/raw')

existing_ids = set()
for f in output_dir.glob('*.json'):
    if f.name not in ('metadata.json', 'all_pick_matches.json', 'allpick_ids.json'):
        try:
            existing_ids.add(int(f.stem))
        except:
            pass

print('Existing matches: %d' % len(existing_ids))

downloaded = 0

# Collect OpenDota pro matches - paginate backwards through many pages
for batch in range(200):
    offset = batch * 100
    url = 'https://api.opendota.com/api/proMatches'
    params = {'offset': offset}
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            print('Rate limited, waiting 120s...')
            time.sleep(120)
            continue
        if resp.status_code != 200:
            print('Error: %d' % resp.status_code)
            break
        
        pro_matches = resp.json()
        if not pro_matches:
            print('No more matches at offset %d' % offset)
            break
        
        new = 0
        for m in pro_matches:
            mid = m['match_id']
            if mid in existing_ids:
                continue
            
            time.sleep(0.3)
            try:
                detail_resp = requests.get(
                    'https://api.opendota.com/api/matches/%d' % mid,
                    timeout=30
                )
                if detail_resp.status_code == 429:
                    print('  Detail rate limited, waiting 60s...')
                    time.sleep(60)
                    continue
                if detail_resp.status_code == 200:
                    detail = detail_resp.json()
                    if detail.get('players') and detail.get('picks_bans'):
                        with open(output_dir / ('%d.json' % mid), 'w') as f:
                            json.dump(detail, f)
                        existing_ids.add(mid)
                        downloaded += 1
                        new += 1
            except:
                pass
        
        print('[%s] Offset %d: +%d new (total: %d)' % (datetime.now().strftime('%H:%M:%S'), offset, new, len(existing_ids)))
        time.sleep(1)
        
    except Exception as e:
        print('Error: %s' % e)
        break

print('\nTotal: %d matches, new: %d' % (len(existing_ids), downloaded))
