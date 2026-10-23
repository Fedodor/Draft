import requests
import json
import time
from pathlib import Path
from datetime import datetime

from settings import get_opendota_key

output_dir = Path('ml/datasets/raw')
output_dir.mkdir(parents=True, exist_ok=True)

API_KEY = get_opendota_key()

# Check existing
existing_ids = set()
for f in output_dir.glob('*.json'):
    if f.name not in ('metadata.json', 'all_pick_matches.json', 'allpick_ids.json'):
        try:
            existing_ids.add(int(f.stem))
        except:
            pass

print('Existing matches: %d' % len(existing_ids))

# Try OpenDota - pro matches for multiple patches
patches = [54, 55, 56, 57, 58, 59, 60]  # 7.35-7.41
downloaded = 0

for patch in patches:
    print('\nFetching OpenDota pro matches, patch %d...' % patch)
    
    url = 'https://api.opendota.com/api/proMatches'
    params = {'patch': patch, 'limit': 100}
    if API_KEY:
        params['api_key'] = API_KEY
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        
        if resp.status_code == 429:
            print('  Rate limited, waiting 120s...')
            time.sleep(120)
            resp = requests.get(url, params=params, timeout=30)
        
        if resp.status_code != 200:
            print('  Error: %d' % resp.status_code)
            continue
        
        matches = resp.json()
        print('  Got %d pro matches' % len(matches))
        
        for m in matches:
            mid = m['match_id']
            if mid in existing_ids:
                continue
            
            time.sleep(0.3)
            
            try:
                detail_resp = requests.get(
                    'https://api.opendota.com/api/matches/%d' % mid,
                    params={'api_key': API_KEY} if API_KEY else {},
                    timeout=30
                )
                
                if detail_resp.status_code == 429:
                    time.sleep(60)
                    continue
                
                if detail_resp.status_code == 200:
                    detail = detail_resp.json()
                    with open(output_dir / ('%d.json' % mid), 'w') as f:
                        json.dump(detail, f)
                    existing_ids.add(mid)
                    downloaded += 1
                    
            except:
                pass
        
        time.sleep(2)
        
    except Exception as e:
        print('  Error: %s' % e)

# Also try public matches (captain's mode filtered)
print('\nFetching OpenDota public matches...')
for mmr in [5000, 4500, 4000, 3500]:
    url = 'https://api.opendota.com/api/publicMatches'
    params = {'limit': 100, 'min_mmr': mmr}
    if API_KEY:
        params['api_key'] = API_KEY
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        
        if resp.status_code == 429:
            time.sleep(120)
            continue
        
        if resp.status_code != 200:
            continue
        
        pub_matches = resp.json()
        
        cm_matches = [m for m in pub_matches if m.get('game_mode') in (22, 23)]
        print('  MMR>=%d: %d CM matches found' % (mmr, len(cm_matches)))
        
        for m in cm_matches[:30]:
            mid = m['match_id']
            if mid in existing_ids:
                continue
            
            time.sleep(0.3)
            
            try:
                detail_resp = requests.get(
                    'https://api.opendota.com/api/matches/%d' % mid,
                    params={'api_key': API_KEY} if API_KEY else {},
                    timeout=30
                )
                
                if detail_resp.status_code == 429:
                    time.sleep(60)
                    continue
                
                if detail_resp.status_code == 200:
                    detail = detail_resp.json()
                    with open(output_dir / ('%d.json' % mid), 'w') as f:
                        json.dump(detail, f)
                    existing_ids.add(mid)
                    downloaded += 1
                    
            except:
                pass
        
        time.sleep(2)
        
    except:
        pass

print('\n=== OPEN DOTA COMPLETE ===')
print('New downloads: %d' % downloaded)
print('Total matches: %d' % len(existing_ids))
