import requests
import json
import time
from pathlib import Path
from datetime import datetime

from settings import require_stratz_key

raw_dir = Path('ml/datasets/raw')
raw_dir.mkdir(parents=True, exist_ok=True)

STRATZ_KEY = require_stratz_key()
headers = {'Authorization': 'Bearer %s' % STRATZ_KEY, 'User-Agent': 'STRATZ_API', 'Content-Type': 'application/json'}

TARGET_TIERS = {'PROFESSIONAL', 'DPC_LEAGUE', 'MAJOR', 'INTERNATIONAL', 'DPC_LEAGUE_QUALIFIER'}

MATCH_QUERY = '''
query GetMatch($matchId: Long!) {
    match(id: $matchId) {
        id
        didRadiantWin
        startDateTime
        durationSeconds
        gameMode
        gameVersionId
        numHumanPlayers
        players {
            heroId
            isRadiant
            playerSlot
            lane
            steamAccountId
        }
        pickBans {
            heroId
            isPick
            isRadiant
            order
        }
    }
}
'''

# Check existing
existing_ids = set()
for f in raw_dir.glob('*.json'):
    if f.name not in ('metadata.json', 'all_pick_matches.json', 'allpick_ids.json'):
        try:
            existing_ids.add(int(f.stem))
        except:
            pass

print('Existing matches: %d' % len(existing_ids))

# First, get high-tier leagues
print('Fetching high-tier leagues...')
league_ids = []
for skip in range(0, 2000, 100):
    resp = requests.post('https://api.stratz.com/graphql', json={
        'query': '{ leagues(request: {skip: %d, take: 100}) { id name displayName tier } }' % skip
    }, headers=headers, timeout=30)
    data = resp.json()
    leagues = data.get('data', {}).get('leagues', [])
    if not leagues:
        break
    for l in leagues:
        tier_str = l.get('tier', 'UNKNOWN')
        if tier_str in TARGET_TIERS:
            league_ids.append(l['id'])
    time.sleep(0.5)

print('Found %d high-tier leagues' % len(league_ids))

# Get matches from these leagues
TARGET_TIERS = {'PROFESSIONAL', 'DPC_LEAGUE', 'MAJOR', 'INTERNATIONAL', 'DPC_LEAGUE_QUALIFIER'}

downloaded = 0
for league_id in league_ids:
    for skip in range(0, 5000, 100):
        try:
            resp = requests.post('https://api.stratz.com/graphql', json={
                'query': 'query { league(id: %d) { matches(request: {take: 100, skip: %d}) { id } } }' % (league_id, skip)
            }, headers=headers, timeout=30)
            
            if resp.status_code == 429:
                time.sleep(30)
                continue
            
            data = resp.json()
            if 'errors' in data:
                break
            
            matches = data.get('data', {}).get('league', {}).get('matches', [])
            if not matches:
                break
            
            new = 0
            for m in matches:
                mid = m.get('id')
                if mid and mid not in existing_ids:
                    # Fetch full match
                    time.sleep(0.1)
                    detail_resp = requests.post('https://api.stratz.com/graphql', json={
                        'query': MATCH_QUERY,
                        'variables': {'matchId': mid}
                    }, headers=headers, timeout=30)
                    
                    if detail_resp.status_code == 429:
                        time.sleep(30)
                        continue
                    
                    ddata = detail_resp.json()
                    match = ddata.get('data', {}).get('match')
                    if match and match.get('players') and match.get('pickBans'):
                        with open(raw_dir / ('%d.json' % mid), 'w') as f:
                            json.dump(match, f)
                        existing_ids.add(mid)
                        downloaded += 1
                        new += 1
            
            if new == 0:
                break
            time.sleep(0.3)
            
        except Exception as e:
            print('Error league %d: %s' % (league_id, e))
            break
    
    time.sleep(0.5)

print('\nDone! Downloaded %d new matches, total: %d' % (downloaded, len(existing_ids)))