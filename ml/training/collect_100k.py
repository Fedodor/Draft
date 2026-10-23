import requests
import json
import time
from pathlib import Path
from datetime import datetime

from settings import require_stratz_key

output_dir = Path('ml/datasets/raw')
output_dir.mkdir(parents=True, exist_ok=True)

STRATZ_KEY = require_stratz_key()

url = 'https://api.stratz.com/graphql'
headers = {
    'Authorization': 'Bearer %s' % STRATZ_KEY,
    'User-Agent': 'STRATZ_API',
    'Content-Type': 'application/json'
}

LEAGUE_MATCHES_QUERY = '''
query GetLeagueMatches($leagueId: Int!, $skip: Int!) {
    league(id: $leagueId) {
        id
        name
        matches(request: {
            take: 50,
            skip: $skip
        }) {
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
}
'''

# Get ALL leagues (more pages)
all_leagues = []
for skip in range(0, 2000, 200):
    try:
        resp = requests.post(url, json={
            'query': '''{ leagues(request: {skip: %d, take: 200}) { id tier } }''' % skip
        }, headers=headers, timeout=30)
        if resp.status_code == 429:
            time.sleep(30)
            continue
        data = resp.json()
        leagues = data.get('data', {}).get('leagues', [])
        if not leagues:
            break
        all_leagues.extend(leagues)
        time.sleep(1)
    except:
        break

print('Found %d leagues total' % len(all_leagues))

# Check existing
existing_ids = set()
for f in output_dir.glob('*.json'):
    if f.name not in ('metadata.json', 'all_pick_matches.json', 'allpick_ids.json'):
        try:
            existing_ids.add(int(f.stem))
        except:
            pass

print('Existing matches: %d' % len(existing_ids))
print('Target: 200,000+')

downloaded = 0
target = 200000 - len(existing_ids)

for league in all_leagues:
    if downloaded >= target:
        break
    
    league_id = league.get('id')
    
    for skip in range(0, 2000, 50):
        try:
            resp = requests.post(url, json={
                'query': LEAGUE_MATCHES_QUERY,
                'variables': {'leagueId': league_id, 'skip': skip}
            }, headers=headers, timeout=30)
            
            if resp.status_code == 429:
                time.sleep(60)
                continue
            
            if resp.status_code != 200:
                break
            
            data = resp.json()
            if 'errors' in data:
                break
            
            matches = data.get('data', {}).get('league', {}).get('matches', [])
            if not matches:
                break
            
            count = 0
            for match in matches:
                mid = match.get('id')
                if mid in existing_ids:
                    continue
                match_file = output_dir / (str(mid) + '.json')
                with open(match_file, 'w') as f:
                    json.dump(match, f)
                existing_ids.add(mid)
                downloaded += 1
                count += 1
            
            if downloaded >= target:
                break
            
            time.sleep(0.5)
            
        except:
            break
    
    time.sleep(0.5)
    
    if downloaded % 500 == 0 and downloaded > 0:
        print('[%s] Downloaded: %d, Total: %d, Remaining: %d' % (
            datetime.now().strftime('%H:%M:%S'), downloaded, len(existing_ids), max(0, target - downloaded)))

print('\n=== COMPLETE ===')
print('New downloads: %d' % downloaded)
print('Total matches: %d' % len(existing_ids))
