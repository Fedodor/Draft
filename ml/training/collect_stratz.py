import requests
import json
import time
from pathlib import Path

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

# Query to get matches from a league
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
}
'''

# Get leagues query - try more leagues with different tiers
LEAGUES_QUERY = '''
{
    leagues(request: {skip: 0, take: 200}) {
        id
        name
        tier
        startDateTime
    }
}
'''

# Get leagues
resp = requests.post(url, json={'query': LEAGUES_QUERY}, headers=headers, timeout=30)
data = resp.json()
leagues = data.get('data', {}).get('leagues', [])
print('Found %d leagues' % len(leagues))

# Check existing
existing_ids = set()
for f in output_dir.glob('*.json'):
    if f.name not in ('metadata.json', 'all_pick_matches.json', 'allpick_ids.json'):
        try:
            existing_ids.add(int(f.stem))
        except:
            pass

print('Existing matches: %d' % len(existing_ids))

# Target patches: 7.35-7.41
# Stratz gameVersionId: 164=7.35, 170=7.36, 178=7.37, 182=7.38
# OpenDota: 58=7.39, 59=7.40, 60=7.41
target_versions = {164, 170, 178, 182}  # Stratz IDs for 7.35-7.38

downloaded = 0
leagues_processed = 0

for league in leagues:
    league_id = league.get('id')
    
    print('\nFetching league %d...' % league_id)
    
    # Paginate through matches
    for skip in range(0, 1000, 50):
        try:
            resp = requests.post(url, json={
                'query': LEAGUE_MATCHES_QUERY,
                'variables': {'leagueId': league_id, 'skip': skip}
            }, headers=headers, timeout=30)
            
            if resp.status_code == 429:
                print('  Rate limited, waiting 60s...')
                time.sleep(60)
                continue
            
            if resp.status_code != 200:
                print('  Error: %d' % resp.status_code)
                break
            
            data = resp.json()
            if 'errors' in data:
                print('  GraphQL errors:', data['errors'])
                break
            
            league_data = data.get('data', {}).get('league', {})
            matches = league_data.get('matches', [])
            
            if not matches:
                break
            
            count = 0
            for match in matches:
                mid = match.get('id')
                game_mode = match.get('gameMode')
                version = match.get('gameVersionId')
                
                # Only save Captain's Mode matches
                if game_mode not in ('CAPTAINS_MODE', 'CAPTAINS_DRAFT'):
                    continue
                if mid in existing_ids:
                    continue
                
                # Save match
                match_file = output_dir / (str(mid) + '.json')
                with open(match_file, 'w') as f:
                    json.dump(match, f, indent=2)
                
                existing_ids.add(mid)
                downloaded += 1
                count += 1
            
            if count > 0:
                print('  Skip %d: saved %d CM matches (total: %d)' % (skip, count, downloaded))
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print('  Error: %s' % e)
            break
    
    leagues_processed += 1
    time.sleep(1)  # Rate limiting between leagues
    
    # Print progress every 10 leagues
    if leagues_processed % 10 == 0:
        print('\n--- Progress: %d leagues processed, %d new matches ---' % (leagues_processed, downloaded))

print('\n=== FINAL ===')
print('Leagues processed: %d' % leagues_processed)
print('Total new downloads: %d' % downloaded)
print('Total matches now: %d' % len(existing_ids))
