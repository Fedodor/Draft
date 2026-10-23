import requests, json, time
from pathlib import Path

from settings import require_stratz_key

STRATZ_KEY = require_stratz_key()
headers = {'Authorization': 'Bearer %s' % STRATZ_KEY, 'User-Agent': 'STRATZ_API', 'Content-Type': 'application/json'}

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

print('Existing: %d' % len(existing_ids))

MATCH_QUERY = '''
query GetPlayerMatches($accountId: Long!, $take: Int!, $skip: Int!) {
    player(steamAccountId: $accountId) {
        steamAccountId
        names { name }
        matches(request: {take: $take, skip: $skip}) {
            id
            didRadiantWin
            startDateTime
            durationSeconds
            gameMode
            gameVersionId
            numHumanPlayers
            players { heroId isRadiant playerSlot steamAccountId }
            pickBans { heroId isPick isRadiant order }
        }
    }
}
'''

# Get pro player IDs from leagues
def get_pro_players_from_leagues():
    pro_ids = set()
    # Get matches from high-tier leagues and extract player IDs
    for skip in range(0, 200, 50):
        resp = requests.post('https://api.stratz.com/graphql', json={
            'query': '{ leagues(request: {skip: %d, take: 50}) { id tier matches(request: {take: 20, skip: 0}) { players { steamAccountId } } } }' % skip
        }, headers=headers, timeout=30)
        data = resp.json()
        for league in data.get('data', {}).get('leagues', []):
            if league.get('tier') in ('PROFESSIONAL', 'DPC_LEAGUE', 'MAJOR', 'INTERNATIONAL'):
                for match in league.get('matches', []):
                    for player in match.get('players', []):
                        sid = player.get('steamAccountId')
                        if sid:
                            pro_ids.add(sid)
        time.sleep(1)
    return list(pro_ids)

print('Finding pro players from leagues...')
pro_ids = get_pro_players_from_leagues()
print('Found %d pro player IDs' % len(pro_ids))

# Now fetch CM matches from these players
downloaded = 0
for pid in pro_ids[:50]:  # Limit to first 50
    for skip in range(0, 200, 50):
        try:
            resp = requests.post('https://api.stratz.com/graphql', json={
                'query': MATCH_QUERY,
                'variables': {'accountId': pid, 'take': 50, 'skip': skip}
            }, headers=headers, timeout=30)
            
            if resp.status_code == 429:
                time.sleep(30)
                continue
            
            data = resp.json()
            player = data.get('data', {}).get('player')
            if not player:
                break
            
            matches = player.get('matches', [])
            if not matches:
                break
            
            new = 0
            for m in matches:
                if m.get('gameMode') != 'CAPTAINS_MODE':
                    continue
                mid = m.get('id')
                if mid in existing_ids:
                    continue
                
                if m.get('players') and m.get('pickBans'):
                    with open(raw_dir / ('%d.json' % mid), 'w') as f:
                        json.dump(m, f)
                    existing_ids.add(mid)
                    downloaded += 1
                    new += 1
            
            if new == 0:
                break
            time.sleep(0.5)
            
        except Exception as e:
            print('Error %d: %s' % (pid, e))
            break
    time.sleep(1)

print('\nDownloaded %d new matches, total: %d' % (downloaded, len(existing_ids)))