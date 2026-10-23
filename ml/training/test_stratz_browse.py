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

# Try a single league without gameMode filter
LEAGUE_MATCHES_QUERY = '''
query GetLeagueMatches($leagueId: Int!) {
    league(id: $leagueId) {
        id
        name
        matches(request: {
            take: 10,
            skip: 0
        }) {
            id
            didRadiantWin
            gameMode
            gameVersionId
            durationSeconds
            players {
                heroId
                isRadiant
            }
            pickBans {
                heroId
                isPick
                isRadiant
            }
        }
    }
}
'''

# Try first professional league
resp = requests.post(url, json={
    'query': LEAGUE_MATCHES_QUERY,
    'variables': {'leagueId': 16251}
}, headers=headers, timeout=30)

print('Status:', resp.status_code)
data = resp.json()
if 'errors' in data:
    print('Errors:', data['errors'])
else:
    league = data.get('data', {}).get('league', {})
    matches = league.get('matches', [])
    print('League: %s, Matches: %d' % (league.get('name'), len(matches)))
    for m in matches:
        print('  Match %d: mode=%s, version=%d, players=%d, pickbans=%d' % (
            m.get('id'), m.get('gameMode'), m.get('gameVersionId'),
            len(m.get('players', [])), len(m.get('pickBans', []))
        ))
