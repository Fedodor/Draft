import requests, json, time

from settings import require_stratz_key

STRATZ_KEY = require_stratz_key()
headers = {'Authorization': 'Bearer %s' % STRATZ_KEY, 'User-Agent': 'STRATZ_API', 'Content-Type': 'application/json'}

# Try some known pro player IDs (these are from public knowledge)
pro_players = [
    103563464,  # placeholder
    86745912,   # Miracle-
    135613220,  # Arteezy
    87518093,   # SumaiL
    105708995,  # Nisha
]

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

for pid in pro_players:
    print('\nTrying player %d...' % pid)
    try:
        resp = requests.post('https://api.stratz.com/graphql', json={
            'query': MATCH_QUERY,
            'variables': {'accountId': pid, 'take': 10, 'skip': 0}
        }, headers=headers, timeout=30)
        data = resp.json()
        player = data.get('data', {}).get('player')
        if player:
            matches = player.get('matches', [])
            names = player.get('names', [])
            print('  Names: %s' % [n['name'] for n in names])
            print('  Matches: %d' % len(matches))
            if matches:
                print('  First match: %s' % matches[0])
        else:
            print('  No player data')
    except Exception as e:
        print('  Error: %s' % e)
    time.sleep(2)