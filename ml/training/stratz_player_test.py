import requests, json

from settings import require_stratz_key

STRATZ_KEY = require_stratz_key()
headers = {'Authorization': 'Bearer %s' % STRATZ_KEY, 'User-Agent': 'STRATZ_API', 'Content-Type': 'application/json'}

# Introspect PlayerType
resp = requests.post('https://api.stratz.com/graphql', json={'query': '{ __type(name: "PlayerType") { fields { name } } }'}, headers=headers, timeout=30)
print('PlayerType fields:')
data = resp.json()
for f in data.get('data', {}).get('__type', {}).get('fields', []):
    print('  ', f['name'])

# Try player query
resp = requests.post('https://api.stratz.com/graphql', json={'query': '{ player(steamAccountId: 103563464) { steamAccountId names { name } matches(request: {take: 5}) { id gameMode didRadiantWin durationSeconds startDateTime players { heroId isRadiant } pickBans { heroId isPick isRadiant order } } } }'}, headers=headers, timeout=30)
print('\nPlayer query:', resp.status_code)
print(resp.text[:2000])