import requests, json, time

from settings import require_stratz_key

time.sleep(10)

STRATZ_KEY = require_stratz_key()
headers = {'Authorization': 'Bearer %s' % STRATZ_KEY, 'User-Agent': 'STRATZ_API', 'Content-Type': 'application/json'}

# First introspect
resp = requests.post('https://api.stratz.com/graphql', json={
    'query': '{ __type(name: "GameVersionType") { fields { name } } }'
}, headers=headers, timeout=30)
print('Introspect:', resp.status_code)
print(resp.text[:1000])

time.sleep(2)

# Now query with correct fields
resp = requests.post('https://api.stratz.com/graphql', json={
    'query': '{ constants { gameVersions { id name } } }'
}, headers=headers, timeout=30)
print('Versions:', resp.status_code)
print(resp.text[:3000])
