import asyncio
from training.data_collector import OpenDotaCollector

async def test():
    async with OpenDotaCollector(rate_limit=1.0) as collector:
        # Test the raw public matches first
        matches = await collector.get_public_matches(limit=100, mmr=4000)
        all_pick = [m for m in matches if m.get('game_mode') == 1]
        print(f"Total matches: {len(matches)}, All Pick (game_mode 1): {len(all_pick)}")
        
        # Check a few
        for m in all_pick[:5]:
            print(f"  Match {m['match_id']}: game_mode={m.get('game_mode')}, lobby_type={m.get('lobby_type')}")

asyncio.run(test())