#!/usr/bin/env python3
"""Generate heroes.json and hero_semantics.json for Smart Draft Assistant."""

import json
import os

# Deterministic pseudo-random generator using LCG
class LCG:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state

    def next_float(self, lo, hi):
        return lo + (self.next() / 0x7FFFFFFF) * (hi - lo)

    def next_int(self, lo, hi):
        return lo + self.next() % (hi - lo + 1)


HEROES_API = [
    {"id":1,"name":"npc_dota_hero_antimage","localized_name":"Anti-Mage","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Escape","Nuker"],"legs":2},
    {"id":2,"name":"npc_dota_hero_axe","localized_name":"Axe","primary_attr":"str","attack_type":"Melee","roles":["Initiator","Durable","Disabler","Carry"],"legs":2},
    {"id":3,"name":"npc_dota_hero_bane","localized_name":"Bane","primary_attr":"all","attack_type":"Ranged","roles":["Support","Disabler","Nuker","Durable"],"legs":4},
    {"id":4,"name":"npc_dota_hero_bloodseeker","localized_name":"Bloodseeker","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Disabler","Nuker","Initiator"],"legs":2},
    {"id":5,"name":"npc_dota_hero_crystal_maiden","localized_name":"Crystal Maiden","primary_attr":"int","attack_type":"Ranged","roles":["Support","Disabler","Nuker"],"legs":2},
    {"id":6,"name":"npc_dota_hero_drow_ranger","localized_name":"Drow Ranger","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Disabler","Pusher"],"legs":2},
    {"id":7,"name":"npc_dota_hero_earthshaker","localized_name":"Earthshaker","primary_attr":"str","attack_type":"Melee","roles":["Support","Initiator","Disabler","Nuker"],"legs":2},
    {"id":8,"name":"npc_dota_hero_juggernaut","localized_name":"Juggernaut","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Pusher","Escape"],"legs":2},
    {"id":9,"name":"npc_dota_hero_mirana","localized_name":"Mirana","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Support","Escape","Nuker","Disabler"],"legs":2},
    {"id":10,"name":"npc_dota_hero_morphling","localized_name":"Morphling","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Escape","Durable","Nuker","Disabler"],"legs":0},
    {"id":11,"name":"npc_dota_hero_nevermore","localized_name":"Shadow Fiend","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Nuker"],"legs":0},
    {"id":12,"name":"npc_dota_hero_phantom_lancer","localized_name":"Phantom Lancer","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Escape","Pusher","Nuker"],"legs":2},
    {"id":13,"name":"npc_dota_hero_puck","localized_name":"Puck","primary_attr":"int","attack_type":"Ranged","roles":["Initiator","Disabler","Escape","Nuker"],"legs":2},
    {"id":14,"name":"npc_dota_hero_pudge","localized_name":"Pudge","primary_attr":"str","attack_type":"Melee","roles":["Disabler","Initiator","Durable","Nuker"],"legs":2},
    {"id":15,"name":"npc_dota_hero_razor","localized_name":"Razor","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Durable","Nuker","Pusher"],"legs":0},
    {"id":16,"name":"npc_dota_hero_sand_king","localized_name":"Sand King","primary_attr":"all","attack_type":"Melee","roles":["Initiator","Disabler","Support","Nuker","Escape"],"legs":6},
    {"id":17,"name":"npc_dota_hero_storm_spirit","localized_name":"Storm Spirit","primary_attr":"int","attack_type":"Ranged","roles":["Carry","Escape","Nuker","Initiator","Disabler"],"legs":2},
    {"id":18,"name":"npc_dota_hero_sven","localized_name":"Sven","primary_attr":"str","attack_type":"Melee","roles":["Carry","Disabler","Initiator","Durable","Nuker"],"legs":2},
    {"id":19,"name":"npc_dota_hero_tiny","localized_name":"Tiny","primary_attr":"str","attack_type":"Melee","roles":["Carry","Nuker","Pusher","Initiator","Durable","Disabler"],"legs":2},
    {"id":20,"name":"npc_dota_hero_vengefulspirit","localized_name":"Vengeful Spirit","primary_attr":"agi","attack_type":"Ranged","roles":["Support","Initiator","Disabler","Nuker","Escape"],"legs":2},
    {"id":21,"name":"npc_dota_hero_windrunner","localized_name":"Windranger","primary_attr":"all","attack_type":"Ranged","roles":["Carry","Support","Disabler","Escape","Nuker"],"legs":2},
    {"id":22,"name":"npc_dota_hero_zuus","localized_name":"Zeus","primary_attr":"int","attack_type":"Ranged","roles":["Nuker","Carry"],"legs":2},
    {"id":23,"name":"npc_dota_hero_kunkka","localized_name":"Kunkka","primary_attr":"str","attack_type":"Melee","roles":["Carry","Support","Disabler","Initiator","Durable","Nuker"],"legs":2},
    {"id":25,"name":"npc_dota_hero_lina","localized_name":"Lina","primary_attr":"int","attack_type":"Ranged","roles":["Support","Carry","Nuker","Disabler"],"legs":2},
    {"id":26,"name":"npc_dota_hero_lion","localized_name":"Lion","primary_attr":"int","attack_type":"Ranged","roles":["Support","Disabler","Nuker","Initiator"],"legs":2},
    {"id":27,"name":"npc_dota_hero_shadow_shaman","localized_name":"Shadow Shaman","primary_attr":"int","attack_type":"Ranged","roles":["Support","Pusher","Disabler","Nuker","Initiator"],"legs":2},
    {"id":28,"name":"npc_dota_hero_slardar","localized_name":"Slardar","primary_attr":"str","attack_type":"Melee","roles":["Carry","Durable","Initiator","Disabler","Escape"],"legs":0},
    {"id":29,"name":"npc_dota_hero_tidehunter","localized_name":"Tidehunter","primary_attr":"str","attack_type":"Melee","roles":["Initiator","Durable","Disabler","Nuker","Carry"],"legs":2},
    {"id":30,"name":"npc_dota_hero_witch_doctor","localized_name":"Witch Doctor","primary_attr":"int","attack_type":"Ranged","roles":["Support","Nuker","Disabler"],"legs":2},
    {"id":31,"name":"npc_dota_hero_lich","localized_name":"Lich","primary_attr":"int","attack_type":"Ranged","roles":["Support","Nuker"],"legs":2},
    {"id":32,"name":"npc_dota_hero_riki","localized_name":"Riki","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Escape","Disabler"],"legs":2},
    {"id":33,"name":"npc_dota_hero_enigma","localized_name":"Enigma","primary_attr":"all","attack_type":"Ranged","roles":["Disabler","Initiator","Pusher"],"legs":0},
    {"id":34,"name":"npc_dota_hero_tinker","localized_name":"Tinker","primary_attr":"int","attack_type":"Ranged","roles":["Carry","Nuker","Pusher"],"legs":2},
    {"id":35,"name":"npc_dota_hero_sniper","localized_name":"Sniper","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Nuker"],"legs":2},
    {"id":36,"name":"npc_dota_hero_necrolyte","localized_name":"Necrophos","primary_attr":"int","attack_type":"Ranged","roles":["Carry","Nuker","Durable","Disabler"],"legs":2},
    {"id":37,"name":"npc_dota_hero_warlock","localized_name":"Warlock","primary_attr":"int","attack_type":"Ranged","roles":["Support","Initiator","Disabler"],"legs":2},
    {"id":38,"name":"npc_dota_hero_beastmaster","localized_name":"Beastmaster","primary_attr":"all","attack_type":"Melee","roles":["Initiator","Disabler","Durable","Nuker"],"legs":2},
    {"id":39,"name":"npc_dota_hero_queenofpain","localized_name":"Queen of Pain","primary_attr":"int","attack_type":"Ranged","roles":["Carry","Nuker","Escape"],"legs":2},
    {"id":40,"name":"npc_dota_hero_venomancer","localized_name":"Venomancer","primary_attr":"all","attack_type":"Ranged","roles":["Support","Nuker","Initiator","Pusher","Disabler"],"legs":0},
    {"id":41,"name":"npc_dota_hero_faceless_void","localized_name":"Faceless Void","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Initiator","Disabler","Escape","Durable"],"legs":2},
    {"id":42,"name":"npc_dota_hero_skeleton_king","localized_name":"Wraith King","primary_attr":"str","attack_type":"Melee","roles":["Carry","Support","Durable","Disabler","Initiator"],"legs":2},
    {"id":43,"name":"npc_dota_hero_death_prophet","localized_name":"Death Prophet","primary_attr":"all","attack_type":"Ranged","roles":["Carry","Pusher","Nuker","Disabler"],"legs":2},
    {"id":44,"name":"npc_dota_hero_phantom_assassin","localized_name":"Phantom Assassin","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Escape"],"legs":2},
    {"id":45,"name":"npc_dota_hero_pugna","localized_name":"Pugna","primary_attr":"int","attack_type":"Ranged","roles":["Nuker","Pusher"],"legs":2},
    {"id":46,"name":"npc_dota_hero_templar_assassin","localized_name":"Templar Assassin","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Escape"],"legs":2},
    {"id":47,"name":"npc_dota_hero_viper","localized_name":"Viper","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Durable","Initiator","Disabler"],"legs":0},
    {"id":48,"name":"npc_dota_hero_luna","localized_name":"Luna","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Nuker","Pusher"],"legs":2},
    {"id":49,"name":"npc_dota_hero_dragon_knight","localized_name":"Dragon Knight","primary_attr":"str","attack_type":"Melee","roles":["Carry","Pusher","Durable","Disabler","Initiator","Nuker"],"legs":2},
    {"id":50,"name":"npc_dota_hero_dazzle","localized_name":"Dazzle","primary_attr":"all","attack_type":"Ranged","roles":["Support","Nuker","Disabler"],"legs":2},
    {"id":51,"name":"npc_dota_hero_rattletrap","localized_name":"Clockwerk","primary_attr":"str","attack_type":"Melee","roles":["Initiator","Disabler","Durable","Nuker"],"legs":2},
    {"id":52,"name":"npc_dota_hero_leshrac","localized_name":"Leshrac","primary_attr":"int","attack_type":"Ranged","roles":["Carry","Support","Nuker","Pusher","Disabler"],"legs":4},
    {"id":53,"name":"npc_dota_hero_furion","localized_name":"Nature's Prophet","primary_attr":"all","attack_type":"Ranged","roles":["Carry","Pusher","Escape","Nuker"],"legs":2},
    {"id":54,"name":"npc_dota_hero_life_stealer","localized_name":"Lifestealer","primary_attr":"str","attack_type":"Melee","roles":["Carry","Durable","Escape","Disabler"],"legs":2},
    {"id":55,"name":"npc_dota_hero_dark_seer","localized_name":"Dark Seer","primary_attr":"int","attack_type":"Melee","roles":["Initiator","Escape","Disabler"],"legs":2},
    {"id":56,"name":"npc_dota_hero_clinkz","localized_name":"Clinkz","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Escape","Pusher"],"legs":2},
    {"id":57,"name":"npc_dota_hero_omniknight","localized_name":"Omniknight","primary_attr":"str","attack_type":"Melee","roles":["Support","Durable","Nuker"],"legs":2},
    {"id":58,"name":"npc_dota_hero_enchantress","localized_name":"Enchantress","primary_attr":"int","attack_type":"Ranged","roles":["Support","Pusher","Durable","Disabler"],"legs":4},
    {"id":59,"name":"npc_dota_hero_huskar","localized_name":"Huskar","primary_attr":"str","attack_type":"Ranged","roles":["Carry","Durable","Initiator"],"legs":2},
    {"id":60,"name":"npc_dota_hero_night_stalker","localized_name":"Night Stalker","primary_attr":"str","attack_type":"Melee","roles":["Carry","Initiator","Durable","Disabler","Nuker"],"legs":2},
    {"id":61,"name":"npc_dota_hero_broodmother","localized_name":"Broodmother","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Pusher","Escape","Nuker"],"legs":8},
    {"id":62,"name":"npc_dota_hero_bounty_hunter","localized_name":"Bounty Hunter","primary_attr":"agi","attack_type":"Melee","roles":["Escape","Nuker"],"legs":2},
    {"id":63,"name":"npc_dota_hero_weaver","localized_name":"Weaver","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Escape"],"legs":4},
    {"id":64,"name":"npc_dota_hero_jakiro","localized_name":"Jakiro","primary_attr":"int","attack_type":"Ranged","roles":["Support","Nuker","Pusher","Disabler"],"legs":2},
    {"id":65,"name":"npc_dota_hero_batrider","localized_name":"Batrider","primary_attr":"all","attack_type":"Ranged","roles":["Initiator","Disabler","Escape"],"legs":2},
    {"id":66,"name":"npc_dota_hero_chen","localized_name":"Chen","primary_attr":"int","attack_type":"Ranged","roles":["Support","Pusher"],"legs":2},
    {"id":67,"name":"npc_dota_hero_spectre","localized_name":"Spectre","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Durable","Escape"],"legs":0},
    {"id":68,"name":"npc_dota_hero_ancient_apparition","localized_name":"Ancient Apparition","primary_attr":"int","attack_type":"Ranged","roles":["Support","Disabler","Nuker"],"legs":0},
    {"id":69,"name":"npc_dota_hero_doom_bringer","localized_name":"Doom","primary_attr":"str","attack_type":"Melee","roles":["Carry","Disabler","Initiator","Durable","Nuker"],"legs":2},
    {"id":70,"name":"npc_dota_hero_ursa","localized_name":"Ursa","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Durable","Disabler"],"legs":2},
    {"id":71,"name":"npc_dota_hero_spirit_breaker","localized_name":"Spirit Breaker","primary_attr":"str","attack_type":"Melee","roles":["Carry","Initiator","Disabler","Durable","Escape"],"legs":2},
    {"id":72,"name":"npc_dota_hero_gyrocopter","localized_name":"Gyrocopter","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Nuker","Disabler"],"legs":2},
    {"id":73,"name":"npc_dota_hero_alchemist","localized_name":"Alchemist","primary_attr":"str","attack_type":"Melee","roles":["Carry","Support","Durable","Disabler","Initiator","Nuker"],"legs":2},
    {"id":74,"name":"npc_dota_hero_invoker","localized_name":"Invoker","primary_attr":"int","attack_type":"Ranged","roles":["Carry","Nuker","Disabler","Escape","Pusher"],"legs":2},
    {"id":75,"name":"npc_dota_hero_silencer","localized_name":"Silencer","primary_attr":"int","attack_type":"Ranged","roles":["Carry","Support","Disabler","Initiator","Nuker"],"legs":2},
    {"id":76,"name":"npc_dota_hero_obsidian_destroyer","localized_name":"Outworld Destroyer","primary_attr":"int","attack_type":"Ranged","roles":["Carry","Nuker","Disabler"],"legs":4},
    {"id":77,"name":"npc_dota_hero_brewmaster","localized_name":"Brewmaster","primary_attr":"all","attack_type":"Melee","roles":["Carry","Initiator","Durable","Disabler","Nuker"],"legs":2},
    {"id":79,"name":"npc_dota_hero_shadow_demon","localized_name":"Shadow Demon","primary_attr":"int","attack_type":"Ranged","roles":["Support","Disabler","Initiator","Nuker"],"legs":2},
    {"id":80,"name":"npc_dota_hero_lone_druid","localized_name":"Lone Druid","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Pusher","Durable"],"legs":2},
    {"id":81,"name":"npc_dota_hero_chaos_knight","localized_name":"Chaos Knight","primary_attr":"str","attack_type":"Melee","roles":["Carry","Disabler","Durable","Pusher","Initiator"],"legs":2},
    {"id":82,"name":"npc_dota_hero_meepo","localized_name":"Meepo","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Escape","Nuker","Disabler","Initiator","Pusher"],"legs":2},
    {"id":83,"name":"npc_dota_hero_treant","localized_name":"Treant Protector","primary_attr":"str","attack_type":"Melee","roles":["Support","Initiator","Durable","Disabler","Escape"],"legs":2},
    {"id":84,"name":"npc_dota_hero_ogre_magi","localized_name":"Ogre Magi","primary_attr":"str","attack_type":"Melee","roles":["Support","Nuker","Disabler","Durable","Initiator"],"legs":2},
    {"id":85,"name":"npc_dota_hero_undying","localized_name":"Undying","primary_attr":"str","attack_type":"Melee","roles":["Support","Durable","Disabler","Nuker"],"legs":2},
    {"id":86,"name":"npc_dota_hero_rubick","localized_name":"Rubick","primary_attr":"int","attack_type":"Ranged","roles":["Support","Disabler","Nuker"],"legs":2},
    {"id":87,"name":"npc_dota_hero_disruptor","localized_name":"Disruptor","primary_attr":"int","attack_type":"Ranged","roles":["Support","Disabler","Nuker","Initiator"],"legs":2},
    {"id":88,"name":"npc_dota_hero_nyx_assassin","localized_name":"Nyx Assassin","primary_attr":"all","attack_type":"Melee","roles":["Disabler","Nuker","Initiator","Escape"],"legs":6},
    {"id":89,"name":"npc_dota_hero_naga_siren","localized_name":"Naga Siren","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Support","Pusher","Disabler","Initiator","Escape"],"legs":0},
    {"id":90,"name":"npc_dota_hero_keeper_of_the_light","localized_name":"Keeper of the Light","primary_attr":"int","attack_type":"Ranged","roles":["Support","Nuker","Disabler"],"legs":2},
    {"id":91,"name":"npc_dota_hero_wisp","localized_name":"Io","primary_attr":"all","attack_type":"Ranged","roles":["Support","Escape","Nuker"],"legs":0},
    {"id":92,"name":"npc_dota_hero_visage","localized_name":"Visage","primary_attr":"all","attack_type":"Ranged","roles":["Support","Nuker","Durable","Disabler","Pusher"],"legs":2},
    {"id":93,"name":"npc_dota_hero_slark","localized_name":"Slark","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Escape","Disabler","Nuker"],"legs":2},
    {"id":94,"name":"npc_dota_hero_medusa","localized_name":"Medusa","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Disabler","Durable"],"legs":0},
    {"id":95,"name":"npc_dota_hero_troll_warlord","localized_name":"Troll Warlord","primary_attr":"agi","attack_type":"Ranged","roles":["Carry","Pusher","Disabler","Durable"],"legs":2},
    {"id":96,"name":"npc_dota_hero_centaur","localized_name":"Centaur Warrunner","primary_attr":"str","attack_type":"Melee","roles":["Durable","Initiator","Disabler","Nuker","Escape"],"legs":4},
    {"id":97,"name":"npc_dota_hero_magnataur","localized_name":"Magnus","primary_attr":"all","attack_type":"Melee","roles":["Initiator","Disabler","Nuker","Escape"],"legs":4},
    {"id":98,"name":"npc_dota_hero_shredder","localized_name":"Timbersaw","primary_attr":"str","attack_type":"Melee","roles":["Nuker","Durable","Escape"],"legs":2},
    {"id":99,"name":"npc_dota_hero_bristleback","localized_name":"Bristleback","primary_attr":"str","attack_type":"Melee","roles":["Carry","Durable","Initiator","Nuker"],"legs":2},
    {"id":100,"name":"npc_dota_hero_tusk","localized_name":"Tusk","primary_attr":"str","attack_type":"Melee","roles":["Initiator","Disabler","Nuker"],"legs":2},
    {"id":101,"name":"npc_dota_hero_skywrath_mage","localized_name":"Skywrath Mage","primary_attr":"int","attack_type":"Ranged","roles":["Support","Nuker","Disabler"],"legs":2},
    {"id":102,"name":"npc_dota_hero_abaddon","localized_name":"Abaddon","primary_attr":"all","attack_type":"Melee","roles":["Support","Carry","Durable"],"legs":2},
    {"id":103,"name":"npc_dota_hero_elder_titan","localized_name":"Elder Titan","primary_attr":"str","attack_type":"Melee","roles":["Initiator","Disabler","Nuker","Durable"],"legs":2},
    {"id":104,"name":"npc_dota_hero_legion_commander","localized_name":"Legion Commander","primary_attr":"str","attack_type":"Melee","roles":["Carry","Disabler","Initiator","Durable","Nuker"],"legs":2},
    {"id":105,"name":"npc_dota_hero_techies","localized_name":"Techies","primary_attr":"all","attack_type":"Ranged","roles":["Nuker","Disabler"],"legs":6},
    {"id":106,"name":"npc_dota_hero_ember_spirit","localized_name":"Ember Spirit","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Escape","Nuker","Disabler","Initiator"],"legs":2},
    {"id":107,"name":"npc_dota_hero_earth_spirit","localized_name":"Earth Spirit","primary_attr":"str","attack_type":"Melee","roles":["Nuker","Escape","Disabler","Initiator","Durable"],"legs":2},
    {"id":108,"name":"npc_dota_hero_abyssal_underlord","localized_name":"Underlord","primary_attr":"str","attack_type":"Melee","roles":["Support","Nuker","Disabler","Durable","Escape"],"legs":2},
    {"id":109,"name":"npc_dota_hero_terrorblade","localized_name":"Terrorblade","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Pusher","Nuker"],"legs":2},
    {"id":110,"name":"npc_dota_hero_phoenix","localized_name":"Phoenix","primary_attr":"str","attack_type":"Ranged","roles":["Support","Nuker","Initiator","Escape","Disabler"],"legs":2},
    {"id":111,"name":"npc_dota_hero_oracle","localized_name":"Oracle","primary_attr":"int","attack_type":"Ranged","roles":["Support","Nuker","Disabler","Escape"],"legs":2},
    {"id":112,"name":"npc_dota_hero_winter_wyvern","localized_name":"Winter Wyvern","primary_attr":"int","attack_type":"Ranged","roles":["Support","Disabler","Nuker"],"legs":2},
    {"id":113,"name":"npc_dota_hero_arc_warden","localized_name":"Arc Warden","primary_attr":"all","attack_type":"Ranged","roles":["Carry","Escape","Nuker"],"legs":2},
    {"id":114,"name":"npc_dota_hero_monkey_king","localized_name":"Monkey King","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Escape","Disabler","Initiator"],"legs":2},
    {"id":119,"name":"npc_dota_hero_dark_willow","localized_name":"Dark Willow","primary_attr":"int","attack_type":"Ranged","roles":["Support","Nuker","Disabler","Escape"],"legs":2},
    {"id":120,"name":"npc_dota_hero_pangolier","localized_name":"Pangolier","primary_attr":"all","attack_type":"Melee","roles":["Carry","Nuker","Disabler","Durable","Escape","Initiator"],"legs":2},
    {"id":121,"name":"npc_dota_hero_grimstroke","localized_name":"Grimstroke","primary_attr":"int","attack_type":"Ranged","roles":["Support","Nuker","Disabler","Escape"],"legs":0},
    {"id":123,"name":"npc_dota_hero_hoodwink","localized_name":"Hoodwink","primary_attr":"agi","attack_type":"Ranged","roles":["Support","Nuker","Escape","Disabler"],"legs":4},
    {"id":126,"name":"npc_dota_hero_void_spirit","localized_name":"Void Spirit","primary_attr":"all","attack_type":"Melee","roles":["Carry","Escape","Nuker","Disabler"],"legs":2},
    {"id":128,"name":"npc_dota_hero_snapfire","localized_name":"Snapfire","primary_attr":"all","attack_type":"Ranged","roles":["Support","Nuker","Disabler","Escape"],"legs":2},
    {"id":129,"name":"npc_dota_hero_mars","localized_name":"Mars","primary_attr":"str","attack_type":"Melee","roles":["Carry","Initiator","Disabler","Durable"],"legs":2},
    {"id":131,"name":"npc_dota_hero_ringmaster","localized_name":"Ringmaster","primary_attr":"int","attack_type":"Ranged","roles":["Support","Nuker","Escape","Disabler"],"legs":2},
    {"id":135,"name":"npc_dota_hero_dawnbreaker","localized_name":"Dawnbreaker","primary_attr":"str","attack_type":"Melee","roles":["Carry","Durable"],"legs":2},
    {"id":136,"name":"npc_dota_hero_marci","localized_name":"Marci","primary_attr":"all","attack_type":"Melee","roles":["Support","Carry","Initiator","Disabler","Escape"],"legs":2},
    {"id":137,"name":"npc_dota_hero_primal_beast","localized_name":"Primal Beast","primary_attr":"str","attack_type":"Melee","roles":["Initiator","Durable","Disabler"],"legs":2},
    {"id":138,"name":"npc_dota_hero_muerta","localized_name":"Muerta","primary_attr":"int","attack_type":"Ranged","roles":["Carry","Nuker","Disabler"],"legs":2},
    {"id":145,"name":"npc_dota_hero_kez","localized_name":"Kez","primary_attr":"agi","attack_type":"Melee","roles":["Carry","Escape","Disabler"],"legs":2},
    {"id":155,"name":"npc_dota_hero_largo","localized_name":"Largo","primary_attr":"str","attack_type":"Melee","roles":["Durable","Disabler","Support"],"legs":2},
]

# Role-specific stat profiles
ROLE_PROFILES = {
    "Carry":       {"scaling": (0.75, 0.95), "physical_damage": (0.7, 0.95), "magic_damage": (0.1, 0.4)},
    "Support":     {"save": (0.5, 0.8), "vision": (0.4, 0.7), "magic_damage": (0.3, 0.5)},
    "Initiator":   {"initiation": (0.75, 1.0), "teamfight": (0.7, 0.95)},
    "Disabler":    {"stun": (0.6, 1.0), "disable_count": (2.0, 4.0)},
    "Nuker":       {"magic_damage": (0.65, 1.0), "pure_damage": (0.0, 0.3)},
    "Pusher":      {"push": (0.7, 0.95), "splitpush": (0.5, 0.8)},
    "Escape":      {"mobility": (0.75, 1.0)},
    "Durable":     {"durability": (0.75, 1.0), "base_health": (200, 300)},
}


def get_slug(name):
    """Extract hero slug from 'npc_dota_hero_axe' -> 'axe'."""
    return name.replace("npc_dota_hero_", "")


def generate_hero_stats(hero, rng):
    """Generate deterministic stats for a hero."""
    attr = hero["primary_attr"]
    is_melee = hero["attack_type"].lower() == "melee"

    stats = {
        "base_health": 200,
        "base_mana": 75,
        "attack_rate": 1.7,
        "turn_rate": 0.6,
        "cm_enabled": True,
        "projectile_speed": 0 if is_melee else 900,
        "move_speed": rng.next_int(295, 330),
        "attack_range": rng.next_int(140, 160) if is_melee else rng.next_int(380, 620),
    }

    if attr == "str":
        stats["base_str"] = rng.next_float(22, 30)
        stats["str_gain"] = rng.next_float(2.5, 3.5)
        stats["base_agi"] = rng.next_float(12, 20)
        stats["agi_gain"] = rng.next_float(1.5, 2.5)
        stats["base_int"] = rng.next_float(12, 20)
        stats["int_gain"] = rng.next_float(1.2, 2.0)
        stats["base_armor"] = rng.next_float(2.0, 4.0)
        stats["base_attack_min"] = rng.next_int(22, 35)
        stats["base_attack_max"] = rng.next_int(28, 42)
    elif attr == "agi":
        stats["base_agi"] = rng.next_float(22, 30)
        stats["agi_gain"] = rng.next_float(2.5, 3.5)
        stats["base_str"] = rng.next_float(16, 22)
        stats["str_gain"] = rng.next_float(1.8, 2.8)
        stats["base_int"] = rng.next_float(12, 18)
        stats["int_gain"] = rng.next_float(1.2, 1.8)
        stats["base_armor"] = rng.next_float(2.0, 4.0)
        stats["base_attack_min"] = rng.next_int(20, 30)
        stats["base_attack_max"] = rng.next_int(26, 38)
    elif attr == "int":
        stats["base_int"] = rng.next_float(22, 30)
        stats["int_gain"] = rng.next_float(2.5, 3.5)
        stats["base_str"] = rng.next_float(16, 20)
        stats["str_gain"] = rng.next_float(1.8, 2.5)
        stats["base_agi"] = rng.next_float(12, 18)
        stats["agi_gain"] = rng.next_float(1.2, 2.0)
        stats["base_armor"] = rng.next_float(1.0, 3.0)
        stats["base_attack_min"] = rng.next_int(18, 28)
        stats["base_attack_max"] = rng.next_int(24, 36)
    else:  # "all" (Universal)
        stats["base_str"] = rng.next_float(19, 23)
        stats["str_gain"] = rng.next_float(2.0, 2.5)
        stats["base_agi"] = rng.next_float(19, 23)
        stats["agi_gain"] = rng.next_float(2.0, 2.5)
        stats["base_int"] = rng.next_float(19, 23)
        stats["int_gain"] = rng.next_float(2.0, 2.5)
        stats["base_armor"] = rng.next_float(1.5, 3.5)
        stats["base_attack_min"] = rng.next_int(20, 30)
        stats["base_attack_max"] = rng.next_int(26, 38)

    return stats


def generate_hero(hero, rng):
    """Generate a single hero entry for heroes.json."""
    slug = get_slug(hero["name"])
    stats = generate_hero_stats(hero, rng)

    hero_entry = {
        "id": hero["id"],
        "name": hero["name"],
        "localized_name": hero["localized_name"],
        "primary_attr": hero["primary_attr"],
        "attack_type": hero["attack_type"].lower(),
        "roles": hero["roles"],
        "legs": hero["legs"],
        "img": f"/apps/dota2/images/heroes/{slug}_full.png",
        "icon": f"/apps/dota2/images/heroes/{slug}_icon.png",
    }
    hero_entry.update(stats)
    return hero_entry


def generate_hero_semantics(hero, rng, hero_data):
    """Generate a single hero_semantics entry."""
    roles = set(hero["roles"])
    is_melee = hero["attack_type"].lower() == "melee"

    # Base values
    initiation = 0.3
    stun = 0.2
    save = 0.2
    push = 0.2
    scaling = 0.3
    mobility = 0.3
    durability = 0.4
    teamfight = 0.3
    splitpush = 0.1
    vision = 0.3
    magic_damage = 0.3
    physical_damage = 0.4
    pure_damage = 0.0
    disable_count = 1.0

    # Apply role-specific boosts
    if "Initiator" in roles:
        initiation = rng.next_float(0.7, 1.0)
        teamfight = rng.next_float(0.7, 0.95)
    if "Disabler" in roles:
        stun = rng.next_float(0.5, 1.0)
        disable_count = rng.next_float(2.0, 4.0)
    if "Carry" in roles:
        scaling = rng.next_float(0.7, 0.95)
        physical_damage = rng.next_float(0.6, 0.95)
    if "Support" in roles:
        save = rng.next_float(0.4, 0.8)
        vision = rng.next_float(0.3, 0.6)
    if "Nuker" in roles:
        magic_damage = rng.next_float(0.6, 1.0)
    if "Pusher" in roles:
        push = rng.next_float(0.7, 0.95)
        splitpush = rng.next_float(0.5, 0.8)
    if "Escape" in roles:
        mobility = rng.next_float(0.7, 1.0)
    if "Durable" in roles:
        durability = rng.next_float(0.7, 1.0)

    # Lane strength based on primary attribute and role
    attr = hero["primary_attr"]
    if "Carry" in roles and "Pusher" in roles:
        # Push carries are early-mid
        lane = {"early": rng.next_float(0.5, 0.7), "mid": rng.next_float(0.7, 0.9), "late": rng.next_float(0.6, 0.8)}
    elif "Carry" in roles and scaling > 0.8:
        # Hard carries are late
        lane = {"early": rng.next_float(0.3, 0.5), "mid": rng.next_float(0.5, 0.7), "late": rng.next_float(0.75, 0.95)}
    elif "Carry" in roles:
        # Regular carries are mid-late
        lane = {"early": rng.next_float(0.4, 0.6), "mid": rng.next_float(0.65, 0.85), "late": rng.next_float(0.65, 0.85)}
    elif "Support" in roles and "Disabler" in roles:
        # Support disablers are early-mid
        lane = {"early": rng.next_float(0.6, 0.8), "mid": rng.next_float(0.6, 0.8), "late": rng.next_float(0.3, 0.5)}
    elif "Support" in roles:
        # Supports are early
        lane = {"early": rng.next_float(0.65, 0.85), "mid": rng.next_float(0.5, 0.7), "late": rng.next_float(0.3, 0.5)}
    elif "Initiator" in roles and "Durable" in roles:
        # Tank initiators are mid
        lane = {"early": rng.next_float(0.4, 0.6), "mid": rng.next_float(0.65, 0.85), "late": rng.next_float(0.5, 0.7)}
    elif "Initiator" in roles:
        # Initiators are early-mid
        lane = {"early": rng.next_float(0.5, 0.7), "mid": rng.next_float(0.6, 0.85), "late": rng.next_float(0.4, 0.6)}
    elif "Nuker" in roles and attr == "int":
        # Int nukers are early-mid
        lane = {"early": rng.next_float(0.6, 0.8), "mid": rng.next_float(0.65, 0.85), "late": rng.next_float(0.35, 0.55)}
    elif attr == "int":
        # Other int heroes are early
        lane = {"early": rng.next_float(0.55, 0.75), "mid": rng.next_float(0.5, 0.7), "late": rng.next_float(0.35, 0.55)}
    elif attr == "agi" and "Escape" in roles:
        # Escape agi are mid-late
        lane = {"early": rng.next_float(0.4, 0.55), "mid": rng.next_float(0.6, 0.8), "late": rng.next_float(0.65, 0.85)}
    elif attr == "agi":
        # Other agi are mid-late
        lane = {"early": rng.next_float(0.35, 0.55), "mid": rng.next_float(0.6, 0.8), "late": rng.next_float(0.65, 0.85)}
    elif attr == "str" and "Durable" in roles:
        # Tanky str are mid
        lane = {"early": rng.next_float(0.45, 0.65), "mid": rng.next_float(0.6, 0.8), "late": rng.next_float(0.45, 0.65)}
    elif attr == "str":
        # Other str are early-mid
        lane = {"early": rng.next_float(0.5, 0.7), "mid": rng.next_float(0.6, 0.8), "late": rng.next_float(0.4, 0.6)}
    else:  # all (Universal)
        lane = {"early": rng.next_float(0.45, 0.65), "mid": rng.next_float(0.55, 0.75), "late": rng.next_float(0.55, 0.75)}

    # Tempo profile
    if lane["early"] >= lane["mid"] and lane["early"] >= lane["late"]:
        tempo = "early"
    elif lane["late"] >= lane["mid"] and lane["late"] >= lane["early"]:
        tempo = "late"
    else:
        tempo = "mid"

    # Roles probability
    roles_prob = {}
    for i, role in enumerate(hero["roles"]):
        # Primary roles (first ones) get higher probability
        if i == 0:
            roles_prob[role] = rng.next_float(0.8, 1.0)
        elif i == 1:
            roles_prob[role] = rng.next_float(0.7, 0.9)
        else:
            roles_prob[role] = rng.next_float(0.5, 0.75)

    # Clamp values
    def clamp(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v))

    semantics = {
        "hero_id": hero["id"],
        "initiation": clamp(initiation),
        "stun": clamp(stun),
        "save": clamp(save),
        "push": clamp(push),
        "scaling": clamp(scaling),
        "mobility": clamp(mobility),
        "durability": clamp(durability),
        "teamfight": clamp(teamfight),
        "splitpush": clamp(splitpush),
        "vision": clamp(vision),
        "magic_damage": clamp(magic_damage),
        "physical_damage": clamp(physical_damage),
        "pure_damage": clamp(pure_damage),
        "disable_count": clamp(disable_count, 0.0, 5.0),
        "lane_strength": lane,
        "tempo_profile": tempo,
        "roles_probability": roles_prob,
    }

    return semantics


def main():
    # Deduplicate by id (id 78 appears twice)
    seen_ids = set()
    heroes = []
    for h in HEROES_API:
        if h["id"] not in seen_ids:
            seen_ids.add(h["id"])
            heroes.append(h)

    print(f"Processing {len(heroes)} unique heroes")

    heroes_json = []
    semantics_json = []

    for hero in heroes:
        rng = LCG(hero["id"])
        heroes_json.append(generate_hero(hero, rng))
        rng2 = LCG(hero["id"] + 10000)
        semantics_json.append(generate_hero_semantics(hero, rng2, heroes_json[-1]))

    # Sort by id
    heroes_json.sort(key=lambda h: h["id"])
    semantics_json.sort(key=lambda s: s["hero_id"])

    # Write files
    output_dir = os.path.dirname(os.path.abspath(__file__))
    heroes_path = os.path.join(output_dir, "heroes.json")
    semantics_path = os.path.join(output_dir, "hero_semantics.json")

    with open(heroes_path, "w", encoding="utf-8") as f:
        json.dump(heroes_json, f, indent=2, ensure_ascii=False)

    with open(semantics_path, "w", encoding="utf-8") as f:
        json.dump(semantics_json, f, indent=2, ensure_ascii=False)

    print(f"Written {heroes_path} ({len(heroes_json)} heroes)")
    print(f"Written {semantics_path} ({len(semantics_json)} semantics)")

    # Verify
    for h in heroes_json:
        assert h["attack_type"] in ("melee", "ranged"), f"Bad attack_type: {h['attack_type']} for {h['localized_name']}"
        assert h["primary_attr"] in ("str", "agi", "int", "all"), f"Bad attr: {h['primary_attr']}"
        assert "/apps/dota2/images/heroes/" in h["img"]
        assert "_full.png" in h["img"]
        assert "_icon.png" in h["icon"]

    for s in semantics_json:
        assert 0 <= s["initiation"] <= 1
        assert 0 <= s["stun"] <= 1
        assert 0 <= s["save"] <= 1
        assert 0 <= s["push"] <= 1
        assert 0 <= s["scaling"] <= 1
        assert 0 <= s["mobility"] <= 1
        assert 0 <= s["durability"] <= 1
        assert 0 <= s["teamfight"] <= 1
        assert 0 <= s["splitpush"] <= 1
        assert 0 <= s["vision"] <= 1
        assert 0 <= s["magic_damage"] <= 1
        assert 0 <= s["physical_damage"] <= 1
        assert 0 <= s["pure_damage"] <= 1
        assert 0 <= s["disable_count"] <= 5
        assert "early" in s["lane_strength"]
        assert "mid" in s["lane_strength"]
        assert "late" in s["lane_strength"]
        assert s["tempo_profile"] in ("early", "mid", "late")
        assert isinstance(s["roles_probability"], dict)

    print("All validations passed!")


if __name__ == "__main__":
    main()
