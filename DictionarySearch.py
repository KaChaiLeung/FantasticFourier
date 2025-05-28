#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:20:39 2025

@author: vee
"""
import json
with open('/Users/vee/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Summer Project/nsynth-train.json') as f:
    metadata = json.load(f)
clips = {}

for clip_id, info in metadata.items():
    instrument_name = info['instrument_family_str']
    filename = str(clip_id) + '.wav'
    clips.update({filename : instrument_name})


clips={filename: instrument_name}

