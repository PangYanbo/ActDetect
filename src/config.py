# src/config.py
from __future__ import annotations

# Project root is added in notebooks via sys.path.append("/Users/pang/Codes/GISRUK")

ACTIVITIES = ["HOME", "WORK", "STUDY", "PURCHASE", "LEISURE", "HEALTH", "OTHER"]
K = len(ACTIVITIES)
act2i = {a: i for i, a in enumerate(ACTIVITIES)}
i2act = {i: a for a, i in act2i.items()}

# Pattern abbreviations (for activity motif plots)
ABBR = {"HOME":"H","WORK":"W","STUDY":"S","PURCHASE":"P","LEISURE":"L","HEALTH":"He","OTHER":"O"}

# Time zones
TZ_PARIS = "Europe/Paris"
TZ_LONDON = "Europe/London"

# Temporal bins (used by HMM priors)
TIME_BINS = [(0,360),(360,600),(600,960),(960,1200),(1200,1440)]  # minutes from 00:00
DUR_BINS  = [(0,10),(10,30),(30,120),(120,360),(360,720),(720,1e9)]

# Default file names (your confirmed ones)
PARIS_TRIPS_PARQUET = "paris_trips_h3.parquet"
PARIS_POI_PARQUET   = "fr_hex_poi_res10.parquet"
UK_POI_PARQUET      = "uk_hex_poi_h10.parquet"