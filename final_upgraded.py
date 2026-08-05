"""
Delhi EV Charging Station Adoption Simulator
=============================================

A Streamlit dashboard that simulates how many GREY (non-EV) houses convert to
GREEN (EV-adopted) as it spreads outward from a charging station placed at
the center of a chosen Delhi locality, cycle by cycle (1 cycle = 1 year),
based on which is closer: the charging station or the nearest petrol pump.

HOW IT WORKS
------------
1. User picks a locality in Delhi from the dashboard dropdown.
2. The locality is geocoded (Nominatim) to a center lat/lon.
3. A FIXED-size area around that center is queried from OpenStreetMap
   (via the Overpass API) to pull:
       - houses / residential buildings
       - petrol pumps (amenity=fuel)
       - the nearest substation (power=substation) to the center, then
         the nearest open area (leisure=park) to THAT substation
4. Every house starts as one of only two states - there is no "red" tier:
       - GREY  = non-EV / not yet adopted
       - GREEN = already EV-adopted
   20% of houses are randomly seeded as GREEN to start; the rest are GREY.
   There is no cap on how many houses are fetched/simulated - every
   building Overpass returns in range is used. For map display only,
   houses are rolled up into groups of 50 (one dot per group, colored by
   majority status) so the map stays renderable at that scale.
5. The charging station is placed at that nearest-open-area point (not
   the geocoded center, and not the substation itself) - cabling then
   runs the straight-line distance from the substation to that open area.
6. CYCLE LOGIC (number of cycles = number of years, set via a slider,
   always starting at FY2023-24):
       - Cycle 1 uses a 3 km radius around the charging station and is
         DISTANCE-ONLY: every GREY house within that radius just compares
         raw distance to the charging station vs. raw distance to the
         nearest petrol pump - whichever is closer wins, no Rs/km cost
         model is applied yet.
       - From cycle 2 onward, every GREY house within radius has its
         distance to the nearest petrol pump and its distance to the
         charging station each converted into a cost:
             petrol_side = distance_to_pump_km * that cycle's petrol Rs/km
                           (looked up directly from PETROL_COST_PER_KM_BY_YEAR
                           for the cycle's fiscal year - no extra multiplier)
             ev_side     = distance_to_station_km * EV_CHARGE_PER_KM (Rs 1.2/km)
         If ev_side is cheaper, the house turns GREEN.
       - Each cycle's petrol Rs/km comes straight from PETROL_COST_PER_KM_BY_YEAR
         (see get_petrol_cost_per_km_for_cycle) - a fixed lookup table of
         actual + projected Rs/km values by fiscal year.
       - Petrol pumps within range are highlighted on the map.
       - Cycle 2 starts from the state left behind by cycle 1 (the newly
         GREEN houses are now the "already adopted" baseline) and the
         radius grows by 2 km (5 km), then 7 km, etc.
7. The dashboard shows the map after each cycle, the petrol-price chart
   that fed the cost comparison, plus a summary table/chart of how many
   houses are grey vs. green over time - i.e. how many non-EV houses
   converted to EV each year.

REQUIRED LIBRARIES
-------------------
    pip install streamlit folium streamlit-folium requests geopy matplotlib

RUN WITH
--------
    streamlit run delhi_ev_dashboard.py
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor

import folium
import matplotlib.pyplot as plt
import requests
import streamlit as st
from geopy.distance import geodesic
from streamlit_folium import st_folium

# --------------------------------------------------------------------------
# CONSTANTS / CONFIG
# --------------------------------------------------------------------------

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Multiple public Overpass mirrors - the free servers occasionally time out
# (504) under load, so we retry across mirrors instead of failing outright.
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
HEADERS = {"User-Agent": "delhi-ev-dashboard/1.0 (educational simulation)"}

# The size of the map area pulled from OSM is FIXED regardless of locality
# (same query shape every time) - but we cap it dynamically based on how many
# cycles were requested, since fetching a huge radius for a 1-cycle run just
# wastes time and is what was causing Overpass to time out.
MAX_FETCH_RADIUS_KM = 5
BASE_CYCLE_RADIUS_KM = 3
CYCLE_RADIUS_INCREMENT_KM = 2
# --------------------------------------------------------------------------
# PETROL COST PER KM (used for the per-cycle cost comparison below)
# --------------------------------------------------------------------------
# Rs/km readings straight off the "Petrol cost per km, India (Delhi RSP /
# 17.5 km/L)" chart - actual FY2019-20 to FY2024-25 (PPAC), projected at
# 7%/yr thereafter. Already Rs/km, so no mileage conversion is needed - the
# per-cycle lookup just reads the value for that fiscal year straight out
# of this table.
PETROL_COST_PER_KM_BY_YEAR = {
    "2019-20": 4.15,
    "2020-21": 4.62,
    "2021-22": 5.60,
    "2022-23": 5.59,
    "2023-24": 5.52,
    "2024-25": 5.41,
    "2025-26": 5.79,
    "2026-27": 6.20,
    "2027-28": 6.63,
    "2028-29": 7.09,
    "2029-30": 7.59,
    "2030-31": 8.12,
    "2031-32": 8.69,
    "2032-33": 9.30,
    "2033-34": 9.95,
    "2034-35": 10.65,
}

# Flat EV running cost, rupees per km (used directly against distance to
# the charging station in the cost comparison).
EV_CHARGE_PER_KM = 1.2

# Charging-station installation cost model.
# - CABLE_COST_PER_METER: cost to run cabling from the charging station to
#   the nearest substation, charged per metre of straight-line distance.
# - BASE_SETUP_COST: fixed cost added on top of cabling (Rs 2.47 lakh).
# - STATION_HARDWARE_COST: fixed cost of the charging station hardware/build
#   itself (Rs 33 lakh).
# - ANNUAL_RECURRING_COST: ongoing cost (maintenance/operations) added once
#   per year on top of the one-time setup cost (Rs 13 lakh/year).
CABLE_COST_PER_METER = 2800
BASE_SETUP_COST = 247000       # Rs 2.47 lakh
STATION_HARDWARE_COST = 3300000  # Rs 33 lakh
ANNUAL_RECURRING_COST = 1300000  # Rs 13 lakh per year

# No RED tier - every house starts as either GREY (non-EV / not yet adopted)
# or GREEN (already EV-adopted). The simulation tracks how many GREY houses
# convert to GREEN each cycle.
INITIAL_GREEN_PCT = 0.20
# remaining 80% starts GREY

# No cap on how many houses are fetched/simulated anymore - every building
# Overpass returns within range is used. To keep the map renderable at that
# scale, houses are rolled up into groups of GROUP_SIZE for display: one dot
# per group instead of one dot per house (simulation logic still runs on
# every individual house underneath).
GROUP_SIZE = 50

# Fraction of a group's houses that must be GREEN before the group's map
# dot itself is shown as green (see group_houses_for_map).
GROUP_GREEN_THRESHOLD = 0.75

DELHI_LOCALITIES = [
    "Lajpat Nagar", "Karol Bagh", "Connaught Place", "Rohini",
    "Dwarka", "Saket", "Vasant Kunj", "Hauz Khas", "Pitampura",
    "Janakpuri", "Mayur Vihar", "Preet Vihar", "Shahdara",
    "Chandni Chowk", "Greater Kailash", "Model Town", "Paschim Vihar",
    "Malviya Nagar", "Punjabi Bagh", "Rajouri Garden",
]

GREY = "grey"
GREEN = "green"


# --------------------------------------------------------------------------
# OSM DATA FETCHING (cached so repeated runs don't hammer the API)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def geocode_locality(locality_name: str):
    """Turn a locality name into a (lat, lon) center point using Nominatim."""
    params = {
        "q": f"{locality_name}, Delhi, India",
        "format": "json",
        "limit": 1,
    }
    resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"Could not geocode locality: {locality_name}")
    return float(results[0]["lat"]), float(results[0]["lon"])


def _run_overpass(query: str, max_retries: int = 2):
    """
    POST a query to Overpass, trying each mirror in turn and retrying with a
    short backoff on timeouts/server errors (502/503/504) before giving up.
    """
    last_error = None
    for attempt in range(max_retries):
        for mirror in OVERPASS_MIRRORS:
            try:
                # Timeout updated to 90
                resp = requests.post(mirror, data={"data": query}, headers=HEADERS, timeout=90)
                if resp.status_code in (502, 503, 504):
                    last_error = f"{resp.status_code} from {mirror}"
                    continue
                resp.raise_for_status()
                return resp.json().get("elements", [])
            except (requests.exceptions.RequestException, ConnectionError) as e:
                # Catching ConnectionError handles the 10054 remote host disconnections
                last_error = str(e)
                continue
        time.sleep(2 * (attempt + 1))  # brief backoff before trying all mirrors again
    raise RuntimeError(f"All Overpass mirrors failed. Last error: {last_error}")


def _element_latlon(el):
    """Overpass returns lat/lon directly for nodes, and a 'center' dict for ways."""
    if "lat" in el and "lon" in el:
        return el["lat"], el["lon"]
    center = el.get("center")
    if center:
        return center["lat"], center["lon"]
    return None


@st.cache_data(show_spinner=False)
def fetch_houses(lat: float, lon: float, radius_km: float):
    # A plain ["building"] presence check is far cheaper for Overpass to
    # evaluate over a dense area than a broad regex - the regex version was
    # the main cause of the 504 timeouts on large Delhi localities.
    query = f"""
    [out:json][timeout:90];
    (
      way["building"](around:{radius_km * 1000},{lat},{lon});
      node["building"](around:{radius_km * 1000},{lat},{lon});
    );
    out center qt;
    """
    elements = _run_overpass(query)
    houses = []
    for el in elements:
        latlon = _element_latlon(el)
        if latlon:
            houses.append({"id": el["id"], "lat": latlon[0], "lon": latlon[1], "status": GREY})

    return houses


@st.cache_data(show_spinner=False)
def fetch_petrol_pumps(lat: float, lon: float, radius_km: float):
    query = f"""
    [out:json][timeout:90];
    (
      node["amenity"="fuel"](around:{radius_km * 1000},{lat},{lon});
      way["amenity"="fuel"](around:{radius_km * 1000},{lat},{lon});
    );
    out center qt;
    """
    elements = _run_overpass(query)
    pumps = []
    for el in elements:
        latlon = _element_latlon(el)
        if latlon:
            pumps.append({"id": el["id"], "lat": latlon[0], "lon": latlon[1]})
    return pumps


@st.cache_data(show_spinner=False)
def fetch_nearest_substation(lat: float, lon: float, radius_km: float):
    """Nearest power=substation to (lat, lon). Returns {lat, lon, dist} or None."""
    query = f"""
    [out:json][timeout:90];
    (
      node["power"="substation"](around:{radius_km * 1000},{lat},{lon});
      way["power"="substation"](around:{radius_km * 1000},{lat},{lon});
    );
    out center qt;
    """
    elements = _run_overpass(query)
    candidates = []
    for el in elements:
        latlon = _element_latlon(el)
        if latlon:
            dist = geodesic((lat, lon), latlon).km
            candidates.append({"lat": latlon[0], "lon": latlon[1], "dist": dist})
    if not candidates:
        return None
    return min(candidates, key=lambda c: c["dist"])


@st.cache_data(show_spinner=False)
def fetch_nearest_open_area(lat: float, lon: float, radius_km: float):
    """Nearest leisure=park to (lat, lon). Returns {lat, lon, dist} or None."""
    query = f"""
    [out:json][timeout:90];
    (
      node["leisure"="park"](around:{radius_km * 1000},{lat},{lon});
      way["leisure"="park"](around:{radius_km * 1000},{lat},{lon});
    );
    out center qt;
    """
    elements = _run_overpass(query)
    candidates = []
    for el in elements:
        latlon = _element_latlon(el)
        if latlon:
            dist = geodesic((lat, lon), latlon).km
            candidates.append({"lat": latlon[0], "lon": latlon[1], "dist": dist})
    if not candidates:
        return None
    return min(candidates, key=lambda c: c["dist"])


# --------------------------------------------------------------------------
# SIMULATION LOGIC
# --------------------------------------------------------------------------

def initial_classification(houses: list):
    """Start every house GREY (non-EV), then randomly promote 5% -> GREEN (already EV-adopted)."""
    house_ids = [h["id"] for h in houses]
    random.shuffle(house_ids)

    n_green = int(len(house_ids) * INITIAL_GREEN_PCT)
    green_ids = set(house_ids[:n_green])

    for h in houses:
        h["status"] = GREEN if h["id"] in green_ids else GREY
    return houses


def fiscal_year_label(cycle_number: int) -> str:
    """Cycle 1 is always FY2023-24, cycle 2 is FY2024-25, etc."""
    start_year = 2023 + (cycle_number - 1)
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def get_petrol_cost_per_km_for_cycle(cycle_number: int) -> float:
    """
    Rs/km petrol running cost used for this cycle's cost comparison -
    a direct lookup into PETROL_COST_PER_KM_BY_YEAR for this cycle's fiscal
    year (cycle 1 = FY2023-24, cycle 2 = FY2024-25, etc). No multiplier of
    any kind is applied - it's exactly the value read off the chart for
    that year. If a cycle's year runs past the last year in the table
    (FY2034-35), the last known value is reused.
    """
    fy_label = fiscal_year_label(cycle_number)
    if fy_label in PETROL_COST_PER_KM_BY_YEAR:
        return PETROL_COST_PER_KM_BY_YEAR[fy_label]
    last_year = list(PETROL_COST_PER_KM_BY_YEAR.keys())[-1]
    return PETROL_COST_PER_KM_BY_YEAR[last_year]


def nearest_distance(point, locations: list):
    """Return the shortest geodesic distance (km) from point to any location."""
    if not locations:
        return float("inf")
    return min(geodesic(point, (loc["lat"], loc["lon"])).km for loc in locations)


def run_cycle(houses: list, charging_station: tuple, petrol_pumps: list, radius_km: float,
              petrol_cost_per_km: float, ev_charge_per_km: float = EV_CHARGE_PER_KM,
              distance_only: bool = False):
    """
    For every GREY house within radius_km of the charging station, decide
    whether it converts to GREEN.

    Normal cycles (distance_only=False) compare the cost of the petrol route
    against the cost of the EV route, both already expressed in Rs/km so
    they're directly comparable - no distance multiplier of any kind is
    applied on either side:

        petrol_side = distance_to_nearest_petrol_pump_km * petrol_cost_per_km
        ev_side     = distance_to_charging_station_km * ev_charge_per_km  (Rs 1.2/km)

    If ev_side is less than petrol_side, the charging station works out
    cheaper, so the house turns GREEN.

    petrol_cost_per_km is this cycle's Rs/km reading from
    PETROL_COST_PER_KM_BY_YEAR (see get_petrol_cost_per_km_for_cycle), not a
    flat constant - so the comparison gets tougher for EV as petrol Rs/km
    rises year over year.

    The FIRST cycle (distance_only=True) skips the cost model entirely and
    just compares raw distances: if the house is closer to the charging
    station than to the nearest petrol pump, it turns GREEN. No Rs/km
    weighting is applied on either side for this cycle.

    Returns the updated house list, the count of newly converted houses, and
    the set of house IDs that were converted THIS cycle (so the map can
    highlight them differently from houses that were already green).
    """
    converted = 0
    converted_ids = set()

    for h in houses:
        if h["status"] != GREY:
            continue
        house_point = (h["lat"], h["lon"])
        dist_to_station = geodesic(house_point, charging_station).km
        if dist_to_station > radius_km:
            continue
        dist_to_pump = nearest_distance(house_point, petrol_pumps)

        if distance_only:
            # First cycle: pure distance comparison, no cost model.
            house_converts = dist_to_station < dist_to_pump
        else:
            petrol_side = dist_to_pump * petrol_cost_per_km
            ev_side = dist_to_station * ev_charge_per_km
            house_converts = ev_side < petrol_side

        if house_converts:
            h["status"] = GREEN
            converted += 1
            converted_ids.add(h["id"])
    return houses, converted, converted_ids


def compute_cost_over_time(num_cycles, cable_distance_km: float = 0.0):
    """
    Build the year-by-year cumulative cost of setting up and running the
    charging station:

        cable_cost   = cable_distance_km * 1000 * CABLE_COST_PER_METER
        setup_cost   = cable_cost + BASE_SETUP_COST + STATION_HARDWARE_COST
        cost(year n) = setup_cost + n * ANNUAL_RECURRING_COST

    The station is placed at the nearest open area to the nearest
    substation, not at the substation itself, so cabling now has to run
    that straight-line distance (cable_distance_km) to reach power -
    cable_cost is 0 only when the station ended up co-located with the
    substation (no open area found nearby, or no substation found at all).

    ANNUAL_RECURRING_COST is the "expansion cost" - what it costs each year
    to extend/support the growing coverage range (the cycle radius grows
    every year too). The resulting running total for each year is the
    "cumulative cost" / maintenance increment shown in the cost table.

    Returns (setup_cost, cable_cost, {year: cumulative_cost}).
    """
    cable_cost = cable_distance_km * 1000 * CABLE_COST_PER_METER
    setup_cost = cable_cost + BASE_SETUP_COST + STATION_HARDWARE_COST

    yearly_costs = {
        year: setup_cost + year * ANNUAL_RECURRING_COST
        for year in range(1, num_cycles + 1)
    }
    return setup_cost, cable_cost, yearly_costs



def status_counts(houses: list, charging_station=None, radius_km=None):
    """
    Count grey/green houses. If charging_station + radius_km are given, only
    counts houses within that radius (i.e. the current cycle's sample space)
    instead of every house ever fetched.
    """
    if charging_station is not None and radius_km is not None:
        houses = [
            h for h in houses
            if geodesic((h["lat"], h["lon"]), charging_station).km <= radius_km
        ]
    grey = sum(1 for h in houses if h["status"] == GREY)
    green = sum(1 for h in houses if h["status"] == GREEN)
    return {"Grey (non-EV)": grey, "Green (EV adopted)": green}


def group_houses_for_map(houses: list, group_size: int = GROUP_SIZE):
    """
    Roll individual houses up into groups of `group_size` for map display
    only - the simulation itself still tracks and converts every house
    individually; this just keeps the map from trying to render thousands
    of markers now that the house cap is gone.

    Each group is shown as a single dot at the group's centroid:
        - grey if fewer than half the houses in the group are GREEN
        - green if half or more are GREEN
        - outlined red if any house in the group converted THIS cycle
    """
    groups = []
    for i in range(0, len(houses), group_size):
        chunk = houses[i:i + group_size]
        lat = sum(h["lat"] for h in chunk) / len(chunk)
        lon = sum(h["lon"] for h in chunk) / len(chunk)
        green_count = sum(1 for h in chunk if h["status"] == GREEN)
        grey_count = len(chunk) - green_count
        groups.append({
            "lat": lat,
            "lon": lon,
            "size": len(chunk),
            "green_count": green_count,
            "grey_count": grey_count,
            # Group only flips GREEN once 75%+ of its houses have converted -
            # a couple of conversions in a group of 50 shouldn't flip the dot.
            "status": GREEN if green_count >= GROUP_GREEN_THRESHOLD * len(chunk) else GREY,
            "ids": [h["id"] for h in chunk],
        })
    return groups


# --------------------------------------------------------------------------
# MAP RENDERING
# --------------------------------------------------------------------------

def build_map(center, houses, petrol_pumps, charging_station, substation, radius_km, newly_converted_ids=None):
    newly_converted_ids = newly_converted_ids or set()
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    # Only show houses that have actually entered the sample space for THIS
    # cycle (i.e. within the current radius of the charging station). As the
    # radius grows each cycle, new grey/green groups appear on the map -
    # matching which houses run_cycle is actually evaluating.
    color_map = {GREY: "gray", GREEN: "green"}
    visible_houses = [
        h for h in houses
        if geodesic((h["lat"], h["lon"]), charging_station).km <= radius_km
    ]
    groups = group_houses_for_map(visible_houses)

    for g in groups:
        has_new_conversion = any(hid in newly_converted_ids for hid in g["ids"])
        color = color_map[g["status"]]
        # Only ring a dot red if it actually IS green now AND something in
        # it converted this cycle - a grey dot never gets the red ring,
        # even if a few of its houses flipped this cycle but not enough to
        # cross GROUP_GREEN_THRESHOLD.
        is_new = has_new_conversion and g["status"] == GREEN
        label = (
            f"Group of {g['size']} houses - {g['green_count']} green / {g['grey_count']} grey"
            + (" (converted this cycle)" if is_new else "")
        )

        folium.CircleMarker(
            location=(g["lat"], g["lon"]),
            radius=7 if is_new else 5,
            color="red" if is_new else color,
            weight=3 if is_new else 1,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=label,
        ).add_to(fmap)

    # Petrol pumps (highlighted)
    for p in petrol_pumps:
        folium.Marker(
            location=(p["lat"], p["lon"]),
            icon=folium.Icon(color="orange", icon="tint", prefix="fa"),
            popup="Petrol Pump",
        ).add_to(fmap)

    # Charging station (center of map)
    folium.Marker(
        location=charging_station,
        icon=folium.Icon(color="blue", icon="bolt", prefix="fa"),
        popup="Charging Station",
    ).add_to(fmap)

    # Nearest substation (the station itself sits at the nearest open area
    # to this substation, not on top of it - see the cable line below)
    if substation:
        folium.Marker(
            location=(substation["lat"], substation["lon"]),
            icon=folium.Icon(color="purple", icon="industry", prefix="fa"),
            popup="Nearest Substation",
        ).add_to(fmap)
        folium.PolyLine(
            locations=[(substation["lat"], substation["lon"]), charging_station],
            color="purple",
            weight=2,
            dash_array="4,6",
            popup="Cable run: substation -> station",
        ).add_to(fmap)

    # Current cycle radius, drawn around the charging station
    folium.Circle(
        location=charging_station,
        radius=radius_km * 1000,
        color="blue",
        fill=False,
        dash_array="5,5",
    ).add_to(fmap)

    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                border: 1px solid #999; font-size: 13px; line-height: 1.6;">
      <b>Legend</b><br>
      Each dot = a group of {GROUP_SIZE} houses<br>
      <span style="color:gray;">&#9679;</span> Grey - majority non-EV<br>
      <span style="color:green;">&#9679;</span> Green - majority EV-adopted<br>
      <span style="border:2px solid red; border-radius:50%; display:inline-block; width:10px; height:10px;"></span>
      &nbsp;Red outline - some houses converted this cycle<br>
      <span style="color:orange;">&#9679;</span> Petrol pump<br>
      <span style="color:blue;">&#9679;</span> Charging station
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap


# --------------------------------------------------------------------------
# STREAMLIT DASHBOARD
# --------------------------------------------------------------------------

def run_simulation(chosen_locality: str, num_cycles: int):
    """Runs the full pipeline and returns a results dict, or raises on failure."""
    center = geocode_locality(chosen_locality)

    # Only fetch as much radius as the chosen number of cycles actually needs
    # (last cycle's radius), capped at MAX_FETCH_RADIUS_KM. Requesting a huge
    # fixed radius regardless of cycle count was the main cause of Overpass
    # 504 timeouts on dense Delhi localities.
    final_cycle_radius = BASE_CYCLE_RADIUS_KM + (num_cycles - 1) * CYCLE_RADIUS_INCREMENT_KM
    fetch_radius_km = min(final_cycle_radius, MAX_FETCH_RADIUS_KM)

    # Houses and petrol pumps are independent of the station-placement chain,
    # so fetch those two sequentially to avoid server rate-limiting.
    houses = fetch_houses(center[0], center[1], fetch_radius_km)
    petrol_pumps = fetch_petrol_pumps(center[0], center[1], fetch_radius_km)

    # Station placement, in order:
    #   1. Find the center (already have it).
    #   2. Find the nearest substation to that center.
    #   3. From the substation's location (not the center), find the
    #      nearest open area to it - that's where the station is built.
    # Falls back a step at a time if something isn't found within range:
    # substation but no open area near it -> station sits at the substation;
    # no substation at all -> station sits at the geocoded center.
    substation = fetch_nearest_substation(center[0], center[1], fetch_radius_km)

    if substation:
        substation_point = (substation["lat"], substation["lon"])
        open_area = fetch_nearest_open_area(
            substation_point[0], substation_point[1], fetch_radius_km
        )
        if open_area:
            charging_station = (open_area["lat"], open_area["lon"])
            cable_distance_km = open_area["dist"]  # substation -> station
        else:
            charging_station = substation_point
            cable_distance_km = 0.0
        station_offset_km = geodesic(center, charging_station).km  # center -> station
    else:
        charging_station = center
        station_offset_km = 0.0
        cable_distance_km = 0.0

    if not houses:
        return None

    houses = initial_classification(houses)

    setup_cost, cable_cost, cost_over_time = compute_cost_over_time(num_cycles, cable_distance_km)

    # Pre-compute every cycle's house state and map ONCE here, so that later
    # map interactions (zoom/pan/click) - which cause Streamlit to rerun the
    # whole script - just redraw from these stored results instead of
    # re-fetching or re-simulating from scratch.
    cycles = []
    history = []
    radius_km = BASE_CYCLE_RADIUS_KM

    for cycle_index in range(num_cycles):
        cycle_number = cycle_index + 1  # simulation always starts at FY2023-24
        fy_label = fiscal_year_label(cycle_number)
        petrol_cost_per_km_this_cycle = get_petrol_cost_per_km_for_cycle(cycle_number)

        houses, converted, converted_ids = run_cycle(
            houses, charging_station, petrol_pumps, radius_km,
            petrol_cost_per_km=petrol_cost_per_km_this_cycle,
            distance_only=(cycle_number == 1),
        )
        counts = status_counts(houses, charging_station, radius_km)
        counts["Cycle"] = cycle_number
        counts["Fiscal Year"] = fy_label
        counts["Radius (km)"] = radius_km
        counts["Petrol cost this cycle (Rs/km)"] = round(petrol_cost_per_km_this_cycle, 2)
        counts["Newly converted"] = converted
        history.append(counts)

        fmap = build_map(
            center, houses, petrol_pumps, charging_station, substation, radius_km,
            newly_converted_ids=converted_ids,
        )
        cycles.append({
            "radius_km": radius_km,
            "converted": converted,
            "converted_ids": converted_ids,
            "counts": counts,
            "map": fmap,
            # snapshot the house statuses for this cycle (houses list is mutated in place)
            "houses_snapshot": [dict(h) for h in houses],
        })
        radius_km += CYCLE_RADIUS_INCREMENT_KM

    return {
        "locality": chosen_locality,
        "center": center,
        "charging_station": charging_station,
        "station_offset_km": station_offset_km,
        "cable_distance_km": cable_distance_km,
        "num_houses": len(houses),
        "num_pumps": len(petrol_pumps),
        "cycles": cycles,
        "history": history,
        "cable_cost": cable_cost,
        "setup_cost": setup_cost,
        "cost_over_time": cost_over_time,
    }


def build_petrol_cost_per_km_figure(num_cycles: int):
    """
    Petrol Rs/km straight from PETROL_COST_PER_KM_BY_YEAR - the EXACT values
    run_cycle() looks up as petrol_cost_per_km and compares against
    EV_CHARGE_PER_KM. Actual FY2019-20 to FY2024-25 are solid, FY2025-26
    onward (7%/yr projection) are dashed. The years this simulation run
    actually uses (cycle 1 = FY2023-24 onward) are highlighted.
    """
    labels = list(PETROL_COST_PER_KM_BY_YEAR.keys())
    values = list(PETROL_COST_PER_KM_BY_YEAR.values())
    x = list(range(len(labels)))
    actual_cutoff = labels.index("2024-25")  # last actual year, rest is projected

    used_labels = {fiscal_year_label(c) for c in range(1, max(num_cycles, 1) + 1)}
    used_x = [i for i, lbl in enumerate(labels) if lbl in used_labels]
    used_y = [values[i] for i in used_x]

    BLUE = "#1f77b4"
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x[:actual_cutoff + 1], values[:actual_cutoff + 1], marker="o", color=BLUE,
            linewidth=2.2, label="Actual (PPAC)")
    ax.plot(x[actual_cutoff:], values[actual_cutoff:], marker="o", linestyle="--",
            color="#ff7f0e", linewidth=2.2, label="Projected (+7%/yr)")

    ax.scatter(used_x, used_y, color="red", zorder=5, s=70,
               label=f"Years used in this run (from FY{fiscal_year_label(1)})")

    for xi, yi in zip(x, values):
        ax.annotate(f"₹{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7.5, color="#444")

    ax.axhline(EV_CHARGE_PER_KM, color="green", linewidth=1.5, linestyle="-.",
               label=f"EV flat rate: Rs {EV_CHARGE_PER_KM}/km")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel("Petrol Cost per km (Rs)")
    ax.set_xlabel("Financial Year")
    ax.set_title("Petrol Cost/km vs EV Cost/km - the values run_cycle() actually compares")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


def render_results(results: dict):
    st.subheader(f"Simulation for {results['locality']}")
    st.write(
        f"Found **{results['num_houses']}** houses and **{results['num_pumps']}** petrol pumps in range. "
        f"Charging station placed at the nearest open area to the nearest substation: "
        f"`{results['charging_station'][0]:.5f}, {results['charging_station'][1]:.5f}` "
        f"({results['station_offset_km']:.3f} km from the geocoded center, "
        f"{results['cable_distance_km']:.3f} km of cable run from the substation)."
    )

    cycles = results["cycles"]
    cycle_tabs = st.tabs([f"Cycle {i+1} (FY {fiscal_year_label(i+1)})" for i in range(len(cycles))])

    for cycle_index, (tab, cycle) in enumerate(zip(cycle_tabs, cycles)):
        with tab:
            st.markdown(
                f"**Radius this cycle:** {cycle['radius_km']} km &nbsp;|&nbsp; "
                f"**Newly converted to green:** {cycle['converted']}"
            )
            # returned_objects=[] keeps st_folium from feeding zoom/pan/click
            # state back into the script every interaction beyond what's
            # needed to just render - the map itself is already pre-built.
            st_folium(
                cycle["map"], width=None, height=550,
                key=f"map_cycle_{cycle_index}", returned_objects=[],
            )
            counts = cycle["counts"]
            st.write(
                {
                    "Grey (non-EV)": counts["Grey (non-EV)"],
                    "Green (EV adopted)": counts["Green (EV adopted)"],
                }
            )

    st.subheader("Adoption over time")
    yearly_table = [
        {
            "Year": row["Fiscal Year"],
            "Grey (non-EV)": row["Grey (non-EV)"],
            "Green (EV adopted)": row["Green (EV adopted)"],
        }
        for row in results["history"]
    ]
    st.dataframe(yearly_table, width='stretch')

    st.markdown("**Absolute counts** (houses within each cycle's radius)")
    st.line_chart(
        {row["Fiscal Year"]: {
            "Grey": row["Grey (non-EV)"],
            "Green": row["Green (EV adopted)"],
        } for row in results["history"]},
    )

    st.markdown("**Share of houses** (grey falling, green rising, as a fraction of all houses within radius)")
    fraction_data = {}
    for row in results["history"]:
        grey = row["Grey (non-EV)"]
        green = row["Green (EV adopted)"]
        total = grey + green
        fraction_data[row["Fiscal Year"]] = {
            "Grey share": grey / total if total else 0.0,
            "Green share": green / total if total else 0.0,
        }
    st.line_chart(fraction_data)

    st.subheader("Charging station cost over time")
    st.write(
        f"Charging station placed at the nearest open area to the nearest substation, "
        f"**{results['station_offset_km']:.3f} km** from the locality's geocoded center."
    )
    st.write(
        f"- Cabling cost: ₹{results['cable_cost']:,.0f} "
        f"({results['cable_distance_km']:.3f} km from substation to station "
        f"@ ₹{CABLE_COST_PER_METER:,}/m)\n"
        f"- Plus base setup cost: ₹{BASE_SETUP_COST:,.0f}\n"
        f"- Plus station hardware cost: ₹{STATION_HARDWARE_COST:,.0f}\n"
        f"- **One-time setup total: ₹{results['setup_cost']:,.0f}**\n"
        f"- Plus ₹{ANNUAL_RECURRING_COST:,.0f} expansion cost (for increasing range) added every year"
    )
    st.line_chart(
        {year: {"Cumulative cost (Rs)": cost} for year, cost in results["cost_over_time"].items()}
    )

    st.markdown("**Cost breakdown by year**")
    cost_table = [
        {
            "Year": year,
            "Expansion cost this year (Rs)": ANNUAL_RECURRING_COST,
            "Cumulative cost / maintenance increment (Rs)": cost,
        }
        for year, cost in results["cost_over_time"].items()
    ]
    st.dataframe(cost_table, width='stretch')

    st.subheader("Petrol cost/km vs EV cost/km used in this simulation")
    st.caption(
        "This is the actual comparison run_cycle() makes each cycle - that "
        "year's petrol Rs/km reading (no multiplier) against the flat EV rate."
    )
    st.pyplot(build_petrol_cost_per_km_figure(len(results["cycles"])))


def main():
    st.set_page_config(page_title="Delhi EV Adoption Simulator", layout="wide")
    st.title("⚡ Delhi EV Charging Adoption Simulator")
    st.caption(
        "Pick a locality, choose how many years (cycles) to simulate, and watch "
        "EV adoption spread outward from a centrally-placed charging station."
    )

    with st.sidebar:
        st.header("Controls")
        locality = st.selectbox("Choose a Delhi locality", DELHI_LOCALITIES)
        custom_locality = st.text_input("...or type a custom locality name (optional)")
        num_cycles = st.slider("Number of cycles (years)", min_value=1, max_value=6, value=2)
        run_button = st.button("Run Simulation", type="primary")

    chosen_locality = custom_locality.strip() if custom_locality.strip() else locality

    # Only recompute the simulation when the button is actually clicked.
    # Everything else (map zoom/pan, tab switches) just triggers a rerun of
    # this script, which re-renders from st.session_state instead of
    # re-fetching/re-simulating - that's what stops results from vanishing.
    if run_button:
        with st.spinner(f"Geocoding {chosen_locality}..."):
            try:
                results = run_simulation(chosen_locality, num_cycles)
            except Exception as e:
                st.session_state["last_error"] = (
                    f"Failed to run simulation: {e}\n\n"
                    "The public Overpass servers can be slow or rate-limited. "
                    "Try again in a minute, pick a smaller number of cycles, or "
                    "choose a locality with a less densely built-up area."
                )
                st.session_state["results"] = None
                results = None

            if results is None and "last_error" not in st.session_state:
                st.session_state["last_error"] = "No houses found in this area from OpenStreetMap data. Try another locality."

            if results is not None:
                st.session_state["last_error"] = None
                st.session_state["results"] = results

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    if st.session_state.get("results"):
        render_results(st.session_state["results"])
    elif not st.session_state.get("last_error"):
        st.info("Configure your options in the sidebar and click **Run Simulation**.")


if __name__ == "__main__":
    main()