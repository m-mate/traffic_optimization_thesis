"""
Regenerate SUMO network, routes, and config from an OSM file.

Usage:
    py src/utils/osm_to_sumo.py data/NewMap.osm

This will produce:
    data/debrecen.net.xml
    data/debrecen.rou.xml
    data/debrecen.sumocfg
"""

import subprocess
import sys
import os

SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)

NET_FILE = os.path.join(DATA_DIR, "debrecen.net.xml")
ROU_FILE = os.path.join(DATA_DIR, "debrecen.rou.xml")
TRIPS_FILE = os.path.join(DATA_DIR, "debrecen.trips.xml")
CFG_FILE = os.path.join(DATA_DIR, "debrecen.sumocfg")
RANDOM_TRIPS = os.path.join(SUMO_HOME, "tools", "randomTrips.py")


def run(cmd, description):
    """Run a command and exit on failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  > {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    osm_file = os.path.abspath(sys.argv[1])
    if not os.path.exists(osm_file):
        print(f"[ERROR] OSM file not found: {osm_file}")
        sys.exit(1)

    # --- Step 1: netconvert ---
    run([
        os.path.join(SUMO_HOME, "bin", "netconvert"),
        "--osm-files", osm_file,
        "-o", NET_FILE,
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess", "true",
        "--tls.join",
        "--edges.join",
        "--no-turnarounds", "true",
        "--remove-edges.by-type",
        "highway.footway,highway.path,highway.steps,highway.pedestrian,"
        "highway.cycleway,highway.bridleway,railway.rail,railway.tram,railway.light_rail",
        "--remove-edges.isolated",
        "--keep-edges.by-vclass", "passenger",
    ], "Converting OSM to SUMO network")

    # --- Step 2: randomTrips → routes ---
    run([
        sys.executable, RANDOM_TRIPS,
        "-n", NET_FILE,
        "-r", ROU_FILE,
        "-o", TRIPS_FILE,
        "-e", "3600",
        "-p", "5",
        "--seed", "42",
        "--validate",
        "--fringe-factor", "10",
        "--allow-fringe.min-length", "100",
        "--min-distance", "200",
    ], "Generating random trips and routes")

    # --- Step 3: Write .sumocfg ---
    sumocfg_content = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="debrecen.net.xml"/>
        <route-files value="debrecen.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <no-warnings value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
    with open(CFG_FILE, "w") as f:
        f.write(sumocfg_content)
    print(f"\n  Wrote {CFG_FILE}")

    # --- Step 4: Print summary ---
    print(f"\n{'='*60}")
    print("  DONE — Summary")
    print(f"{'='*60}")
    print(f"  Network:  {NET_FILE}")
    print(f"  Routes:   {ROU_FILE}")
    print(f"  Config:   {CFG_FILE}")
    print(f"\n  Launch GUI with:")
    print(f'  sumo-gui -c "{CFG_FILE}"')

    # Quick stats via sumolib
    try:
        import sumolib
        net = sumolib.net.readNet(NET_FILE, withPrograms=True)
        tls = net.getTrafficLights()
        print(f"\n  Junctions:      {len(net.getNodes())}")
        print(f"  Edges:          {len(net.getEdges())}")
        print(f"  Traffic lights: {len(tls)}")
        for t in tls:
            print(f"    - {t.getID()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
