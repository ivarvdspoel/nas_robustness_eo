from pathlib import Path
import math
import urllib.request
import pandas as pd

# PILOT = Path("cache/worldcover/pilot_200/trusted_patches_worldcover_pilot_200.csv")
# OUT_DIR = Path("cache/worldcover/pilot_200/tiles")
FULL = Path("cache/worldcover/all_trusted/trusted_patches_worldcover_all.csv")
OUT_DIR = Path("cache/worldcover/all_trusted/tiles")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not FULL.exists():
    raise FileNotFoundError(f"FULL CSV not found: {FULL}")

df = pd.read_csv(FULL, usecols=["center_lat", "center_lon"])

def tile_name(lat, lon):
    lat0 = math.floor(float(lat) / 3) * 3
    lon0 = math.floor(float(lon) / 3) * 3

    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"

    return f"ESA_WorldCover_10m_2021_v200_{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}_Map.tif"

tiles = sorted({tile_name(lat, lon) for lat, lon in zip(df["center_lat"], df["center_lon"])})

tile_list = OUT_DIR / "needed_tiles.txt"
tile_list.write_text("\n".join(tiles) + "\n")

print(f"N FULL patches: {len(df)}")
print(f"N WorldCover tiles needed: {len(tiles)}")
print(f"Tile list: {tile_list}")
print("First tiles:")
for t in tiles[:25]:
    print(" ", t)

for i, t in enumerate(tiles, 1):
    dst = OUT_DIR / t

    if dst.exists() and dst.stat().st_size > 0:
        print(f"[{i}/{len(tiles)}] exists {t}")
        continue

    url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{t}"
    print(f"[{i}/{len(tiles)}] downloading {t}")

    try:
        urllib.request.urlretrieve(url, dst)
        if not dst.exists() or dst.stat().st_size == 0:
            raise RuntimeError("empty download")
    except Exception as e:
        print(f"[WARN] failed: {url}")
        print("       ", repr(e))
        if dst.exists():
            dst.unlink()

print("Done.")
