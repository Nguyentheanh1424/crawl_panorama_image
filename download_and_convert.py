import asyncio
import aiohttp
import aiofiles
import json
import os
import shutil
from PIL import Image
import logging
import numpy as np
import py360convert

# Config
DATA_FILE = "data.json"
OUT_DIR = "panoramas"
TEMP_DIR = "temp_tiles"
FACES = ["f", "b", "l", "r", "u", "d"]

TILE_SIZE = 512

# Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pano")

def ensure_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"

def build_version_suffix(version: str) -> str:
    if not version:
        return ""
    v = str(version)
    return v if v.startswith("?") else f"?v={v}"

async def load_data():
    async with aiofiles.open(DATA_FILE, "r") as f:
        raw = json.loads(await f.read())

    result = []
    locations = raw if isinstance(raw, list) else raw.get("locations", [])
    for idx, loc in enumerate(locations):
        base_url = ensure_slash(str(loc.get("baseUrl") or loc.get("baseurl") or ""))
        name = loc.get("name") or f"loc{idx}"
        panoramas = loc.get("panoramas") or []
        version = loc.get("version") or ""
        result.append({
            "name": name,
            "baseUrl": base_url,
            "panoramas": panoramas,
            "version": version
        })
    return result

async def url_exists(session, url: str) -> bool:
    try:
        async with session.head(url, timeout=5) as resp:
            return resp.status == 200
    except:
        return False

async def fetch_tile(session, url, out_path):
    if os.path.exists(out_path):
        return True
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.read()
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                async with aiofiles.open(out_path, "wb") as f:
                    await f.write(data)
                return True
            return False
    except:
        return False

async def detect_max_grid(session, base_url, pano_id, face, version_suffix, level=0, max_check=10):
    max_x, max_y = -1, -1
    # Scan ngang (x)
    for x in range(max_check):
        url = f"{base_url}{pano_id}/{face}/{level}/{x}_0.jpg{version_suffix}"
        if await url_exists(session, url):
            max_x = x
        else:
            break
    # Scan dọc (y)
    for y in range(max_check):
        url = f"{base_url}{pano_id}/{face}/{level}/0_{y}.jpg{version_suffix}"
        if await url_exists(session, url):
            max_y = y
        else:
            break
    return max_x, max_y

async def get_tile_with_fallback(session, base_url, pano_id, face, level, x, y, version_suffix, cache_dir):
    out_path = os.path.join(cache_dir, f"{level}_{x}_{y}.jpg")
    url = f"{base_url}{pano_id}/{face}/{level}/{x}_{y}.jpg{version_suffix}"

    if await fetch_tile(session, url, out_path):
        return Image.open(out_path)

    # Fallback xuống level thấp hơn
    for fallback_level in range(level-1, -1, -1):
        scale = 2**(level - fallback_level)
        fx, fy = x // scale, y // scale
        sub_tile = await get_tile_with_fallback(session, base_url, pano_id, face,
                                                fallback_level, fx, fy, version_suffix, cache_dir)
        if sub_tile:
            sub_size = TILE_SIZE // scale
            cx, cy = (x % scale) * sub_size, (y % scale) * sub_size
            crop_box = (cx, cy, cx + sub_size, cy + sub_size)
            cropped = sub_tile.crop(crop_box).resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
            return cropped
    return None

async def download_and_merge_face(session, base_url, pano_id, face, version_suffix, save_dir, level=0):
    face_dir = os.path.join(TEMP_DIR, pano_id, face)
    os.makedirs(save_dir, exist_ok=True)

    max_x, max_y = await detect_max_grid(session, base_url, pano_id, face, version_suffix, level)
    if max_x < 0 or max_y < 0:
        logger.warning(f"Face {face}: no tiles found")
        return None

    width, height = (max_x+1)*TILE_SIZE, (max_y+1)*TILE_SIZE
    pano = Image.new("RGB", (width, height))

    logger.info(f"Face {face}: grid {max_x+1}x{max_y+1} → {width}x{height}px")

    for x in range(max_x+1):
        for y in range(max_y+1):
            tile = await get_tile_with_fallback(session, base_url, pano_id, face,
                                                level, x, y, version_suffix, face_dir)
            if tile:
                pano.paste(tile, (y*TILE_SIZE, x*TILE_SIZE))  # đảo Y
            else:
                logger.warning(f"Tile missing even after fallback: {face} {x}_{y}")

    out_path = os.path.join(save_dir, f"{face}.jpg")
    pano.save(out_path)
    logger.info(f"✓ Merged face {face} → {out_path}")
    return out_path

async def process_pano(session, loc, pano_id):
    version_suffix = build_version_suffix(loc["version"])
    base_url = loc["baseUrl"]
    pano_dir = os.path.join(OUT_DIR, loc["name"], pano_id)

    logger.info(f"▶ Downloading panorama {pano_id} ({loc['name']})")
    face_size = None
    for face in FACES:
        face_path = await download_and_merge_face(session, base_url, pano_id, face, version_suffix, pano_dir, level=0)
        if face_path and face_size is None:
            face_size = Image.open(face_path).size[0]

    if face_size:
        equi_path = os.path.join(OUT_DIR, loc["name"], f"{pano_id}.jpg")
        merge_to_equirect(pano_dir, equi_path, face_size)

        # Cleanup: remove whole pano folder + temp tiles
        shutil.rmtree(pano_dir, ignore_errors=True)
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        logger.info("✓ Cleaned up pano folder and temp tiles")


def merge_to_equirect(pano_dir, out_path, face_size):
    """
    Convert 6 face images (f,b,l,r,u,d) into one equirectangular image.
    Uses py360convert.c2e(cubemap, h, w, mode=..., cube_format='dict').
    """
    # Load faces and ensure they exist
    face_images = {}
    for face in FACES:
        face_path = os.path.join(pano_dir, f"{face}.jpg")
        if not os.path.exists(face_path):
            raise FileNotFoundError(f"Missing face image: {face_path}")
        # ensure RGB and consistent dtype
        im = Image.open(face_path).convert("RGB")
        face_images[face] = np.array(im)

    # Build cube dict in py360convert's expected key order
    # Keys must be: 'F','R','B','L','U','D'
    cube_dict = {
        "F": face_images["f"],
        "R": face_images["r"],
        "B": face_images["b"],
        "L": face_images["l"],
        "U": face_images["u"],
        "D": face_images["d"],
    }

    # Output size
    out_h = face_size * 2
    out_w = face_size * 4

    # Call c2e with (cube_dict, h, w, ...) and specify cube_format='dict'
    equi = py360convert.c2e(cube_dict, out_h, out_w, mode="bilinear", cube_format="dict")

    # Save result
    img = Image.fromarray(equi.astype(np.uint8))
    img.save(out_path, quality=100)
    logger.info(f"✓ Merged panorama (equirect) → {out_path}")

async def main():
    data = await load_data()
    async with aiohttp.ClientSession() as session:
        for loc in data:
            for pano_id in loc["panoramas"]:
                await process_pano(session, loc, pano_id)

if __name__ == "__main__":
    asyncio.run(main())
