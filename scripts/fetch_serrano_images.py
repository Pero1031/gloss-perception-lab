"""Fetch a subset of Serrano et al. 2021 images from the remote zip with HTTP range requests.

Only the zip directory and the selected entries are downloaded; the zip itself is never stored.
Each image is resized and saved under the same file name as in manifest.csv. Existing files are skipped.
"""
import argparse
import io
import struct
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

URL = "https://mig.mpi-inf.mpg.de/dataset/dataset_jpg.zip"
LOCAL_HEADER = struct.Struct("<4s5H3L2H")  # 30 bytes
MARGIN = 1024  # room for the local header name and extra fields


class RangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP, used only to parse the zip directory."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        data = fetch_range(self.url, self.pos, self.pos + n - 1)
        self.pos += len(data)
        return data


def fetch_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def read_entry(url, info):
    end = info.header_offset + LOCAL_HEADER.size + len(info.filename.encode()) + info.compress_size + MARGIN
    buf = fetch_range(url, info.header_offset, end)
    name_len, extra_len = LOCAL_HEADER.unpack_from(buf)[9:11]
    start = LOCAL_HEADER.size + name_len + extra_len
    raw = buf[start:start + info.compress_size]
    if len(raw) != info.compress_size:
        raise IOError("short read")
    data = zlib.decompressobj(-15).decompress(raw) if info.compress_type == 8 else raw
    if zlib.crc32(data) != info.CRC:
        raise IOError("crc mismatch")
    return data


def process(url, info, out_dir, size):
    name = info.filename.split("/", 1)[1]
    dest = out_dir / name
    if dest.exists():
        return "skipped"
    for attempt in range(4):
        try:
            data = read_entry(url, info)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((size, size), Image.LANCZOS)
    tmp = dest.with_name(dest.name + ".tmp")
    img.save(tmp, format="JPEG", quality=95)
    tmp.replace(dest)
    return "saved"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=URL)
    p.add_argument("--out", default="data/serrano/images")
    p.add_argument("--geometry", default="sphere", help="exact geometry name, e.g. sphere, blob, bunny_vt")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--limit", type=int, help="only fetch the first N matching images (for testing)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    infos = [i for i in zipfile.ZipFile(RangeFile(args.url)).infolist()
             if i.filename.endswith(".jpg") and f"][{args.geometry}][" in i.filename]
    if args.limit:
        infos = infos[:args.limit]
    gb = sum(i.compress_size for i in infos) / 1e9
    print(f"{len(infos)} images matching geometry={args.geometry!r} (~{gb:.1f} GB to download)", flush=True)

    counts = {"saved": 0, "skipped": 0}
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for n, result in enumerate(ex.map(lambda i: process(args.url, i, out_dir, args.size), infos), 1):
            counts[result] += 1
            if n % 100 == 0 or n == len(infos):
                print(f"{n}/{len(infos)}  {counts}  {time.time() - t0:.0f}s", flush=True)
    print("done", counts)


if __name__ == "__main__":
    main()
