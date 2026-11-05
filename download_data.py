"""从公开镜像下载 CIFAR-10，校验 MD5 并自动解压。

下载源是 hf-mirror，适合官方源较慢的情况；强制使用 IPv4，避免部分
网络环境下 IPv6 握手超时。
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import math
import tarfile
from pathlib import Path

import requests
import urllib3.util.connection as _c

_c.HAS_IPV6 = False

URL = "https://hf-mirror.com/datasets/VerisimilitudeX/cifar10/resolve/main/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "cifar-10-python.tar.gz"
PARTS = ROOT / "data" / "parts"
CHUNK = 4 * 1024 * 1024


def get_part(item: tuple[int, int]) -> int:
    """下载一个分片，支持断点续传。"""
    i, size = item
    start = i * CHUNK
    end = min(size - 1, start + CHUNK - 1)
    target = PARTS / f"{i:04d}.part"
    expected = end - start + 1
    if target.exists() and target.stat().st_size == expected:
        return i
    for attempt in range(10):
        try:
            r = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(30, 300))
            r.raise_for_status()
            if r.status_code != 206 or len(r.content) != expected:
                raise RuntimeError(f"分片 {i}：状态码 {r.status_code}，长度 {len(r.content)}")
            target.write_bytes(r.content)
            return i
        except Exception as e:
            print(f"分片 {i} 第 {attempt + 1} 次重试：{e}", flush=True)
    raise RuntimeError(f"分片 {i} 下载失败")


def ensure_extracted() -> None:
    """如果还没解压，就把 tar.gz 解压到 data/。"""
    target = ROOT / "data" / "cifar-10-batches-py"
    if not target.exists():
        with tarfile.open(OUT, "r:gz") as tar:
            tar.extractall(OUT.parent)
        print("已解压到 data/cifar-10-batches-py", flush=True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and hashlib.md5(OUT.read_bytes()).hexdigest() == MD5:
        print(f"文件已存在且 MD5 正确：{OUT}", flush=True)
        ensure_extracted()
        return

    PARTS.mkdir(exist_ok=True)
    head = requests.get(URL, headers={"Range": "bytes=0-0"}, timeout=60)
    head.raise_for_status()
    size = int(head.headers["Content-Range"].split("/")[-1])
    jobs = [(i, size) for i in range(math.ceil(size / CHUNK))]
    print(f"文件大小 {size} 字节，共 {len(jobs)} 个分片", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for n, i in enumerate(pool.map(get_part, jobs), 1):
            print(f"已完成 {n}/{len(jobs)}", flush=True)

    md5 = hashlib.md5()
    with OUT.open("wb") as f:
        for i, _ in jobs:
            data = (PARTS / f"{i:04d}.part").read_bytes()
            f.write(data)
            md5.update(data)
    if md5.hexdigest() != MD5:
        OUT.unlink()
        raise RuntimeError(f"MD5 校验失败：{md5.hexdigest()}")
    print(f"下载完成，MD5 校验通过：{OUT}", flush=True)
    ensure_extracted()


if __name__ == "__main__":
    main()
