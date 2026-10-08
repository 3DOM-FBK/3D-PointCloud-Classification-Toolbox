#!/usr/bin/env python3
"""
Range-request throughput benchmark of /pointcloud-data/ (what the viewer does: many Range requests
over 6 parallel keep-alive connections, Chrome's per-host limit).

Usage: bench_range.py <url> [n_requests=1628] [parallel=6] [bytes_per_request=60000]
"""
import http.client
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse


def main():
    url = urlparse(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1628
    par = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
    path = url.path + ("?" + url.query if url.query else "")

    c = http.client.HTTPConnection(url.hostname, url.port)
    c.request("HEAD", path)
    r = c.getresponse(); r.read()
    total = int(r.getheader("Content-Length"))
    c.close()

    rng = random.Random(1)
    reqs = [rng.randrange(0, total - size) for _ in range(n)]
    chunks = [reqs[i::par] for i in range(par)]

    def worker(starts):
        conn = http.client.HTTPConnection(url.hostname, url.port)
        got = 0
        for s in starts:
            conn.request("GET", path, headers={"Range": f"bytes={s}-{s + size - 1}"})
            resp = conn.getresponse()
            data = resp.read()
            assert resp.status == 206 and len(data) == size, (resp.status, len(data))
            got += len(data)
        conn.close()
        return got

    t0 = time.time()
    with ThreadPoolExecutor(par) as ex:
        got = sum(ex.map(worker, chunks))
    dt = time.time() - t0
    print(f"{n} requests x {size} B, {par} parallel: {got / 1e6:.1f} MB in {dt:.2f} s -> {got / 1e6 / dt:.1f} MB/s, {n / dt:.0f} req/s")


if __name__ == "__main__":
    main()
