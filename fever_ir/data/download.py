from pathlib import Path
import argparse
import requests
from tqdm import tqdm

BASE = "https://fever.ai/download/fever"


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"exists: {destination}")
        return
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(destination, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def run(args):
    raw = Path(args.raw_dir)
    for url_name, local_name in (("train.jsonl", "train.jsonl"), ("shared_task_dev.jsonl", "dev.jsonl"), ("shared_task_test.jsonl", "test.jsonl")):
        download_file(f"{BASE}/{url_name}", raw / local_name)
    if args.full:
        download_file(f"{BASE}/wiki-pages.zip", raw / "wiki-pages.zip")
    print("Download complete. Use --full to include the Wikipedia archive.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
