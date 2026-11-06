from pathlib import Path
import argparse
import json
import re
import zipfile
from tqdm import tqdm


def normalize(text):
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def read_claims(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            row["claim"] = normalize(row.get("claim", ""))
            rows.append(row)
    return rows


def iter_wiki_pages(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        for member in (n for n in z.namelist() if n.endswith(".jsonl")):
            with z.open(member) as raw:
                for line in raw:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Some mirrors include a metadata/header line in a JSONL member.
                        continue


def prepare_claims(raw_dir, processed_dir, limit):
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        src = raw_dir / f"{split}.jsonl"
        if not src.exists():
            continue
        rows = read_claims(src, limit)
        with open(processed_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"prepared {split}: {len(rows):,} claims")


def prepare_passages(raw_dir, processed_dir, max_passages=None):
    zip_path = raw_dir / "wiki-pages.zip"
    output = processed_dir / "passages.jsonl"
    if not zip_path.exists():
        print("Wikipedia archive not present; skipping passages. Run download --full for evidence retrieval.")
        return 0
    count = 0
    with open(output, "w", encoding="utf-8") as out:
        for page in tqdm(iter_wiki_pages(zip_path), desc="extracting wiki passages"):
            title = page.get("id") or page.get("title")
            for line in page.get("lines", "").splitlines():
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                sid, text = parts
                text = normalize(text)
                if text:
                    out.write(json.dumps({"passage_id": f"{title}::{sid}", "title": title, "sentence_id": sid, "text": text}, ensure_ascii=False) + "\n")
                    count += 1
                    if max_passages and count >= max_passages:
                        return count
    return count


def run(args):
    raw, processed = Path(args.raw_dir), Path(args.processed_dir)
    prepare_claims(raw, processed, args.sample)
    count = prepare_passages(raw, processed)
    with open(processed / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"sample_size": args.sample, "passage_count": count, "source": "FEVER official download"}, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
