import json
import time
import re
import os
import tiktoken
from pathlib import Path
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth

# ------------- CONFIG -------------
# Weaviate / OpenAI config   
load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY_EMBED = os.getenv("OPENAI_API_KEY_EMBED")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "MyDocs")


# Folders with .txt files
INPUT_DIRS = [Path("data/001"), Path("data/002")]

# Chunking settings
CHUNK_SIZE_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 70
MIN_DOC_TOKENS_SINGLE_CHUNK = 220  # below this, keep as one chunk

# Drop documents that are too small to be useful
MIN_WORDS_PER_DOC = 5


# Batch settings
BATCH_SIZE = 100              # smaller batch to keep each embedding request modest
PER_BATCH_SLEEP_SECONDS = 3   # wait after each full batch to reduce TPM pressure

# For text-embedding-3-* models
encoding = tiktoken.get_encoding("cl100k_base")

# ------------- TEXT CLEANING AND CHUNKING -------------

# Precompiled regex patterns for speed
OCR_JUNK_PATTERN = re.compile(r"[^A-Za-z0-9\s.,;:!?'\-()/]")
PAGE_MARKER_PATTERN = re.compile(r"page\s*\d+\s*of\s*\d+", re.IGNORECASE)


def clean_text(text: str) -> str:
    """
    Light but robust cleanup for OCR text.

    - Normalize line breaks
    - Remove obvious page markers
    - Strip OCR junk characters
    - Collapse repeated whitespace
    """
    if text is None:
        return ""

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove form feeds and similar control chars
    text = text.replace("\x0c", " ")

    # Strip leading/trailing whitespace on each line and drop empty lines
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    text = " ".join(lines)

    # Remove generic "page X of Y" markers
    text = PAGE_MARKER_PATTERN.sub(" ", text)

    # Remove non standard OCR garbage characters
    text = OCR_JUNK_PATTERN.sub(" ", text)

    # Normalize some common punctuation variants
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_token_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
):
    """
    Yield successive text chunks based on token counts.
    """
    tokens = encoding.encode(text)
    if not tokens:
        return

    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            continue
        yield encoding.decode(chunk_tokens)


def derive_ids_from_path(path: Path):
    """
    From folder and filename infer:
      - doc_id: last segment after underscore, e.g. 'PRE_FIX_12345' -> '12345'
      - source_group: parent folder name, e.g. '001' or '002'
    """
    stem = path.stem  # e.g. "PRE_FIX_12345"
    parts = stem.split("_")
    doc_id = parts[-1]  # just the ID part, still ignoring prefix as you currently do
    source_group = path.parent.name
    return doc_id, source_group


def generate_chunk_objects_from_file(path: Path):
    """
    Read a .txt file, clean it, chunk it, and yield dicts ready for Weaviate.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    text = clean_text(raw_text)
    if not text:
        return

    # Filter out documents that are too short to be meaningful
    word_count = len(text.split())
    if word_count < MIN_WORDS_PER_DOC:
        return

    doc_id, source_group = derive_ids_from_path(path)
    tokens = encoding.encode(text)
    n_tokens = len(tokens)

    if n_tokens < MIN_DOC_TOKENS_SINGLE_CHUNK:
        chunk_id = f"{doc_id}_c001"
        yield {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "source_group": source_group,
            "text": text,
        }
        return

    chunk_index = 1
    for chunk_text in split_into_token_chunks(text):
        chunk_id = f"{doc_id}_c{chunk_index:03d}"
        yield {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "source_group": source_group,
            "text": chunk_text,
        }
        chunk_index += 1


def iter_all_chunk_objects():
    """
    Iterate over all .txt files in INPUT_DIRS and yield chunk objects.
    """
    all_txt_paths = []
    for input_dir in INPUT_DIRS:
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"Warning: {input_dir} not found or not a directory, skipping")
            continue
        all_txt_paths.extend(sorted(input_dir.glob("*.txt")))

    if not all_txt_paths:
        print("No .txt files found in input directories.")
        return

    for path in all_txt_paths:
        for obj in generate_chunk_objects_from_file(path):
            yield obj


# ------------- WEAVIATE UPLOAD -------------


def connect_weaviate():
    """
    Connect to Weaviate Cloud with API key auth.
    """
    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        raise RuntimeError("WEAVIATE_URL and WEAVIATE_API_KEY must be set.")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-key": OPENAI_API_KEY_EMBED} if OPENAI_API_KEY_EMBED else {},
    )
    return client


def upload_chunks():
    # Materialize all objects so we can:
    # - Know total count up front
    # - Reuse them for retries if needed
    all_objects = list(iter_all_chunk_objects())
    total_objects = len(all_objects)
    print(f"Total objects to upload: {total_objects}")

    if total_objects == 0:
        print("Nothing to upload, exiting.")
        return

    client = connect_weaviate()
    collection = client.collections.get(COLLECTION_NAME)

    uploaded = 0
    failed_objects = []
    batch_counter = 0

    try:
        # Fixed batch size plus manual sleep between batches
        with collection.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
            for obj in all_objects:
                batch.add_object(properties=obj)
                uploaded += 1

                # Progress logging
                if uploaded % 1000 == 0 or uploaded == total_objects:
                    pct = uploaded / total_objects * 100
                    print(f"Uploaded {uploaded} / {total_objects} objects ({pct:.1f}%).")
 
                if uploaded % BATCH_SIZE == 0:
                    batch_counter += 1
                    time.sleep(PER_BATCH_SLEEP_SECONDS)

            # After the context, capture any failed objects recorded by the batcher
            if batch.failed_objects:
                print(f"{len(batch.failed_objects)} objects failed in batch, capturing for retry...")
                for fo in batch.failed_objects:
                    # fo.properties is the original dict we sent (new client)
                    if hasattr(fo, "properties") and fo.properties is not None:
                        failed_objects.append(fo.properties)
                    elif hasattr(fo, "object") and fo.object is not None:
                        props = fo.object.get("properties")
                        if props:
                            failed_objects.append(props)

    finally:
        client.close()

    print(f"Finished upload loop. Seen {uploaded} objects in total.")

    # Persist failed objects, if any
    if failed_objects:
        failed_path = Path("failed_objects.jsonl")
        with failed_path.open("w", encoding="utf-8") as f:
            for obj in failed_objects:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(
            f"Saved {len(failed_objects)} failed objects to {failed_path} "
            "for rerun with a smaller batch or later window."
        )
    else:
        print("No failed objects recorded by the batch client.")


if __name__ == "__main__":
    upload_chunks()
