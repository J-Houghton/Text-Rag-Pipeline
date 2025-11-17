import csv
from pathlib import Path
import tiktoken

# Config
INPUT_DIRS = [Path("001"), Path("002")]  # folders next to this script

OUTPUT_BASENAME = "chunks"  # will create chunks_001.csv, chunks_002.csv, ...
MAX_BYTES = 9 * 1024 * 1024  # ~9 MB

CHUNK_SIZE_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 70
MIN_DOC_TOKENS_SINGLE_CHUNK = 220  # below this, keep as one chunk

# For text-embedding-3-* models
encoding = tiktoken.get_encoding("cl100k_base")


def clean_text(text: str) -> str:
    """
    Light cleanup for OCR text.
    - Normalize line breaks.
    - Strip leading or trailing spaces on each line.
    - Drop empty lines.
    - Replace remaining newlines with a single space so CSV rows stay on one line.
    """
    if text is None:
        return ""

    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip spaces and drop empty lines
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    # Join with a space so we have one logical line
    text = " ".join(lines)

    # Collapse multiple spaces
    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()


def split_into_token_chunks(text: str,
                            chunk_size: int = CHUNK_SIZE_TOKENS,
                            chunk_overlap: int = CHUNK_OVERLAP_TOKENS):
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
    doc_id = parts[-1]  # just the ID part
    source_group = path.parent.name
    return doc_id, source_group


def process_file(path: Path, writer: csv.DictWriter):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    text = clean_text(raw_text)
    if not text:
        return

    doc_id, source_group = derive_ids_from_path(path)
    tokens = encoding.encode(text)
    n_tokens = len(tokens)

    # Short document, single chunk
    if n_tokens < MIN_DOC_TOKENS_SINGLE_CHUNK:
        chunk_id = f"{doc_id}_c001"
        writer.writerow({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "source_group": source_group,
            "text": text,
        })
        return

    # Long document, split into overlapping chunks
    chunk_index = 1
    for chunk_text in split_into_token_chunks(text):
        chunk_id = f"{doc_id}_c{chunk_index:03d}"
        writer.writerow({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "source_group": source_group,
            "text": chunk_text,
        })
        chunk_index += 1


def open_new_output_file(index: int):
    """
    Open a new CSV file for writing and return (file_handle, writer).
    """
    filename = f"{OUTPUT_BASENAME}_{index:03d}.csv"
    f_out = open(filename, "w", newline="", encoding="utf-8")
    fieldnames = ["doc_id", "chunk_id", "source_group", "text"]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    return f_out, writer


def main():
    # Collect all txt paths first to have a predictable order
    all_txt_paths = []
    for input_dir in INPUT_DIRS:
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"Warning: {input_dir} not found or not a directory, skipping")
            continue
        all_txt_paths.extend(sorted(input_dir.glob("*.txt")))

    if not all_txt_paths:
        print("No .txt files found in input directories.")
        return

    file_index = 1
    f_out, writer = open_new_output_file(file_index)

    try:
        for path in all_txt_paths:
            # Write rows for this file
            process_file(path, writer)

            # Check size and rotate file if needed
            if f_out.tell() >= MAX_BYTES:
                f_out.close()
                file_index += 1
                f_out, writer = open_new_output_file(file_index)

    finally:
        f_out.close()

    print(f"Done. Created {file_index} CSV file(s) with chunks.")


if __name__ == "__main__":
    main()
