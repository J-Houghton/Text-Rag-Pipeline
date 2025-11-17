# Text-to-Weaviate RAG Pipeline

Small, reusable project to:

1. Chunk raw `.txt` documents.
2. Upload them into a Weaviate Cloud collection using OpenAI embeddings.
3. Run clustering and basic validation in notebooks.

Once a Weaviate cluster and collection are created, `upload_chunks.py` calls the OpenAI embedding model via Weaviate and creates objects with vector info in the collection.
Further analysis (e.g. k-means clusters) is done from Jupyter notebooks.

---

## Project layout

Suggested layout:

```text
project-root/
├─ README.md
├─ .env                # local env vars (not committed)
├─ requirements.txt    # Python dependencies (optional)
├─ upload_chunks.py    # main ingestion script -> Weaviate
├─ chunk_txt_to_csv.py # legacy chunking to CSV (not used in current flow)
├─ token_count.py      # counts tokens in data dirs (001/, 002/)
├─ notebooks/
│  ├─ clustering_new.ipynb
│  ├─ 02_kmeans_clusters.ipynb
│  └─ validate_clusters.ipynb
└─ data/
   ├─ 001/             # input .txt files
   └─ 002/
