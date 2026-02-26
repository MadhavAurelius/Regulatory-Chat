import os
import chromadb
from sentence_transformers import SentenceTransformer
import re

# ----------------------------
# FOLDERS / DB
# ----------------------------
CHUNK_FOLDER = "chunks_individual"  # every chunk as a separate file
CHROMA_DB_PATH = "chroma_db"        # persistent ChromaDB folder
COLLECTION_NAME = "bank_docs"

# ----------------------------
# HELPER: Clean text for embedding
# ----------------------------
def clean_text(text: str) -> str:
    """
    Remove markdown symbols for better embeddings but keep meaning.
    """
    text = re.sub(r'#', '', text)         # remove headers
    text = re.sub(r'\*', '', text)        # remove asterisks
    text = re.sub(r'\n+', ' ', text)      # collapse newlines
    return text.strip()

# ----------------------------
# LOAD EMBEDDING MODEL
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# CREATE / GET CHROMA COLLECTION
# ----------------------------
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# ----------------------------
# STORE CHUNKS IN CHROMA
# ----------------------------
doc_id = 0

for file_name in os.listdir(CHUNK_FOLDER):
    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join(CHUNK_FOLDER, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        chunk_text = f.read()

    # clean text for embedding
    embedding_text = clean_text(chunk_text)
    embedding = model.encode(embedding_text).tolist()

    # add to Chroma
    collection.add(
        ids=[f"{file_name}_{doc_id}"],
        documents=[chunk_text],       # keep original chunk
        embeddings=[embedding],
        metadatas=[{"source": file_name}]
    )
    doc_id += 1

print("✅ All chunks stored in ChromaDB successfully!")
print("Total stored chunks:", collection.count())