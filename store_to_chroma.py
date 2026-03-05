import os
import sys
import chromadb

sys.path.append("src")

from sentence_transformers import SentenceTransformer
from chunknorris.parsers.pdf import PdfParser
from chunknorris.chunkers.markdown_chunker import MarkdownChunker


PDF_FOLDER = "Sebi"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "sebi_docs"

# Folder to store chunk files
CHUNK_OUTPUT_FOLDER = "chunks"

# create chunk folder if not exist
os.makedirs(CHUNK_OUTPUT_FOLDER, exist_ok=True)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

parser = PdfParser(use_ocr="never")
chunker = MarkdownChunker()

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

doc_id = 0

for file in os.listdir(PDF_FOLDER):

    if not file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_FOLDER, file)

    print("Processing:", file)

    parsed_md = parser.parse_file(pdf_path)

    chunks = chunker.chunk(parsed_md)

    print("Chunks:", len(chunks))

    chunk_index = 0

    for chunk in chunks:

        text = chunk.get_text()

        # -----------------------------
        # SAVE CHUNK AS FILE
        # -----------------------------
        chunk_filename = f"{file}_chunk_{chunk_index}.txt"
        chunk_path = os.path.join(CHUNK_OUTPUT_FOLDER, chunk_filename)

        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(text)

        # -----------------------------
        # VECTOR EMBEDDING  
        # -----------------------------
        embedding = embedding_model.encode(text).tolist()

        collection.add(
            ids=[f"{file}_{doc_id}"],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"source": file}]
        )

        doc_id += 1
        chunk_index += 1


print("Finished")
print("Total chunks:", collection.count())
