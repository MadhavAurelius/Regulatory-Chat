import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re

# ==============================
# Globals
# ==============================
client = None
collection = None
embedding_model = None
qa_pipeline = None

# ==============================
# Startup models
# ==============================
def startup_models():
    global client, collection, embedding_model, qa_pipeline

    if client is None:
        # 🔹 Chroma DB
        client = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_collection(name="bank_docs")

        # 🔹 Embedding model
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # 🔹 QA model (legal/regulatory)
        MODEL_NAME = "deepset/deberta-v3-large-squad2"
        print(f"🔹 Loading {MODEL_NAME}...")
        qa_pipeline = pipeline("question-answering", model=MODEL_NAME, tokenizer=MODEL_NAME)
        print("✅ Models loaded successfully!")

# ==============================
# OCR cleanup
# ==============================
def fix_ocr(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r'(?<=\d),(?=\d)', '', text)  # 1,00,000 → 100000
    return text

# ==============================
# Remove legal noise
# ==============================
def clean_legal_noise(text: str) -> str:
    # Remove page numbers
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', ' ', text, flags=re.I)

    # Remove markdown headings
    text = re.sub(r'#+\s*', ' ', text)

    # Remove excessive stars
    text = re.sub(r'\*+', '', text)

    return text

# ==============================
# Convert symbols → natural words
# ==============================
def normalize_symbols(text: str) -> str:
    replacements = {
        ">=": " greater than or equal to ",
        "<=": " less than or equal to ",
        ">": " greater than ",
        "<": " less than ",
        "=": " equal to ",
        "%": " percent ",
        "Rs.": " rupees ",
        "₹": " rupees ",
    }

    for sym, word in replacements.items():
        text = text.replace(sym, word)

    return text

# ==============================
# Convert table rows → sentences
# ==============================
def table_to_sentences(text: str) -> str:
    lines = text.split("\n")
    sentences = []

    for line in lines:
        if "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]

            if len(cells) == 2:
                sentences.append(f"{cells[0]} is {cells[1]}.")
            elif len(cells) == 3:
                sentences.append(f"{cells[0]} is {cells[1]} and duration is {cells[2]}.")
            elif len(cells) > 3:
                row = ", ".join(cells[:-1])
                sentences.append(f"{row} has value {cells[-1]}.")
        else:
            sentences.append(line)

    return " ".join(sentences)

# ==============================
# Master normalization pipeline
# ==============================
def normalize_context(text: str) -> str:
    text = fix_ocr(text)
    text = clean_legal_noise(text)
    text = table_to_sentences(text)
    text = normalize_symbols(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ==============================
# Remove duplicate sentences
# ==============================
def remove_duplicate_sentences(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen = set()
    result = []

    for s in sentences:
        key = re.sub(r'\s+', ' ', s.lower().strip())
        if key and key not in seen:
            result.append(s.strip())
            seen.add(key)

    return " ".join(result)

# ==============================
# Get supporting sentence
# ==============================
def get_support_sentences(answer: str, context: str, window: int = 1) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', context)

    for i, s in enumerate(sentences):
        if answer.lower() in s.lower():
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            return " ".join(sentences[start:end]).strip()

    return answer

# ==============================
# 🔥 MAIN RAG + QA FUNCTION
# ==============================
def get_llm_answer(query: str) -> str:
    global client, collection, embedding_model, qa_pipeline

    if qa_pipeline is None:
        startup_models()

    # 🔹 Encode query
    query_emb = embedding_model.encode(query).tolist()

    # 🔹 Retrieve top chunks (IMPORTANT)
    results = collection.query(query_embeddings=[query_emb], n_results=5)
    chunks = results["documents"][0]

    # Merge context
    context = " ".join(chunks)

    if not context.strip():
        return "Not found in context"

    # 🔹 Normalize context
    context = normalize_context(context)

    # 🔹 Run QA model
    result = qa_pipeline(question=query, context=context)
    answer_span = result.get("answer", "").strip()

    if not answer_span:
        return "Not found in context"

    # 🔹 Expand to full sentence
    snippet = get_support_sentences(answer_span, context, window=1)

    # 🔹 Remove duplicates
    final_answer = remove_duplicate_sentences(snippet)

    return final_answer if final_answer else answer_span

# ==============================
# Interactive testing
# ==============================
if __name__ == "__main__":
    print("💡 Regulatory Q&A System (type 'exit' to quit)\n")

    while True:
        query = input("Ask: ")

        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        print("\n🔹 Answer:\n")
        print(get_llm_answer(query))
        print("-" * 60)