import os
import numpy as np
import faiss
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai

# --------------------------
# Load API Key
# --------------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1"}
)

# --------------------------
# Detect Available Model Automatically
# --------------------------
print("Checking available Gemini models...")
available_models = [m.name for m in client.models.list()]

preferred_models = [
    "models/gemini-1.5-flash-8b",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro"
]

MODEL_NAME = None
for m in preferred_models:
    if m in available_models:
        MODEL_NAME = m.replace("models/", "")
        break

if MODEL_NAME is None:
    print("No preferred model found. Using first available model.")
    MODEL_NAME = available_models[0].replace("models/", "")

print(f"Using model: {MODEL_NAME}")

# --------------------------
# Load PDF
# --------------------------
print("Loading PDF...")
pdf_path = "swiggy_annual_report.pdf"
reader = PdfReader(pdf_path)

documents = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        documents.append({
            "page": i + 1,
            "text": text
        })

print(f"Loaded {len(documents)} pages.")

# --------------------------
# Chunk Text
# --------------------------
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

all_chunks = []

for doc in documents:
    chunks = chunk_text(doc["text"])
    for chunk in chunks:
        all_chunks.append({
            "page": doc["page"],
            "text": chunk
        })

print(f"Created {len(all_chunks)} chunks.")

# --------------------------
# Create Embeddings
# --------------------------
print("Creating embeddings...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in all_chunks]
embeddings = embedding_model.encode(texts, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("FAISS index ready.")

# --------------------------
# Retrieval Function
# --------------------------
def retrieve(query, top_k=6):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    retrieved_docs = []
    for idx in indices[0]:
        retrieved_docs.append(all_chunks[idx])

    return retrieved_docs

# --------------------------
# Generate Answer
# --------------------------
def generate_answer(query):
    retrieved_docs = retrieve(query)
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])

    prompt = f"""
You are a financial analyst assistant.

Answer ONLY using the provided context.
If the answer is not present in the context, say:
"I cannot find this information in the Swiggy Annual Report."

Do NOT use outside knowledge.

Context:
{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    print("\n==============================")
    print("ANSWER:\n")
    print(response.text)

    print("\nSOURCES:")
    for doc in retrieved_docs:
        print("Page:", doc["page"])
    print("==============================\n")

# --------------------------
# CLI Loop
# --------------------------
print("\nSwiggy Annual Report RAG (Gemini Powered)")
print("Type 'exit' to quit.\n")

while True:
    question = input("Ask a question: ")
    if question.lower() == "exit":
        break
    generate_answer(question)
