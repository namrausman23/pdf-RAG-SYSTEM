import os
from pypdf import PdfReader
import requests
from pinecone import Pinecone, ServerlessSpec

# ---------------------------
# CONFIG
# ---------------------------
PINECONE_API_KEY = "pcsk_6ANK2q_81jvb5TwP6RNzvrtc1VX8XWbBQXfxcL7KTpu1NHLz8bWnHxawzscd6U37pnQWoj"   # Replace with your key
INDEX_NAME = "pdf-rag-index"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"  # Embedding model
PDF_PATH = "sample.pdf"                        # Your PDF file
CHUNK_SIZE = 500                             # Number of words per chunk
# ---------------------------

# Initialize Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # Must match embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")  # Required in latest SDK
    )
    print(f"✅ Created index '{INDEX_NAME}'")

# Connect to the index
index = pc.Index(INDEX_NAME)

# Step 1: Read PDF text
reader = PdfReader(PDF_PATH)
text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

# Step 2: Split text into chunks
def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

chunks = list(chunk_text(text))
print(f"📄 Total chunks created: {len(chunks)}")

# Step 3: Get embeddings from Ollama
def get_embedding(text_chunk):
    try:
        response = requests.post(
            "http://localhost:11434/v1/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "input": text_chunk}
        )
        response.raise_for_status()
        resp_json = response.json()
        return resp_json["data"][0]["embedding"]
    except Exception as e:
        print("❌ Error getting embedding:", e)
        if response is not None:
            print("Raw response:", response.text)
        return None

# Step 4: Push chunks to Pinecone
vectors = []
for i, chunk in enumerate(chunks):
    embedding = get_embedding(chunk)
    if embedding:
        vectors.append({"id": f"chunk-{i}", "values": embedding, "metadata": {"text": chunk}})

if vectors:
    index.upsert(vectors=vectors)
    print(f"✅ Ingested {len(vectors)} chunks from {PDF_PATH} into Pinecone index '{INDEX_NAME}'")
else:
    print("❌ No vectors ingested. Check Ollama server and PDF content.")






