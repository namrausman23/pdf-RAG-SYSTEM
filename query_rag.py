import requests
from pinecone import Pinecone

# ---------------------------
# CONFIG
# ---------------------------
PINECONE_API_KEY = "pcsk_6ANK2q_81jvb5TwP6RNzvrtc1VX8XWbBQXfxcL7KTpu1NHLz8bWnHxawzscd6U37pnQWoj"
INDEX_NAME = "pdf-rag-index"
OLLAMA_MODEL = "gemma3:4b"              # LLM model
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"  # Embedding model
TOP_K = 3                               # Number of relevant chunks to retrieve
# ---------------------------

# Initialize Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Step 1: Ask the user a question
query = input("Ask a question about your PDF: ")

# Step 2: Get embedding for the query from Ollama
try:
    emb_response = requests.post(
        "http://localhost:11434/v1/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "input": query}
    )
    emb_response.raise_for_status()
    query_embedding = emb_response.json()["data"][0]["embedding"]
except Exception as e:
    print("❌ Error getting embedding:", e)
    exit()

# Step 3: Query Pinecone for relevant chunks
results = index.query(vector=query_embedding, top_k=TOP_K, include_metadata=True)
chunks = [match["metadata"]["text"] for match in results["matches"]]

if not chunks:
    print("❌ No relevant chunks found in Pinecone.")
    exit()

# Step 4: Ask Ollama to answer using the retrieved chunks
context = "\n\n".join(chunks)
prompt = f"Use the following context to answer the question:\n{context}\n\nQuestion: {query}\nAnswer:"

try:
    answer_response = requests.post(
        "http://localhost:11434/v1/completions",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "max_tokens": 500}
    )
    answer_response.raise_for_status()
    answer_json = answer_response.json()
    answer = answer_json.get("completion") or answer_json.get("choices", [{}])[0].get("text", "")
except Exception as e:
    print("❌ Error getting completion:", e)
    exit()

print("\n💡 Answer:\n", answer)


