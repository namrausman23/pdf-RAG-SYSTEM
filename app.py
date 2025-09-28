import gradio as gr
import requests
from pinecone import Pinecone

# ---------------------------
# CONFIG
PINECONE_API_KEY = "pcsk_6ANK2q_81jvb5TwP6RNzvrtc1VX8XWbBQXfxcL7KTpu1NHLz8bWnHxawzscd6U37pnQWoj"
INDEX_NAME = "pdf-rag-index"
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
TOP_K = 3
# ---------------------------

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def answer_question(query):
    try:
        # Step 1: Get embedding from Ollama
        emb_response = requests.post(
            "http://localhost:11434/v1/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "input": query}
        )
        emb_response.raise_for_status()
        query_embedding = emb_response.json()["data"][0]["embedding"]
    except Exception as e:
        return f"❌ Error getting embedding: {e}"

    # Step 2: Query Pinecone
    results = index.query(vector=query_embedding, top_k=TOP_K, include_metadata=True)
    chunks = [match["metadata"]["text"] for match in results["matches"]]

    if not chunks:
        return "⚠️ No relevant chunks found."

    # Step 3: Ask Ollama for answer
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
        return f"❌ Error getting answer: {e}"

    return answer

# Create Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about your PDF..."),
    outputs="text",
    title="📄 PDF RAG System",
    description="Ask questions about your ingested PDF and get answers."
)

if __name__ == "__main__":
    iface.launch()
