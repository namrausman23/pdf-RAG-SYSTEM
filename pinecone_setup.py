from pinecone import Pinecone, ServerlessSpec

# ---------------------------
# CONFIG
# ---------------------------
PINECONE_API_KEY = "pcsk_6ANK2q_81jvb5TwP6RNzvrtc1VX8XWbBQXfxcL7KTpu1NHLz8bWnHxawzscd6U37pnQWoj"   # 🔑 replace with your Pinecone key
INDEX_NAME = "pdf-rag-index"
DIMENSION = 768                          # Ollama embedding size
CLOUD = "aws"                            # or "gcp" depending on your Pinecone project
REGION = "us-east-1"                     # pick the region from your Pinecone console
# ---------------------------

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Delete the index if it already exists
if INDEX_NAME in [i.name for i in pc.list_indexes()]:
    print(f"Deleting existing index '{INDEX_NAME}'...")
    pc.delete_index(INDEX_NAME)

# Create a new index with the correct dimension
print(f"Creating new index '{INDEX_NAME}' with dimension {DIMENSION}...")
pc.create_index(
    name=INDEX_NAME,
    dimension=DIMENSION,
    metric="cosine",
    spec=ServerlessSpec(
        cloud=CLOUD,
        region=REGION
    )
)

print(f"✅ Pinecone index '{INDEX_NAME}' ready with dimension {DIMENSION}")

