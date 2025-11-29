import chromadb
from sentence_transformers import SentenceTransformer

# Load FREE local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB persistent storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or load resume collection
collection = chroma_client.get_or_create_collection(
    name="resumes",
    metadata={"hnsw:space": "cosine"}   # cosine similarity search
)

def embed_text(text: str):
    """Generate FREE local embeddings using SentenceTransformer."""
    if not text:
        text = " "   # avoid failure on empty extraction
    return model.encode(text).tolist()

def index_resume(resume_id: str, resume_text: str):
    """
    Store resume text + embedding in Chroma.
    Automatically removes the old entry if same ID exists.
    """
    # Delete old version of resume (avoid duplicate errors)
    try:
        collection.delete(ids=[resume_id])
    except:
        pass  # ignore if not present

    embedding = embed_text(resume_text)

    collection.add(
        ids=[resume_id],
        documents=[resume_text],
        embeddings=[embedding]
    )

def query_jd(job_description: str, top_k: int = 5):
    """Return top matching resumes sorted by semantic similarity."""
    jd_embedding = embed_text(job_description)

    results = collection.query(
        query_embeddings=[jd_embedding],
        n_results=top_k
    )

    return results
