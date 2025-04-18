from source.ingest import load_documents, chunk_text
from source.embed import get_model, embed_chunks
from source.store import create_faiss_index, save_index, save_chunks, load_index, load_chunks
from source.retrieve import search
from source.generate import get_generator, generate_answer
import numpy as np


print("🔹 Loading documents...")
docs = load_documents("data/raw_documents")

print("🔹 Chunking documents...")
all_chunks = []
for doc in docs:
    all_chunks.extend(chunk_text(doc))

print(f"✅ Loaded {len(all_chunks)} chunks")

print("🔹 Loading embedding model...")
embedder = get_model()

print("🔹 Embedding chunks...")
embeddings = embed_chunks(embedder, all_chunks).cpu().numpy()

print("🔹 Creating FAISS index...")
index = create_faiss_index(embeddings)
save_index(index, "data/index.faiss")
save_chunks(all_chunks, "data/chunks.pkl")

print("🔹 Awaiting user question...")
query = input("Ask a question: ")
query_vec = embedder.encode([query])

print("🔹 Retrieving top documents...")
indices = search(index, query_vec, k=5)[0]
retrieved_chunks = [all_chunks[i] for i in indices]

print("🔹 Loading generator...")
gen = get_generator()

print("🔹 Generating answer...")
answer = generate_answer(gen, query, retrieved_chunks)

print("\n🧠 Answer:")
print(answer)