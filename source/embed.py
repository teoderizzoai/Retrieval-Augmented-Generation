from sentence_transformers import SentenceTransformer

def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(model, chunks):
    return model.encode(chunks, convert_to_tensor=True)