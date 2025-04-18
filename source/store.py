import faiss
import numpy as np
import pickle

def create_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, file_path: str):
    faiss.write_index(index, file_path)

def load_index(file_path: str):
    return faiss.read_index(file_path)

def save_chunks(chunks, path):
    with open(path, 'wb') as f:
        pickle.dump(chunks, f)

def load_chunks(path):
    with open(path, 'rb') as f:
        return pickle.load(f)