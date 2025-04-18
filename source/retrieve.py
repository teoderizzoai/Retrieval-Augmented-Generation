def search(index, query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)
    return indices