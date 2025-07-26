import faiss

def build_faiss_index(embedding_matrix):
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    return index

def find_similar_events(index, events, event_index, k=5):
    query_vector = np.array(events.iloc[event_index]['Embedding'], dtype='float32').reshape(1, -1)
    D, I = index.search(query_vector, k)
    return events.iloc[I[0]]

