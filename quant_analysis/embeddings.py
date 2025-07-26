from sentence_transformers import SentenceTransformer
import numpy as np

def compute_embeddings(events):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    events['News_Headlines'] = events['News_Headlines'].fillna('')
    embeddings = model.encode(events['News_Headlines'].tolist(), show_progress_bar=True)
    events['Embedding'] = embeddings.tolist()
    return events, np.array(embeddings).astype('float32')

