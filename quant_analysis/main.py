from config import *
from data_loader import fetch_market_data
from anomaly_detection import detect_anomalies
from news_retrieval import enrich_with_news
from embeddings import compute_embeddings
from similar_events import build_faiss_index, find_similar_events
from explain import explain_market_event

def main():
    print("Fetching market data...")
    data = fetch_market_data(TICKER, BENCHMARK, START_DATE, END_DATE)

    print("Detecting anomalies...")
    events = detect_anomalies(data, Z_THRESHOLD, VOL_WINDOW, VOL_MULTIPLIER, TICKER)

    print("Fetching news...")
    events = enrich_with_news(events, TICKER)

    print("Generating embeddings...")
    events, embedding_matrix = compute_embeddings(events)

    print("Building FAISS index...")
    index = build_faiss_index(embedding_matrix)

    print("Finding similar events...")
    event_index = len(events) - 4  # Example: last few events
    similar = find_similar_events(index, events, event_index)

    print("Explaining market movement...")
    explanation = explain_market_event(events, index, similar, event_index)
    print(explanation)

if __name__ == "__main__":
    main()

