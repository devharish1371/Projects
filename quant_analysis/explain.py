from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def explain_market_event(events, index, similar_events, event_index):
    context = "\n\n".join([
        f"Date: {row['Date']}, Event: {row['Event_Type']}, News: {row['News_Headlines']}"
        for _, row in similar_events.iterrows()
    ])

    prompt = f"""
You are a financial analyst assistant.
Based on the following historical news and events similar to today's anomaly, explain what might be causing this market movement.

Similar Events:
{context}

Today's Event: {events.iloc[event_index]['Event_Type']}, News: {events.iloc[event_index]['News_Headlines']}

Explanation:
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
    )

    return response.choices[0].message.content

