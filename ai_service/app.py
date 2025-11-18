from fastapi import FastAPI, Request
from openai import OpenAI
import joblib
import json

app = FastAPI()
client = OpenAI()

# Load anomaly detection model
model = joblib.load("model.pkl")

@app.post("/enrich")
async def enrich(request: Request):
    data = await request.json()

    features = extract_features(data)
    anomaly_score = model.decision_function([features])[0]

    llm_summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a threat analyst."},
            {"role": "user", "content": f"Summarize this alert: {json.dumps(data)}"}
        ]
    )

    return {
        "anomaly_score": float(anomaly_score),
        "summary": llm_summary.choices[0].message["content"]
    }
