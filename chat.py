from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_drift(results):
    prompt = f"""
You are a data scientist.

Drift results:
{json.dumps(results, indent=2)}

Explain:
1. Which features drifted most
2. Why it might have happened
3. What actions to take

Respond in JSON format:
{{
  "summary": "...",
  "issues": "...",
  "actions": "..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)