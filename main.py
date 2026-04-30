from fastapi import FastAPI, UploadFile
import pandas as pd
from drift import detect_drift  
from chat import explain_drift
from fastapi.responses import HTMLResponse
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/detect-drift/")
async def detect(file_old: UploadFile, file_new: UploadFile):
    df_old = pd.read_csv(file_old.file)
    df_new = pd.read_csv(file_new.file)

    # Step 1: calculate drift
    result = detect_drift(df_old, df_new)

    # Step 2: generate explanation using AI
    explanation = explain_drift(result)

    # Step 3: return both
    return {
        "drift_result": result,
        "ai_explanation": explanation
    }