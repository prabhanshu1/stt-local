from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel
import tempfile

app = FastAPI()

# Load Whisper model (modify for performance)
model = WhisperModel("tiny", device="cpu", compute_type="float32")

@app.post("/transcribe/")
async def transcribe(audio: UploadFile = File(...)):
    print("printing audio: ", audio)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    segments, _ = model.transcribe(tmp_path)

    transcript = " ".join(segment.text for segment in segments)
    return {"transcription": transcript}

# Run using: uvicorn whisper_server:app --host 0.0.0.0 --port 8000

