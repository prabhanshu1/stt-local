from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel
import tempfile
import logging
import time

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

app = FastAPI()

# Load Whisper model (modify for performance)
model = WhisperModel("turbo", device="cuda", compute_type="float16")

@app.post("/transcribe/")
async def transcribe(audio: UploadFile = File(...)):
    start_time = time.time()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    segments, _ = model.transcribe(tmp_path)

    transcript = " ".join(segment.text for segment in segments)
    logging.logger.info(f"Transcription: {transcript}")
    logging.logger.info(f"Execution Time: {time.time() - start_time:.4f} secconds")
    return {"transcription": transcript}

# Run using: uvicorn whisper_server:app --host 0.0.0.0 --port 8000

