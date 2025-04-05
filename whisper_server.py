# from fastapi import FastAPI, File, UploadFile
# from faster_whisper import WhisperModel
# import tempfile
# import logging
# import time

# logging.basicConfig()
# logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

# app = FastAPI()

# # Load Whisper model (modify for performance)
# model = WhisperModel("base", device="cpu", compute_type="float32")

# @app.post("/transcribe/")
# async def transcribe(audio: UploadFile = File(...)):
#     start_time = time.time()
#     with tempfile.NamedTemporaryFile(delete=False) as tmp:
#         tmp.write(await audio.read())
#         tmp_path = tmp.name

#     segments, _ = model.transcribe(tmp_path)

#     transcript = " ".join(segment.text for segment in segments)
#     logging.info(f"Transcription: {transcript}")
#     logging.info(f"Execution Time: {time.time() - start_time:.4f} secconds")
#     return {"transcription": transcript}



# OpenAPI whisper
from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import torch
import tempfile
import shutil
import time


app = FastAPI()

# Load the Whisper model pipeline
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0 if torch.cuda.is_available() else -1)

@app.post("/transcribe/")
async def transcribe_audio(audio: UploadFile = File(...)):
    start_time = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    result = pipe(tmp_path)
    end = time.time()
    print(f"Execution time: {end - start_time:.3f} seconds")
    return {"transcription": result["text"]}

# Run using: uvicorn whisper_server:app --host 0.0.0.0 --port 8000



# pip install transformers accelerate torch fastapi uvicorn
# pip uninstall torch -y
# pip cache purge  # clear old torch wheels
# pip install torch --index-url https://download.pytorch.org/whl/cu121
# sudo apt update && sudo apt install ffmpeg -y
