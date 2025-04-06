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
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",torch_dtype=torch_dtype, device=0 if torch.cuda.is_available() else -1)

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

# from fastapi import FastAPI
# from fastapi.responses import JSONResponse
# import torch
# from torch.nn.attention import SDPBackend, sdpa_kernel
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from datasets import load_dataset
# from tqdm import tqdm

# # Setup
# torch.set_float32_matmul_precision("high")
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# model_id = "openai/whisper-large-v3"

# # Load model
# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
# ).to(device)
# model.generation_config.cache_implementation = "static"
# model.generation_config.max_new_tokens = 256
# model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

# # Processor and pipeline
# processor = AutoProcessor.from_pretrained(model_id)
# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     torch_dtype=torch_dtype,
#     device=device,
# )

# # Warm-up
# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]
# for _ in tqdm(range(2), desc="Warm-up step"):
#     with sdpa_kernel(SDPBackend.MATH):
#         pipe(sample.copy(), generate_kwargs={"min_new_tokens": 256, "max_new_tokens": 256})

# # FastAPI app
# app = FastAPI()

# @app.get("/transcribe/")
# def transcribe_sample():
#     with sdpa_kernel(SDPBackend.MATH):
#         result = pipe(sample.copy())
#     return JSONResponse({"transcription": result["text"]})

# Run using: uvicorn whisper_server:app --host 0.0.0.0 --port 8000



# pip install transformers accelerate torch fastapi uvicorn
# pip uninstall torch -y
# pip cache purge  # clear old torch wheels
# pip install torch --index-url https://download.pytorch.org/whl/cu121
# sudo apt update && sudo apt install ffmpeg -y
