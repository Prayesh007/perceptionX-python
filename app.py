# import sys
# import time
# import asyncio
# import base64
# import requests
# from motor.motor_asyncio import AsyncIOMotorClient
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
# import cv2
# import tempfile
# import subprocess
# import os
# import imageio_ffmpeg as ffmpeg
# from io import BytesIO
# from bson import ObjectId

# # Matplotlib non-GUI setup
# os.environ['MPLCONFIGDIR'] = '/tmp'
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# # For Ultralytics on Render
# os.environ["ULTRALYTICS_CONFIG_DIR"] = "/tmp/Ultralytics"

# # MongoDB Config
# MONGO_URI = "mongodb+srv://aitools2104:kDTRxzV6MgO4nicA@cluster0.tqkyb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&tls=true&tlsInsecure=false"
# DATABASE_NAME = "test"
# COLLECTION_NAME = "files"

# # Mongo Connection
# client = AsyncIOMotorClient(MONGO_URI)
# db = client[DATABASE_NAME]
# collection = db[COLLECTION_NAME]

# # YOLO Weights
# weights_path = "./yolov11/best.pt"
# weights_url = "https://your-domain.com/path-to-best.pt"  # Replace this with your actual model file URL

# def load_model():
#     print(f"🔍 Checking if model exists: {weights_path}")
    
#     if not os.path.exists(weights_path):
#         print("⚠️ best.pt not found. Downloading...")
#         try:
#             os.makedirs(os.path.dirname(weights_path), exist_ok=True)
#             response = requests.get(weights_url)
#             with open(weights_path, "wb") as f:
#                 f.write(response.content)
#             print("✅ Model downloaded successfully.")
#         except Exception as e:
#             print(f"❌ Error downloading model: {e}")
#             exit(1)

#     print(f"✅ Model exists at: {weights_path}")
#     model = YOLO(weights_path)
#     print("🧠 YOLO model loaded.")
#     return model

# async def fetch_file_from_mongo(file_id):
#     print(f"🔎 Fetching file from MongoDB ID: {file_id}")
#     try:
#         object_id = ObjectId(file_id)
#     except Exception as e:
#         print(f"❌ Invalid ObjectId: {e}")
#         return None

#     file = await collection.find_one({"_id": object_id})
#     if file and file.get("data"):
#         print(f"📦 File found: {file['filename']}")
#         return file["data"]
#     else:
#         print("❌ File not found or data missing.")
#         return None

# async def process_image(file_id, model):
#     image_data = await fetch_file_from_mongo(file_id)
#     if not image_data:
#         print("❌ No image data to process.")
#         return

#     try:
#         image = Image.open(BytesIO(image_data))
#         image.verify()
#         image = Image.open(BytesIO(image_data)).convert("RGB")
#     except Exception as e:
#         print(f"❌ Image error: {e}")
#         return

#     image_np = np.array(image)
#     results = model(image_np)
#     annotated_image = results[0].plot()

#     _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#     processed_image_binary = buffer.tobytes()

#     await save_processed_file(file_id, processed_image_binary)

# async def process_video(file_id, model):
#     file_data = await fetch_file_from_mongo(file_id)
#     if not file_data:
#         print("❌ No video data found.")
#         return

#     temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
#     with open(temp_video_path, "wb") as f:
#         f.write(file_data)

#     cap = cv2.VideoCapture(temp_video_path)
#     if not cap.isOpened():
#         print("❌ Error opening video.")
#         return

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     processed_frames = 0
#     temp_output = os.path.join(tempfile.gettempdir(), "temp_output.avi")

#     fourcc = cv2.VideoWriter_fourcc(*"XVID")
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results = model(frame)
#         annotated_frame = results[0].plot()
#         out.write(annotated_frame)
#         processed_frames += 1
#         print(f"📽️ Processing... {int((processed_frames / total_frames) * 100)}%")

#     cap.release()
#     out.release()

#     processed_video_path = convert_to_mp4(temp_output)
#     with open(processed_video_path, "rb") as f:
#         processed_video_binary = f.read()

#     await save_processed_file(file_id, processed_video_binary)

# def convert_to_mp4(input_file):
#     ffmpeg_command = ffmpeg.get_ffmpeg_exe()
#     output_file = input_file.replace(".avi", ".mp4")
#     command = [ffmpeg_command, "-y", "-i", input_file, "-vcodec", "libx264", "-crf", "23", "-preset", "fast", output_file]
#     subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     return output_file

# async def save_processed_file(file_id, processed_data):
#     try:
#         object_id = ObjectId(file_id)
#     except Exception as e:
#         print(f"❌ Invalid ObjectId: {e}")
#         return

#     result = await collection.update_one(
#         {"_id": object_id},
#         {"$set": {"processedData": processed_data}}
#     )

#     # if result.modified_count > 0:
#     #     print(f"✅ Processed data saved to MongoDB for ID: {file_id}")
#     # else:
#     #     print("⚠️ Failed to update document.")
#     if result.modified_count > 0:
#         print(f"✅ Processed file saved to MongoDB for file ID: {file_id}")
#     elif result.matched_count > 0:
#         print("⚠️ Document found, but data was already the same — nothing changed.")
#     else:
#         print("❌ No matching document found. Update failed.")


# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python app.py <file_id> <file_type>")
#         sys.exit(1)

#     file_id = sys.argv[1]
#     file_type = sys.argv[2]
#     model = load_model()

#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

#     if file_type.startswith("image"):
#         loop.run_until_complete(process_image(file_id, model))
#     elif file_type.startswith("video"):
#         loop.run_until_complete(process_video(file_id, model))
#     else:
#         print("❌ Invalid file type. Use 'image' or 'video'.")

# app.py

try:
    import requests
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


import sys
import asyncio
import os
import tempfile
import subprocess
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from bson import ObjectId
from bson.binary import Binary
import imageio_ffmpeg as ffmpeg
import requests

# --------------------- CONFIG ------------------------
FASTAPI_PORT = int(os.environ.get("PORT", 8000))  # Render sets PORT env
MONGO_URI = os.environ.get("MONGO_URI")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "test")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "files")
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "./yolov11/best.pt")
MODEL_WEIGHTS_URL = os.environ.get("MODEL_WEIGHTS_URL", "")

os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ["ULTRALYTICS_CONFIG_DIR"] = "/tmp/ultralytics"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

_MODEL = None

# --------------------- MODEL ------------------------
def download_weights_if_missing():
    if os.path.exists(WEIGHTS_PATH):
        return True
    if not MODEL_WEIGHTS_URL:
        print("No MODEL_WEIGHTS_URL provided")
        return False
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    r = requests.get(MODEL_WEIGHTS_URL, stream=True, timeout=60)
    r.raise_for_status()
    with open(WEIGHTS_PATH, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return True

def load_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(WEIGHTS_PATH):
            download_weights_if_missing()
        _MODEL = YOLO(WEIGHTS_PATH)
        print("🧠 YOLO model loaded.")
    return _MODEL

# --------------------- MONGO HELPERS ------------------------
async def fetch_file_from_mongo(file_id):
    try:
        doc = await collection.find_one({"_id": ObjectId(file_id)})
    except:
        return None, None
    if not doc or "data" not in doc:
        return None, None
    return bytes(doc["data"]), doc.get("mimetype", "")

async def save_processed_file(file_id, processed_bytes):
    try:
        obj_id = ObjectId(file_id)
    except:
        return False
    result = await collection.update_one(
        {"_id": obj_id},
        {"$set": {"processedData": Binary(processed_bytes)}}
    )
    return result.modified_count > 0 or result.matched_count > 0

# --------------------- PROCESSING ------------------------
async def process_image(file_id, model):
    data, _ = await fetch_file_from_mongo(file_id)
    if not data:
        return False
    img = Image.open(BytesIO(data)).convert("RGB")
    results = model(np.array(img))
    annotated = results[0].plot()
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".jpg", bgr)
    if not success:
        return False
    return await save_processed_file(file_id, buffer.tobytes())

async def process_video(file_id, model):
    data, _ = await fetch_file_from_mongo(file_id)
    if not data:
        return False
    tmp_in = os.path.join(tempfile.gettempdir(), f"input_{file_id}.mp4")
    tmp_out = os.path.join(tempfile.gettempdir(), f"out_{file_id}.avi")
    with open(tmp_in, "wb") as f:
        f.write(data)
    cap = cv2.VideoCapture(tmp_in)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        try:
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        except:
            annotated_bgr = annotated
        out.write(annotated_bgr)
    cap.release()
    out.release()
    # convert to mp4
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    mp4_path = tmp_out.replace(".avi", ".mp4")
    subprocess.run([ffmpeg_exe, "-y", "-i", tmp_out, "-vcodec", "libx264", "-crf", "23", "-preset", "fast", mp4_path],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open(mp4_path, "rb") as f:
        processed_video = f.read()
    for f in [tmp_in, tmp_out, mp4_path]:
        try: os.remove(f)
        except: pass
    return await save_processed_file(file_id, processed_video)

async def run_process(file_id, file_type):
    model = load_model()
    if file_type.startswith("image"):
        return await process_image(file_id, model)
    elif file_type.startswith("video"):
        return await process_video(file_id, model)
    else:
        return False

# --------------------- FASTAPI ------------------------
app = FastAPI(title="YOLO Processing Service")

class ProcessRequest(BaseModel):
    fileId: str
    fileType: str

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model)
    print("Service startup complete.")

@app.get("/")
async def root():
    return {"message": "YOLO backend is running!"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/process")
async def process_endpoint(payload: ProcessRequest):
    ok = await run_process(payload.fileId, payload.fileType)
    if not ok:
        raise HTTPException(status_code=500, detail="Processing failed")
    return {"status": "ok", "fileId": payload.fileId}

# --------------------- CLI ENTRY ------------------------
async def cli_main():
    if len(sys.argv) < 3:
        print("Usage: python app.py <file_id> <file_type>")
        sys.exit(1)
    ok = await run_process(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    args = parser.parse_args()
    if args.cli:
        asyncio.run(cli_main())
