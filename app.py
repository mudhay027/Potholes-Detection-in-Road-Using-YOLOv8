from flask import Flask, render_template, request, jsonify, send_from_directory
import os, cv2, numpy as np, threading, time
from ultralytics import YOLO
from collections import deque
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---- Paths (absolute paths are safer on PythonAnywhere) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best.pt')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Optional: limit upload size (e.g., 100 MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# ---- Load YOLO (CPU) ----
model = YOLO(MODEL_PATH)

# ---- Global progress state ----
# NOTE: simple single-job tracker. For multi-users, use a dict keyed by job id/filename.
progress = {"upload": 0, "processing": 0, "eta": "--", "file": None}
damage_deque = deque(maxlen=20)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # reset progress for new job
    progress.update({"upload": 0, "processing": 0, "eta": "--", "file": None})

    file = request.files.get('video')
    if not file or file.filename == '':
        return "No file uploaded", 400

    # Save uploaded file
    safe_name = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(upload_path)
    progress["upload"] = 100

    # Force MP4 output filename
    base, _ = os.path.splitext(safe_name)
    result_filename = f"processed_{base}.mp4"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    progress["file"] = result_filename

    # Start processing in a background thread so the request can return immediately
    t = threading.Thread(target=process_video, args=(upload_path, result_path), daemon=True)
    t.start()

    # Return immediately so frontend can poll /progress
    return jsonify({"result_file": result_filename})

@app.route('/progress')
def get_progress():
    # Always return a safe JSON payload
    return jsonify({
        "upload": progress.get("upload", 0),
        "processing": progress.get("processing", 0),
        "eta": progress.get("eta", "--"),
        "file": progress.get("file")
    })

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)

def process_video(input_path, output_path):
    """Run YOLO segmentation, overlay % damage, write MP4, and update progress + ETA."""
    try:
        cap = cv2.VideoCapture(input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        # MP4 writer (OpenCV manylinux wheels include ffmpeg in headless builds)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = fps if fps > 0 else 20.0
        width = width if width > 0 else 640
        height = height if height > 0 else 480

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)


        font = cv2.FONT_HERSHEY_SIMPLEX
        text_position = (40, 80)
        font_color = (255, 255, 255)
        background_color = (0, 0, 255)

        damage_deque.clear()
        processed = 0
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[DEBUG] Frame read failed at {processed}/{frame_count}")
                break

            print(f"[DEBUG] Processing frame {processed+1}/{frame_count}")


            # Inference (CPU)
            results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
            processed_frame = results[0].plot(boxes=False)

            # Compute damage % from masks
            percentage_damage = 0.0
            if results[0].masks is not None:
                total_area = 0.0
                masks = results[0].masks.data.cpu().numpy()
                image_area = float(frame.shape[0] * frame.shape[1])
                for mask in masks:
                    binary = (mask > 0).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        total_area += cv2.contourArea(c)
                if image_area > 0:
                    percentage_damage = (total_area / image_area) * 100.0

            damage_deque.append(percentage_damage)
            smoothed_damage = sum(damage_deque) / len(damage_deque)

            # Annotate
            cv2.line(processed_frame, (text_position[0], text_position[1]-10),
                     (text_position[0] + 380, text_position[1]-10), background_color, 40)
            cv2.putText(processed_frame, f'Road Damage: {smoothed_damage:.2f}%',
                        text_position, font, 1, font_color, 2, cv2.LINE_AA)

            out.write(processed_frame)

            processed += 1
            progress["processing"] = int((processed / frame_count) * 100)

            # ETA (based on actual processing speed)
            elapsed = max(time.time() - t_start, 1e-3)
            fps_proc = processed / elapsed
            remain_frames = max(frame_count - processed, 0)
            eta_sec = int(remain_frames / fps_proc) if fps_proc > 0 else 0
            progress["eta"] = eta_sec

        cap.release()
        out.release()
        print(f"[DEBUG] Video saved to {output_path}, size: {os.path.getsize(output_path)} bytes")
        progress["processing"] = 100
        progress["eta"] = 0
    except Exception as e:
        # In case of crash, surface info in logs and frontend
        progress["eta"] = "--"
        progress["processing"] = 0
        print("Processing error:", e)

if __name__ == "__main__":
    # For local testing only. On PythonAnywhere, WSGI runs the app.
    app.run(host='0.0.0.0',debug=True)
