import os
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify
import time
from collections import defaultdict, deque
import threading
import queue
import uuid

# Configuration
WEIGHTS = 'models/yolo11n.pt'
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
HEAVY_TRAFFIC_THRESHOLD = 5
UPLOAD_DIR = 'temp'
FRAME_SKIP = 2
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car=2, motorcycle=3, bus=5, truck=7
TRACKING_EXPIRY = 5.0
DENSITY_RED_THRESHOLD = 0.06  # Density threshold for red color

# Lane vertices and thresholds
VERTICES1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
VERTICES2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)
X1 = 325
X2 = 635
LANE_THRESHOLD = 609

# Traffic light colors
RED_COLOR = (0, 0, 255)
YELLOW_COLOR = (0, 255, 255)
GREEN_COLOR = (0, 255, 0)
OUTLINE_COLOR = (255, 255, 255)
HOUSING_COLOR = (0, 0, 0)

# Initialize YOLO model
try:
    print(f"Loading YOLO model from {WEIGHTS}")
    model = YOLO(WEIGHTS)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

# Global metrics with thread-safe access
global_metrics = strs = {
    'left_vehicles': 0,
    'right_vehicles': 0,
    'left_intensity': 'Smooth',
    'right_intensity': 'Smooth',
    'left_density': 0.0,
    'right_density': 0.0,
    'left_speed': 0.0,
    'right_speed': 0.0,
    'peak_vehicle_count': 0
}
metrics_lock = threading.Lock()

# Traffic light state tracking
red_light_start_frame = 0
red_light_duration = 3
is_red_light = False
yellow_start_frame = 0
is_yellow_light = False

# Vehicle tracking
vehicle_centroids = defaultdict(lambda: deque(maxlen=10))
tracked_vehicles = {}
unique_vehicle_ids = set()

# Heatmap storage
latest_heatmap = None
heatmap_lock = threading.Lock()

# Reset state function with initial heatmap
def reset_state():
    global global_metrics, red_light_start_frame, is_red_light, yellow_start_frame, is_yellow_light, vehicle_centroids, tracked_vehicles, unique_vehicle_ids, latest_heatmap
    with metrics_lock:
        global_metrics = {
            'left_vehicles': 0,
            'right_vehicles': 0,
            'left_intensity': 'Smooth',
            'right_intensity': 'Smooth',
            'left_density': 0.0,
            'right_density': 0.0,
            'left_speed': 0.0,
            'right_speed': 0.0,
            'peak_vehicle_count': 0
        }
    red_light_start_frame = 0
    is_red_light = False
    yellow_start_frame = 0
    is_yellow_light = False
    vehicle_centroids.clear()
    tracked_vehicles.clear()
    unique_vehicle_ids.clear()
    with heatmap_lock:
        # Create a default heatmap with green lanes
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.fillPoly(placeholder, [VERTICES1], GREEN_COLOR)
        cv2.fillPoly(placeholder, [VERTICES2], GREEN_COLOR)
        _, buffer = cv2.imencode('.jpg', placeholder)
        latest_heatmap = buffer.tobytes()

# Corrected Shoelace formula for lane area
def calculate_area(vertices):
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    area = abs(area) / 2.0
    print(f"Calculated area: {area} pixels^2")
    return area

left_lane_area = calculate_area(VERTICES1)
right_lane_area = calculate_area(VERTICES2)

def draw_traffic_light(frame, state, center_x, center_y):
    radius = 20
    spacing = 60
    housing_width = 60
    housing_height = 180
    top_left = (center_x - housing_width // 2, center_y - housing_height // 2)
    bottom_right = (center_x + housing_width // 2, center_y + housing_height // 2)
    cv2.rectangle(frame, top_left, bottom_right, HOUSING_COLOR, -1)
    cv2.circle(frame, (center_x, center_y - spacing), radius, RED_COLOR, -1 if state == 'Red' else 1)
    cv2.circle(frame, (center_x, center_y - spacing), radius, OUTLINE_COLOR, 1)
    cv2.circle(frame, (center_x, center_y), radius, YELLOW_COLOR, -1 if state == 'Yellow' else 1)
    cv2.circle(frame, (center_x, center_y), radius, OUTLINE_COLOR, 1)
    cv2.circle(frame, (center_x, center_y + spacing), radius, GREEN_COLOR, -1 if state == 'Green' else 1)
    cv2.circle(frame, (center_x, center_y + spacing), radius, OUTLINE_COLOR, 1)

def create_heatmap(frame, left_density, right_density):
    # Create heatmap with the same dimensions as the input frame
    heatmap = np.zeros_like(frame, dtype=np.uint8)
    
    # Determine colors based on density
    left_color = RED_COLOR if left_density > DENSITY_RED_THRESHOLD else GREEN_COLOR
    right_color = RED_COLOR if right_density > DENSITY_RED_THRESHOLD else GREEN_COLOR
    
    # Fill lane polygons with appropriate colors
    cv2.fillPoly(heatmap, [VERTICES1], left_color)
    cv2.fillPoly(heatmap, [VERTICES2], right_color)
    
    # Blend heatmap with the original frame
    alpha = 0.5
    blended_frame = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0.0)
    
    return blended_frame, heatmap

def process_frame(frame, frame_count, fps, prev_metrics=None):
    global is_red_light, red_light_start_frame, is_yellow_light, yellow_start_frame, vehicle_centroids, tracked_vehicles, unique_vehicle_ids, latest_heatmap
    if frame is None or frame.size == 0:
        print(f"Invalid frame at {frame_count}: Empty or None")
        return frame

    resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    processed_frame = frame.copy()

    vehicles_in_left_lane = 0
    vehicles_in_right_lane = 0
    left_speeds = []
    right_speeds = []
    current_centroids = {}
    current_detections = []

    if frame_count % FRAME_SKIP == 0:
        try:
            results = model.predict(resized_frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)
        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            return frame

        if results[0].boxes:
            scaled_boxes = results[0].boxes.xyxy.cpu().numpy() * (frame.shape[1] / IMG_SIZE, frame.shape[0] / IMG_SIZE, frame.shape[1] / IMG_SIZE, frame.shape[0] / IMG_SIZE)
            for box, cls in zip(scaled_boxes, results[0].boxes.cls):
                if int(cls) in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box)
                    label = model.names[int(cls)]
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        current_time = frame_count / fps
        for i, (box, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls)):
            if int(cls) not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = box.cpu().numpy() * (frame.shape[1] / IMG_SIZE, frame.shape[0] / IMG_SIZE, frame.shape[1] / IMG_SIZE, frame.shape[0] / IMG_SIZE)
            class_name = model.names[int(cls)].lower()
            vehicle_count = 2 if class_name == 'truck' else 1
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            vehicle_id = f"v{frame_count}_{i}"

            min_dist = float('inf')
            matched_id = None
            for vid, (vc, vt, _) in tracked_vehicles.items():
                dist = np.sqrt((centroid[0] - vc[0])**2 + (centroid[1] - vc[1])**2)
                if dist < min_dist and dist < 50:
                    min_dist = dist
                    matched_id = vid

            if matched_id:
                vehicle_id = matched_id
                tracked_vehicles[vehicle_id] = (centroid, current_time, centroid[0] < LANE_THRESHOLD)
            else:
                tracked_vehicles[vehicle_id] = (centroid, current_time, centroid[0] < LANE_THRESHOLD)
                unique_vehicle_ids.add(vehicle_id)

            current_detections.append((vehicle_id, centroid, vehicle_count, class_name))
            current_centroids[vehicle_id] = centroid

            if vehicle_id in vehicle_centroids and len(vehicle_centroids[vehicle_id]) > 1:
                prev_centroid = vehicle_centroids[vehicle_id][-1]
                displacement = np.sqrt((centroid[0] - prev_centroid[0])**2 + (centroid[1] - prev_centroid[1])**2)
                speed = displacement * fps
                if centroid[0] < LANE_THRESHOLD:
                    left_speeds.append(speed)
                else:
                    right_speeds.append(speed)

            vehicle_centroids[vehicle_id].append(centroid)

        expired = [vid for vid, (_, t, _) in tracked_vehicles.items() if current_time - t > TRACKING_EXPIRY]
        for vid in expired:
            tracked_vehicles.pop(vid, None)
            vehicle_centroids.pop(vid, None)

        for vid, _, count, _ in current_detections:
            if tracked_vehicles[vid][2]:
                vehicles_in_left_lane += count
            else:
                vehicles_in_right_lane += count

        print(f"Frame {frame_count}: Left={vehicles_in_left_lane}, Right={vehicles_in_right_lane}, Classes={[c for _, _, _, c in current_detections]}")
    else:
        if prev_metrics:
            vehicles_in_left_lane = prev_metrics['left_vehicles']
            vehicles_in_right_lane = prev_metrics['right_vehicles']
            left_speeds = [prev_metrics['left_speed']] if prev_metrics['left_speed'] > 0 else []
            right_speeds = [prev_metrics['right_speed']] if prev_metrics['right_speed'] > 0 else []

    traffic_intensity_left = "Heavy" if vehicles_in_left_lane >= HEAVY_TRAFFIC_THRESHOLD else "Smooth"
    traffic_intensity_right = "Heavy" if vehicles_in_right_lane >= HEAVY_TRAFFIC_THRESHOLD else "Smooth"
    left_density = (vehicles_in_left_lane / left_lane_area) * 1000 if left_lane_area > 0 else 0.0
    right_density = (vehicles_in_right_lane / right_lane_area) * 1000 if right_lane_area > 0 else 0.0
    left_speed = np.mean(left_speeds) if left_speeds else 0.0
    right_speed = np.mean(right_speeds) if right_speeds else 0.0
    total_vehicles = vehicles_in_left_lane + vehicles_in_right_lane

    # Debug density values
    print(f"Density - Left: {left_density:.6f}, Right: {right_density:.6f}")

    # Generate heatmap
    processed_frame, heatmap = create_heatmap(processed_frame, left_density, right_density)
    with heatmap_lock:
        success, buffer = cv2.imencode('.jpg', heatmap)
        if success:
            latest_heatmap = buffer.tobytes()
        else:
            placeholder = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillPoly(placeholder, [VERTICES1], GREEN_COLOR)
            cv2.fillPoly(placeholder, [VERTICES2], GREEN_COLOR)
            _, buffer = cv2.imencode('.jpg', placeholder)
            latest_heatmap = buffer.tobytes()

    with metrics_lock:
        global_metrics['peak_vehicle_count'] = max(global_metrics['peak_vehicle_count'], len(unique_vehicle_ids))
        global_metrics.update({
            'left_vehicles': vehicles_in_left_lane,
            'right_vehicles': vehicles_in_right_lane,
            'left_intensity': traffic_intensity_left,
            'right_intensity': traffic_intensity_right,
            'left_density': round(left_density, 2),
            'right_density': round(right_density, 2),
            'left_speed': round(left_speed, 2),
            'right_speed': round(right_speed, 2)
        })

    if is_red_light:
        if (frame_count - red_light_start_frame) / fps >= red_light_duration:
            is_red_light = False
            is_yellow_light = False
        traffic_light_state = "Red"
    elif is_yellow_light:
        if (frame_count - yellow_start_frame) / fps >= 0.5:
            is_yellow_light = False
            is_red_light = True
            red_light_start_frame = frame_count
        traffic_light_state = "Yellow"
    elif vehicles_in_left_lane >= HEAVY_TRAFFIC_THRESHOLD and vehicles_in_right_lane >= HEAVY_TRAFFIC_THRESHOLD:
        is_yellow_light = True
        yellow_start_frame = frame_count
        traffic_light_state = "Yellow"
    else:
        traffic_light_state = "Green"
        is_yellow_light = False

    traffic_light_x = (LANE_THRESHOLD + 678) // 2
    traffic_light_y = frame.shape[0] // 2
    draw_traffic_light(processed_frame, traffic_light_state, traffic_light_x, traffic_light_y)

    # Draw lane lines on all frames
    cv2.polylines(processed_frame, [VERTICES1], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(processed_frame, [VERTICES2], isClosed=True, color=(255, 0, 0), thickness=2)

    return processed_frame

def process_frames(cap, frame_queue, fps):
    frame_count = 0
    prev_metrics = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame, frame_count, fps, prev_metrics)
        if processed_frame is not None and processed_frame.size > 0:
            success, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if success:
                frame_queue.put(buffer.tobytes())
        with metrics_lock:
            prev_metrics = global_metrics.copy()
        frame_count += 1
    cap.release()
    frame_queue.put(None)

def stream_video():
    video_path = os.path.join(UPLOAD_DIR, 'video.mp4')
    if not os.path.exists(video_path):
        return Response("Video file not found.", status=404)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return Response("Could not open video file.", status=500)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    frame_queue = queue.Queue(maxsize=10)

    threading.Thread(target=process_frames, args=(cap, frame_queue, fps), daemon=True).start()

    while True:
        frame_bytes = frame_queue.get()
        if frame_bytes is None:
            break
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/heatmap_feed')
def heatmap_feed():
    with heatmap_lock:
        if latest_heatmap is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.fillPoly(placeholder, [VERTICES1], GREEN_COLOR)
            cv2.fillPoly(placeholder, [VERTICES2], GREEN_COLOR)
            _, buffer = cv2.imencode('.jpg', placeholder)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        return Response(latest_heatmap, mimetype='image/jpeg')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files.get('video')
        if video:
            video_path = os.path.join(UPLOAD_DIR, 'video.mp4')
            video.save(video_path)
            reset_state()
            return redirect(url_for('detection'))
        return Response("No video file uploaded.", status=400)
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(stream_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def metrics():
    with metrics_lock:
        return jsonify(global_metrics)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    reset_state()  # Initial reset
    app.run(debug=True)