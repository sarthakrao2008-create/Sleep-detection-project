import cv2
import mediapipe as mp
import numpy as np
import winsound
import threading
import time
from collections import deque

# =========================================================
# CONFIG
# =========================================================
CALIBRATION_TIME = 3          # seconds to learn normal EAR
EAR_MARGIN = 0.75             # % of baseline for closed eye
DROWSY_FRAMES = 18
RESET_FRAMES = 8
HAZARD_DELAY = 3

PERCLOS_WINDOW = 60           # frames window
PERCLOS_THRESHOLD = 0.4       # 40% eye closure

# =========================================================
# GLOBAL STATE
# =========================================================
alarm_on = False
drowsy_counter = 0
awake_counter = 0
drowsy_start_time = None
vehicle_speed = 100

ear_history = deque(maxlen=5)
perclos_buffer = deque(maxlen=PERCLOS_WINDOW)

# Adaptive EAR
ear_baseline = None
calibrating = True
calibration_start = time.time()

# =========================================================
# ALARM
# =========================================================
def sound_alarm():
    global alarm_on
    while alarm_on:
        winsound.Beep(1200, 200)
        time.sleep(0.1)

# =========================================================
# METRICS
# =========================================================
def eye_aspect_ratio(pts):
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    if h == 0:
        return 0
    return (v1 + v2) / (2.0 * h)

def draw_eye_region(frame, pts, color):
    x_min, x_max = np.min(pts[:,0]), np.max(pts[:,0])
    y_min, y_max = np.min(pts[:,1]), np.max(pts[:,1])

    pad_x = int((x_max - x_min) * 0.3)
    pad_y = int((y_max - y_min) * 0.5)

    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    h, w = frame.shape[:2]
    x_min, y_min = max(0,x_min), max(0,y_min)
    x_max, y_max = min(w,x_max), min(h,y_max)

    center = ((x_min+x_max)//2, (y_min+y_max)//2)
    axes = ((x_max-x_min)//2, (y_max-y_min)//2)

    cv2.ellipse(frame, center, axes, 0, 0, 360, color, 2)

# =========================================================
# MEDIAPIPE
# =========================================================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

# =========================================================
# CAMERA
# =========================================================
cap = cv2.VideoCapture(0)
print("🚗 Driver Monitoring Started")

# =========================================================
# LOOP
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "AWAKE"
    color = (0, 200, 0)
    hazards = False

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        coords = [(int(l.x*w), int(l.y*h)) for l in face.landmark]

        left_eye = np.array([coords[i] for i in LEFT_EYE])
        right_eye = np.array([coords[i] for i in RIGHT_EYE])

        ear_raw = (eye_aspect_ratio(left_eye) +
                   eye_aspect_ratio(right_eye)) / 2.0

        # Smooth EAR
        ear_history.append(ear_raw)
        ear = np.mean(ear_history)

        # =====================================================
        # 🔥 ADAPTIVE CALIBRATION
        # =====================================================
        if calibrating:
            if ear_baseline is None:
                ear_baseline = ear
            else:
                ear_baseline = 0.9 * ear_baseline + 0.1 * ear

            if time.time() - calibration_start > CALIBRATION_TIME:
                calibrating = False

            cv2.putText(frame, "CALIBRATING...",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255,200,0), 2)
            cv2.imshow("Driver Monitoring System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Dynamic threshold
        dynamic_threshold = ear_baseline * EAR_MARGIN

        eyes_closed = ear < dynamic_threshold
        perclos_buffer.append(1 if eyes_closed else 0)
        perclos = sum(perclos_buffer) / len(perclos_buffer)

        # =====================================================
        # 🧠 DROWSINESS LOGIC (HYBRID)
        # =====================================================
        if eyes_closed or perclos > PERCLOS_THRESHOLD:
            drowsy_counter += 1
            awake_counter = 0

            if drowsy_counter >= DROWSY_FRAMES:
                status = "DROWSY"
                color = (0, 0, 255)

                if drowsy_start_time is None:
                    drowsy_start_time = time.time()

                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=sound_alarm, daemon=True).start()
        else:
            awake_counter += 1
            if awake_counter >= RESET_FRAMES:
                drowsy_counter = 0
                alarm_on = False
                drowsy_start_time = None
                vehicle_speed = 100

        # Hazard logic
        if drowsy_start_time and (time.time() - drowsy_start_time > HAZARD_DELAY):
            hazards = True
            vehicle_speed = max(0, vehicle_speed - 0.6)

        # Eye visuals
        eye_color = (0,0,255) if status=="DROWSY" else (0,255,255)
        draw_eye_region(frame, left_eye, eye_color)
        draw_eye_region(frame, right_eye, eye_color)

    # =========================================================
    # CLEAN UI
    # =========================================================
    cv2.putText(frame, f"STATUS: {status}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2)

    # YELLOW SPEED
    cv2.putText(frame, f"SPEED: {int(vehicle_speed)} km/h",
                (20,75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (0,255,255), 2)

    # Hazard lights
    if hazards:
        blink = int(time.time()*4) % 2
        if blink == 0:
            cv2.rectangle(frame,(20,h-60),(120,h-20),(0,165,255),-1)
            cv2.rectangle(frame,(w-120,h-60),(w-20,h-20),(0,165,255),-1)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()