import face_recognition
import cv2
import numpy as np
import os
import time
from scipy.spatial import distance as dist
import dlib
import csv
from datetime import datetime
import requests

# --- Configuration ---
script_dir = os.path.dirname(__file__)
AUTHORIZED_FACES_DIR = os.path.join(script_dir, "authorized_faces")
SHAPE_PREDICTOR_PATH = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")
ACCESS_LOG_FILE = os.path.join(script_dir, "access_log.csv")

FACE_RECOGNITION_TOLERANCE = 0.55

# --- Telegram Bot Configuration ---
# !!! IMPORTANT: REPLACE WITH YOUR ACTUAL BOT TOKEN AND CHAT ID !!!
TELEGRAM_BOT_TOKEN = 'Enter BOT Token'   # <--- PASTE YOUR BOT TOKEN HERE
TELEGRAM_CHAT_ID = 'Enter Chat_ID'     # <--- PASTE YOUR CHAT ID HERE

# --- Lock Re-activation Cooldown ---
COOLDOWN_PERIOD_SECONDS = 4.0

# --- Liveness Detection Parameters ---
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
LIVENESS_CHECK_DURATION_SEC = 5
LIVENESS_CHECK_BLINK_COUNT = 1

# --- Helper function to send Telegram message ---
def send_telegram_message(message):
    """Sends a message to the configured Telegram chat."""
    if TELEGRAM_BOT_TOKEN == 'YOUR_TELEGRAM_BOT_TOKEN_HERE' or TELEGRAM_CHAT_ID == 'YOUR_TELEGRAM_CHAT_ID_HERE':
        print("WARNING: Telegram bot not configured. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the script.")
        return

    telegram_api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(telegram_api_url, data=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not send Telegram message: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while sending Telegram message: {e}")

# --- Helper function to log access events ---
def log_access(person_name):
    """Logs a successful access event to a CSV file and sends a Telegram message."""
    now = datetime.now()
    log_time = now.strftime("%H:%M:%S")
    log_date = now.strftime("%Y-%m-%d")

    log_entry = [person_name, log_time, log_date]

    file_exists = os.path.exists(ACCESS_LOG_FILE)

    try:
        with open(ACCESS_LOG_FILE, 'a', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            if not file_exists:
                log_writer.writerow(['Name', 'Time', 'Date'])
            log_writer.writerow(log_entry)
        print(f"LOG: Access recorded for '{person_name}' at {log_time} on {log_date}.")

        telegram_message = f"🚪 *ACCESS GRANTED* 🚪\n\n*Name:* {person_name}\n*Time:* {log_time}\n*Date:* {log_date}"
        send_telegram_message(telegram_message)

    except IOError as e:
        print(f"ERROR: Could not write to access log file '{ACCESS_LOG_FILE}': {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while logging: {e}")

# --- Load all known faces from the 'authorized_faces' directory ---
known_face_encodings = []
known_face_names = []

print(f"Loading authorized faces from: {AUTHORIZED_FACES_DIR}")
if not os.path.exists(AUTHORIZED_FACES_DIR):
    print(f"ERROR: '{AUTHORIZED_FACES_DIR}' directory not found.")
    print("Please create this folder and place authorized person images inside it.")
    exit()

for filename in os.listdir(AUTHORIZED_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        person_name = os.path.splitext(filename)[0]
        person_name = person_name.replace("_", " ").title()
        image_path = os.path.join(AUTHORIZED_FACES_DIR, filename)

        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)
                print(f"Loaded '{person_name}' from '{filename}'.")
            else:
                print(f"WARNING: No face found in '{filename}'. Skipping this image.")
        except Exception as e:
            print(f"ERROR: Could not process '{filename}': {e}")

if not known_face_encodings:
    print("ERROR: No authorized faces loaded. Please ensure 'authorized_faces' folder contains images with clear faces.")
    exit()
print(f"Successfully loaded {len(known_face_encodings)} authorized faces.")

# --- Dlib Face Landmark Predictor ---
print(f"Loading Dlib shape predictor from: {SHAPE_PREDICTOR_PATH}")
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    print("Dlib shape predictor loaded successfully.")
except RuntimeError as e:
    print(f"ERROR loading Dlib shape predictor: {e}")
    print(f"Please ensure '{SHAPE_PREDICTOR_PATH}' is in the correct path and readable.")
    print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

# --- Eye Aspect Ratio (EAR) Calculation ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# --- State variables for Liveness Detection ---
blink_counter = 0
consecutive_frames_eye_closed = 0

# --- System States ---
SYSTEM_STATE_IDLE = 0
SYSTEM_STATE_LIVENESS_CHECK = 1
SYSTEM_STATE_ACCESS_GRANTED = 2
SYSTEM_STATE_ACCESS_DENIED = 3

current_system_state = SYSTEM_STATE_IDLE
liveness_check_start_time = 0
last_access_granted_time = 0
recognized_person_name = "Unknown"

# --- Webcam Initialization ---
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("ERROR: Could not open video stream. Make sure webcam is connected and not in use.")
    print("Also ensure no other applications are using the camera.")
    exit()

print(f"\nWebcam started successfully. Press 'q' to quit.")
print("Liveness detection is active. Please blink when prompted (after a face is detected).")

# Removed initial Arduino command
# send_arduino_command('N') # This line is now effectively 'pass' due to function redefinition

# --- Main loop for live recognition and liveness ---
frame_count = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("ERROR: Failed to grab frame from webcam. Exiting.")
        break

    frame_count += 1

    FRAME_SCALE_FACTOR = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE_FACTOR, fy=FRAME_SCALE_FACTOR)

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    current_time = time.time()

    face_locations_dlib = detector(gray_frame, 0)

    # --- State Machine Logic ---
    if current_system_state == SYSTEM_STATE_IDLE:
        # Removed Arduino command
        cv2.putText(frame, "STATUS: Waiting for Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if len(face_locations_dlib) == 1:
            print(f"\n--- Single face detected. Initiating liveness check. ---")
            print(f"Current EYE_AR_THRESH: {EYE_AR_THRESH}")
            print(f"Current EYE_AR_CONSEC_FRAMES: {EYE_AR_CONSEC_FRAMES}")
            print(f"Required Blinks: {LIVENESS_CHECK_BLINK_COUNT}")
            blink_counter = 0
            consecutive_frames_eye_closed = 0
            current_system_state = SYSTEM_STATE_LIVENESS_CHECK
            liveness_check_start_time = current_time
            recognized_person_name = "Unknown"
            # Removed Arduino command
            # send_arduino_command('N')

        elif len(face_locations_dlib) > 1:
            print(f"\n--- Multiple faces detected ({len(face_locations_dlib)}). Denying access. ---")
            cv2.putText(frame, "MULTIPLE FACES DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Removed Arduino command
            # send_arduino_command('U')

    elif current_system_state == SYSTEM_STATE_LIVENESS_CHECK:
        cv2.putText(frame, "LIVENESS CHECK: Please BLINK!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        remaining_time = max(0, LIVENESS_CHECK_DURATION_SEC - (current_time - liveness_check_start_time))
        cv2.putText(frame, f"Time left: {remaining_time:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if current_time - liveness_check_start_time > LIVENESS_CHECK_DURATION_SEC:
            print("\nLiveness check failed: Time expired.")
            current_system_state = SYSTEM_STATE_ACCESS_DENIED
            # Removed Arduino command
            # send_arduino_command('U')

        elif len(face_locations_dlib) == 1:
            shape = predictor(gray_frame, face_locations_dlib[0])
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)])

            leftEye = landmarks[lStart:lEnd]
            rightEye = landmarks[rStart:rEnd]

            scaled_landmarks = landmarks * int(1.0 / FRAME_SCALE_FACTOR)

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye bounding boxes on the original frame using scaled landmarks
            leftEyeHull = cv2.convexHull(scaled_landmarks[lStart:lEnd].astype(np.int32))
            rightEyeHull = cv2.convexHull(scaled_landmarks[rStart:rEnd].astype(np.int32))
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            cv2.putText(frame, f"EAR: {ear:.2f}", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if ear < EYE_AR_THRESH:
                consecutive_frames_eye_closed += 1
            else:
                if consecutive_frames_eye_closed >= EYE_AR_CONSEC_FRAMES:
                    blink_counter += 1
                    print(f"\nBlink detected! Total blinks: {blink_counter}")
                consecutive_frames_eye_closed = 0

            cv2.putText(frame, f"Blinks: {blink_counter}/{LIVENESS_CHECK_BLINK_COUNT}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if blink_counter >= LIVENESS_CHECK_BLINK_COUNT:
                print("\n--- Liveness check SUCCESS! Proceeding to Face Recognition. ---")

                face_locations_fr = [(face_locations_dlib[0].top(), face_locations_dlib[0].right(), face_locations_dlib[0].bottom(), face_locations_dlib[0].left())]

                face_encodings_recognition = face_recognition.face_encodings(rgb_small_frame, face_locations_fr)

                recognized_known_face_found = False
                if face_encodings_recognition:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encodings_recognition[0], tolerance=FACE_RECOGNITION_TOLERANCE)

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encodings_recognition[0])
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        recognized_known_face_found = True
                        recognized_person_name = known_face_names[best_match_index]
                        print(f"Recognized: {recognized_person_name}")
                    else:
                        recognized_person_name = "Unknown"

                if recognized_known_face_found:
                    if current_time - last_access_granted_time > COOLDOWN_PERIOD_SECONDS:
                        # Removed Arduino command
                        # send_arduino_command('K')
                        current_system_state = SYSTEM_STATE_ACCESS_GRANTED
                        last_access_granted_time = current_time
                        print(f"Access granted to {recognized_person_name} after liveness.")
                        log_access(recognized_person_name)
                    else:
                        cv2.putText(frame, "COOLDOWN ACTIVE!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        current_system_state = SYSTEM_STATE_ACCESS_GRANTED
                        print(f"Known person ({recognized_person_name}) detected but cooldown active. Remaining: {COOLDOWN_PERIOD_SECONDS - (current_time - last_access_granted_time):.1f}s")
                else:
                    print(f"Liveness success, but face is UNKNOWN or not authorized.")
                    current_system_state = SYSTEM_STATE_ACCESS_DENIED
                    # Removed Arduino command
                    # send_arduino_command('U')
        else: # Face lost or multiple faces during liveness check
            print("\nLiveness check failed: Face lost or multiple faces detected during check.")
            current_system_state = SYSTEM_STATE_ACCESS_DENIED
            # Removed Arduino command
            # send_arduino_command('U')

    elif current_system_state == SYSTEM_STATE_ACCESS_GRANTED:
        cv2.putText(frame, f"ACCESS GRANTED: {recognized_person_name}!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if current_time - last_access_granted_time > COOLDOWN_PERIOD_SECONDS + 1:
            print(f"\nAccess granted cooldown expired for {recognized_person_name}. Returning to idle.")
            current_system_state = SYSTEM_STATE_IDLE
            recognized_person_name = "Unknown"
            # Removed Arduino command
            # send_arduino_command('N')

    elif current_system_state == SYSTEM_STATE_ACCESS_DENIED:
        cv2.putText(frame, f"STATUS: ACCESS DENIED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Removed Arduino command
        # send_arduino_command('U')
        if current_time - liveness_check_start_time > LIVENESS_CHECK_DURATION_SEC + 2:
            print("\nAccess denied state timeout. Returning to idle.")
            current_system_state = SYSTEM_STATE_IDLE
            recognized_person_name = "Unknown"
            # Removed Arduino command
            # send_arduino_command('N')

    # --- Drawing Face Bounding Boxes ---
    for dlib_rect in face_locations_dlib:
        top = dlib_rect.top() * int(1.0 / FRAME_SCALE_FACTOR)
        right = dlib_rect.right() * int(1.0 / FRAME_SCALE_FACTOR)
        bottom = dlib_rect.bottom() * int(1.0 / FRAME_SCALE_FACTOR)
        left = dlib_rect.left() * int(1.0 / FRAME_SCALE_FACTOR)

        name_to_display = "Face Detected"
        color = (0, 255, 255)

        if current_system_state == SYSTEM_STATE_LIVENESS_CHECK:
            name_to_display = "Blink to Verify!"
        elif current_system_state == SYSTEM_STATE_ACCESS_GRANTED:
            color = (0, 255, 0)
            name_to_display = recognized_person_name
        elif current_system_state == SYSTEM_STATE_ACCESS_DENIED:
            color = (0, 0, 255)
            name_to_display = "Access Denied"
        elif current_system_state == SYSTEM_STATE_IDLE and len(face_locations_dlib) > 1:
            name_to_display = "Multiple Faces!"
            color = (0, 165, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name_to_display, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n'q' pressed. Exiting...")
        break

# --- Cleanup ---
# Removed final Arduino command
# send_arduino_command('N')
time.sleep(0.5)
video_capture.release()
cv2.destroyAllWindows()
print("\nProgram ended.")
