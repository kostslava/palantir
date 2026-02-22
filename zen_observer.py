import cv2
import mediapipe as mp
import time
import urllib.request
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from google import genai
from google.genai import types

# --- 1. CONFIGURATION & INITIALIZATION ---
GEMINI_API_KEY = "AIzaSyDCcs5gppsewgUMuEQJSNtqX_3oj_0V0Y8"
client = genai.Client(api_key=GEMINI_API_KEY)

# Download model files on first run
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

MODEL_URLS = {
    FACE_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    HAND_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

for path, url in MODEL_URLS.items():
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print(f"  Done.")

# Initialize Face Landmarker (Tasks API — replaces mp.solutions.holistic)
face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        num_faces=4,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

# Initialize Hand Landmarker
hand_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=4,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

# Hand skeleton connections (21-point model)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm knuckles
]


def draw_face_landmarks(frame, face_landmarks_list):
    h, w = frame.shape[:2]
    for face_landmarks in face_landmarks_list:
        for lm in face_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (200, 200, 200), -1)


def draw_hand_landmarks(frame, hand_landmarks_list):
    h, w = frame.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, pts[start], pts[end], (0, 255, 128), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 200, 255), -1)

# --- 2. THE CALM SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are a calm social observer. Analyze the expressions and hand gestures of all people visible.
Focus on:
- Subtle facial emotions (relaxed, smirk, thoughtful).
- Hand positions (resting, gesturing, pointing).
- The overall 'vibe' of the group.

Output ONLY valid JSON in this format:
{
  "people": [{"id": int, "expression": str, "hands": str}],
  "social_vibe": str
}
"""


def get_ai_analysis(frame):
    """Sends a single frame to Gemini for deep social analysis."""
    _, buffer = cv2.imencode(".jpg", frame)
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(
                    data=buffer.tobytes(), mime_type="image/jpeg"
                ),
                "Analyze the faces and hands in this scene.",
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                media_resolution="media_resolution_low",  # Keep it fast
            ),
        )
        return response.text
    except Exception as e:
        return f"AI Error: {e}"


# --- 3. MAIN LOOP ---
cap = cv2.VideoCapture(0)
last_ai_ping = 0
ai_result = "Waiting for first scan..."

print("System Online. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # A. MEDIAPIPE Tasks API (runs every frame — 30fps)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    face_result = face_landmarker.detect(mp_image)
    hand_result = hand_landmarker.detect(mp_image)

    draw_face_landmarks(frame, face_result.face_landmarks)
    draw_hand_landmarks(frame, hand_result.hand_landmarks)

    # B. GEMINI AI (Runs every 10 seconds)
    current_time = time.time()
    if current_time - last_ai_ping > 10:
        ai_result = get_ai_analysis(frame)
        print(f"[{time.strftime('%H:%M:%S')}] AI Update: {ai_result}")
        last_ai_ping = current_time

    # C. UI OVERLAY
    cv2.putText(
        frame,
        "AI ANALYST (Every 10s):",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    # Show the first 60 chars of result on screen for vibe
    cv2.putText(
        frame,
        f"{ai_result[:60]}...",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    cv2.imshow("Zen Observer: MediaPipe + Gemini", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()
hand_landmarker.close()
