import cv2
import mediapipe as mp
import math
import pyttsx3
import threading
import time
from collections import deque
from flask import Flask, Response, jsonify, render_template

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

engine = pyttsx3.init()
engine.setProperty('rate', 160)
audio_lock = threading.Lock()

def speak_letter_async(letter):
    """Speaks the letter in a separate thread to prevent webcam/stream freezing."""
    def target():
        with audio_lock:
            engine.say(letter)
            engine.runAndWait()
    threading.Thread(target=target, daemon=True).start()

current_word = ""               # Holds the typed text string
stable_letter = "Searching..."  # Current tracked letter match
stable_since = None             # Timestamp when the letter first became stable
REQUIRED_STABLE_TIME = 2.0      # Time window in seconds to type a letter

def dist_2d(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

wrist_x_history = deque(maxlen=12)
pinky_history = deque(maxlen=20)
index_history = deque(maxlen=25) 
j_cooldown = 0
n_cooldown = 0
g_cooldown = 0
z_cooldown = 0
s_cooldown = 0 

app = Flask(__name__, static_folder='templates', static_url_path='')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Set buffer size low for instant frame delivery

def generate_lsc_frames():
    global j_cooldown, n_cooldown, g_cooldown, z_cooldown, s_cooldown
    global current_word, stable_letter, stable_since

    while cap.isOpened():
        cap.grab()
        success, img = cap.retrieve()
        if not success:
            continue
        
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        label = "Searching..."

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                lm = hand_lms.landmark

                # Update gesture deques
                wrist_x_history.append(lm[0].x)
                pinky_history.append((lm[20].x, lm[20].y))
                index_history.append((lm[8].x, lm[8].y))

                if j_cooldown > 0: j_cooldown -= 1
                if n_cooldown > 0: n_cooldown -= 1 
                if g_cooldown > 0: g_cooldown -= 1
                if z_cooldown > 0: z_cooldown -= 1
                if s_cooldown > 0: s_cooldown -= 1

                mov_x = 0
                if len(wrist_x_history) == wrist_x_history.maxlen:
                    mov_x = abs(wrist_x_history[-1] - wrist_x_history[0])

                def get_ext_ratio(tip_idx):
                    knuckle_dist = dist_2d(lm[tip_idx-2], lm[0])
                    tip_dist = dist_2d(lm[tip_idx], lm[0])
                    return tip_dist / knuckle_dist if knuckle_dist != 0 else 0

                i_ext = get_ext_ratio(8) > 1.2
                m_ext = get_ext_ratio(12) > 1.2
                r_ext = get_ext_ratio(16) > 1.2
                p_ext = get_ext_ratio(20) > 1.2

                hand_scale = dist_2d(lm[5], lm[0]) if dist_2d(lm[5], lm[0]) != 0 else 1.0

                # Key Metric Ratios
                thumb_to_index_tip = dist_2d(lm[4], lm[8]) / hand_scale
                thumb_to_mid_tip = dist_2d(lm[4], lm[12]) / hand_scale
                thumb_to_point_1 = dist_2d(lm[4], lm[1]) / hand_scale
                thumb_to_knuckle_5 = dist_2d(lm[4], lm[5]) / hand_scale
                thumb_to_knuckle_6 = dist_2d(lm[4], lm[6]) / hand_scale
                thumb_to_knuckle_9 = dist_2d(lm[4], lm[9]) / hand_scale
                thumb_to_knuckle_10 = dist_2d(lm[4], lm[10]) / hand_scale
                thumb_to_knuckle_11 = dist_2d(lm[4], lm[11]) / hand_scale
                thumb_to_knuckle_14 = dist_2d(lm[4], lm[14]) / hand_scale
                thumb_to_knuckle_15 = dist_2d(lm[4], lm[15]) / hand_scale
                thumb_to_knuckle_18 = dist_2d(lm[4], lm[18]) / hand_scale
                thumb_to_knuckle_19 = dist_2d(lm[4], lm[19]) / hand_scale
                thumb_to_point_0 = dist_2d(lm[4], lm[0]) / hand_scale

                i_tip_to_thumb = dist_2d(lm[8], lm[4]) / hand_scale
                m_tip_to_thumb = dist_2d(lm[12], lm[4]) / hand_scale
                r_tip_to_thumb = dist_2d(lm[16], lm[4]) / hand_scale
                p_tip_to_thumb = dist_2d(lm[20], lm[4]) / hand_scale

                i_tip_to_mid_tip = dist_2d(lm[8], lm[12]) / hand_scale
                index_to_point_2 = dist_2d(lm[8], lm[2]) / hand_scale
                index_tip_to_pip = dist_2d(lm[8], lm[6]) / hand_scale
                index_tip_to_mcp = dist_2d(lm[8], lm[5]) / hand_scale
                i_hook = (not i_ext) and index_tip_to_pip < 0.25 and index_tip_to_mcp < 0.55

                mid_to_index_knuckle_6 = dist_2d(lm[9], lm[6]) / hand_scale
                mid_to_index_kuckle_7 = dist_2d(lm[9], lm[7]) / hand_scale
                middle_to_point_3 = dist_2d(lm[12], lm[3]) / hand_scale

                ring_to_point_1 = dist_2d(lm[16], lm[1]) / hand_scale
                ring_to_point_0 = dist_2d(lm[16], lm[0]) / hand_scale

                is_s_motion = False
                if i_ext and not m_ext and not r_ext and not p_ext:
                    if len(index_history) == index_history.maxlen:
                        x_coords = [p[0] for p in index_history]
                        y_coords = [p[1] for p in index_history]
                        total_dx = sum(abs(x_coords[i] - x_coords[i-1]) for i in range(1, len(x_coords)))
                        total_dy = sum(abs(y_coords[i] - y_coords[i-1]) for i in range(1, len(y_coords)))
                        if total_dx > 0.06 and total_dy > 0.06:
                            is_s_motion = True
                            s_cooldown = 20

                # --- EXTRACTED LSC SIGN ALGORITHMS ---
                if (i_ext and m_ext and r_ext and p_ext and (thumb_to_index_tip < 0.25 or thumb_to_mid_tip < 0.25)):
                    label = "Q"
                elif i_ext and m_ext and r_ext and p_ext and thumb_to_point_0 < 0.8:
                    label = "B"
                elif (i_ext or i_hook) and not m_ext and not r_ext and not p_ext and (abs(lm[6].x - lm[5].x) > abs(lm[6].y - lm[5].y) or g_cooldown > 0):
                    label = "G"
                    if abs(lm[6].x - lm[5].x) > abs(lm[6].y - lm[5].y): g_cooldown = 15
                elif (i_ext and not m_ext and not r_ext and not p_ext) and (is_s_motion or s_cooldown > 0):
                    label = "S"
                elif (i_ext and not m_ext and not r_ext and not p_ext and index_tip_to_mcp > 0.65 and index_tip_to_pip > 0.35 and (m_tip_to_thumb < 0.4 and r_tip_to_thumb < 0.4 and p_tip_to_thumb < 0.4)):
                    label = "D"
                elif (p_ext and m_ext and r_ext and (i_tip_to_thumb < 0.5)):
                    label = "T"
                elif (not p_ext and not r_ext and i_ext and (mid_to_index_knuckle_6 < 0.4 or mid_to_index_kuckle_7 < 0.4)):
                    label = "P"
                elif i_ext and m_ext and not r_ext and not p_ext:
                    is_z_motion = False
                    if len(index_history) == index_history.maxlen:
                        x_coords = [p[0] for p in index_history]
                        y_coords = [p[1] for p in index_history]
                        total_dx = sum(abs(x_coords[i] - x_coords[i-1]) for i in range(1, len(x_coords)))
                        net_dx = abs(x_coords[-1] - x_coords[0])
                        vertical_travel = max(y_coords) - min(y_coords)
                        if total_dx > 0.08 and total_dx > 1.8 * (net_dx if net_dx > 0 else 0.001) and vertical_travel > 0.05:
                            is_z_motion = True
                            z_cooldown = 22

                    is_horizontal = abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y)
                    if is_horizontal or mov_x > 0.04: label = "H"
                    elif i_tip_to_mid_tip < 0.35: label = "R"
                    elif (thumb_to_knuckle_6 < 0.4 or thumb_to_knuckle_10 < 0.4): label = "K"
                    elif (i_tip_to_mid_tip < 0.6) and (is_z_motion or z_cooldown > 0): label = "Z"
                    elif (thumb_to_knuckle_14 < 0.3 or thumb_to_knuckle_15 < 0.3): label = "V"
                elif i_ext and p_ext and not m_ext and not r_ext: label = "U"
                elif r_ext and m_ext and i_ext and not p_ext: label = "W"
                elif p_ext and thumb_to_knuckle_5 > 0.5 and not i_ext and not m_ext and not r_ext: label = "Y"
                elif p_ext and not i_ext and not m_ext and not r_ext:
                    is_j_motion = False 
                    if len(pinky_history) == pinky_history.maxlen:
                        y_coords = [p[1] for p in pinky_history]
                        x_coords = [p[0] for p in pinky_history]
                        max_y_idx = y_coords.index(max(y_coords) if y_coords else 0)
                        if (max(y_coords) - min(y_coords)) > 0.05 and (max(x_coords) - min(x_coords)) > 0.03:
                            if 4 < max_y_idx < 16: is_j_motion = True
                    if is_j_motion or j_cooldown > 0:
                        label = "J"
                        if is_j_motion: j_cooldown = 18
                    else: label = "I"
                elif (not m_ext and not r_ext and not p_ext and i_ext and (thumb_to_knuckle_5 < 0.35 or thumb_to_knuckle_6 < 0.35)):
                    label = "F"
                elif (i_hook and not m_ext and not r_ext and not p_ext and (thumb_to_knuckle_18 < 0.8 or thumb_to_knuckle_19 < 0.8 or thumb_to_knuckle_10 < 0.5 or thumb_to_knuckle_14 < 0.5) and thumb_to_knuckle_9 < 0.5 and thumb_to_point_0 > 0.45 and thumb_to_index_tip > 0.25):
                    label = "X"
                elif not i_ext and not m_ext and not r_ext and not p_ext:
                    i_tip_to_mcp = dist_2d(lm[8], lm[5]) / hand_scale
                    m_tip_to_mcp = dist_2d(lm[12], lm[9]) / hand_scale
                    r_tip_to_mcp = dist_2d(lm[16], lm[13]) / hand_scale
                    p_tip_to_mcp = dist_2d(lm[20], lm[17]) / hand_scale
                    if thumb_to_index_tip < 0.25 or thumb_to_mid_tip < 0.2: label = "O"
                    elif (i_tip_to_mcp < 0.5 and m_tip_to_mcp < 0.5 and r_tip_to_mcp < 0.5 and p_tip_to_mcp < 0.45 and (thumb_to_point_0 < 0.8 or thumb_to_point_1 < 1) and thumb_to_knuckle_5 > 0.45):
                        label = "E"
                    elif (index_to_point_2 < 0.35 and middle_to_point_3 < 0.35):
                        if (ring_to_point_1 < 0.45 or ring_to_point_0 < 0.45): label = "M"
                        else:
                            if mov_x > 0.04 or n_cooldown > 0:
                                label = "Ñ"
                                if mov_x > 0.04: n_cooldown = 15
                            else: label = "N"
                    elif (thumb_to_knuckle_5 < 0.35 or thumb_to_knuckle_6 < 0.35) and not i_hook: label = "A"
                    elif thumb_to_index_tip > 0.45 and thumb_to_index_tip < 0.65: label = "C"
                    else: label = "closed fist"
                elif i_ext and thumb_to_knuckle_5 > 0.5 and not m_ext:
                    label = "L"

        valid_signs = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","Ñ","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
        
        if label in valid_signs:
            if label == stable_letter:
                elapsed = time.time() - stable_since
                remaining_time = max(0.0, REQUIRED_STABLE_TIME - elapsed)
                
                # Visual ring progress indicator on video frame
                cv2.circle(img, (70, 200), 30, (50, 50, 50), -1)
                angle = int((elapsed / REQUIRED_STABLE_TIME) * 360)
                cv2.ellipse(img, (70, 200), (30, 30), 0, 0, min(angle, 360), (0, 255, 0), 4)
                
                if elapsed >= REQUIRED_STABLE_TIME:
                    current_word += label  # Commit the letter to the text engine
                    speak_letter_async(label)
                    stable_since = time.time() # Reset clock to allow typing repeats
            else:
                stable_letter = label
                stable_since = time.time()
        else:
            stable_letter = "Searching..."
            stable_since = None

        # Graphic HUD Layer
        cv2.putText(img, f"Active: {label}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
        cv2.putText(img, f"Word: {current_word}", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

        ret, buffer = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- WEB OVERLAYS AND ENDPOINTS ---
@app.route('/video_feed')
def video_feed():
    return Response(generate_lsc_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    """Poll endpoint to keep the web dashboard metrics in sync."""
    global current_word, stable_letter, stable_since
    elapsed = (time.time() - stable_since) if stable_since else 0.0
    pct = min(100, int((elapsed / REQUIRED_STABLE_TIME) * 100)) if stable_since else 0
    return jsonify({
        "word": current_word,
        "letter": stable_letter,
        "progress": pct
    })

@app.route('/clear_word')
def clear_word():
    global current_word
    current_word = ""
    return jsonify({"status": "cleared"})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)