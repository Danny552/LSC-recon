import cv2
import mediapipe as mp
import math

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def dist_2d(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    
    # img = cv2.flip(img, 1) # Counter-mirroring (off)
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    label = "Searching..."

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            lm = hand_lms.landmark

            # 1. Distance Helper: Tip to Palm Base (Landmark 0)
            def get_ext_ratio(tip_idx):
                # Ratio of (Tip-to-Wrist) / (Knuckle-to-Wrist)
                # If > 1.2, finger is likely extended.
                knuckle_dist = dist_2d(lm[tip_idx-2], lm[0])
                tip_dist = dist_2d(lm[tip_idx], lm[0])
                return tip_dist / knuckle_dist if knuckle_dist != 0 else 0

            i_ext = get_ext_ratio(8) > 1.2
            m_ext = get_ext_ratio(12) > 1.2
            r_ext = get_ext_ratio(16) > 1.2
            p_ext = get_ext_ratio(20) > 1.2

            # 2. Key Landmark Distances for A vs O
            thumb_to_index_tip = dist_2d(lm[4], lm[8])
            thumb_to_mid_tip = dist_2d(lm[4], lm[12])
            thumb_to_index_knuckle = dist_2d(lm[4], lm[5])

            # --- LSC TROUBLESHOOTING LOGIC ---

            # LETTER B: All extended and touching
            if i_ext and m_ext and r_ext and p_ext:
                label = "LSC: B"

            # LETTER D: Only index up
            elif i_ext and not m_ext and not r_ext and not p_ext:
                label = "LSC: D"

            # LETTER V vs U: Two fingers up
            elif i_ext and m_ext and not r_ext:
                label = "LSC: V" if dist_2d(lm[8], lm[12]) > 0.08 else "LSC: U"

            # LETTER I: Only pinky up
            elif p_ext and not i_ext and not m_ext and not r_ext:
                label = "LSC: I"

            # --- THE "A" vs "O" vs "C" ZONE ---
            elif not i_ext and not m_ext and not r_ext:
                
                # LETTER O: Circle (Thumb tip touches Index and/or Middle tip)
                if thumb_to_index_tip < 0.05 or thumb_to_mid_tip < 0.05:
                    label = "LSC: O"
                
                # LETTER C: Claw (Fingers curved, but a wide gap)
                elif thumb_to_index_tip > 0.12 and thumb_to_index_tip < 0.25:
                    label = "LSC: C"
                
                # LETTER A: Fist (Thumb is tucked near knuckles, NOT tips)
                elif thumb_to_index_knuckle < 0.07:
                    label = "LSC: A"
                
                else:
                    label = "Closed Fist / A?"

            # LETTER L: Index up + Thumb out
            if i_ext and dist_2d(lm[4], lm[5]) > 0.12 and not m_ext:
                label = "LSC: L"

    # Display results
    cv2.putText(img, label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    cv2.imshow("LSC Troubleshooter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()