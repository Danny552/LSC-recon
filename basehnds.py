import cv2
import mediapipe as mp
import math
from collections import deque 
"""
Letters with motion: G, J, Ñ, S, Z
Improvements possible for: R, Q, M, N, J
failed: P

"""

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def dist_2d(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

#for dynamic gestures
#H
wrist_x_history = deque(maxlen=12)
pinky_history = deque(maxlen=20)  
j_cooldown = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    label = "Searching..."

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            lm = hand_lms.landmark

            # save wrist X position in history for dynamic gesture detection
            wrist_x_history.append(lm[0].x)
            pinky_history.append((lm[20].x, lm[20].y))

            if j_cooldown > 0:
                j_cooldown -= 1

            mov_x = 0
            if len(wrist_x_history) == wrist_x_history.maxlen:
                mov_x = abs(wrist_x_history[-1] - wrist_x_history[0])

            #Distance Helper: Tip to Palm Base (Landmark 0)
            def get_ext_ratio(tip_idx):
                knuckle_dist = dist_2d(lm[tip_idx-2], lm[0])
                tip_dist = dist_2d(lm[tip_idx], lm[0])
                return tip_dist / knuckle_dist if knuckle_dist != 0 else 0

            i_ext = get_ext_ratio(8) > 1.2
            m_ext = get_ext_ratio(12) > 1.2
            r_ext = get_ext_ratio(16) > 1.2
            p_ext = get_ext_ratio(20) > 1.2

            # --- HAND SCALE (to normalize distances) ---
            # Use the distance from the index knuckle to the wrist as a size reference
            hand_scale = dist_2d(lm[5], lm[0]) if dist_2d(lm[5], lm[0]) != 0 else 1.0

            #THUMB
            #Key landmark distances (normalized by hand scale)
            thumb_to_index_tip = dist_2d(lm[4], lm[8]) / hand_scale
            thumb_to_mid_tip = dist_2d(lm[4], lm[12]) / hand_scale
            
            # specific distances from thumb (4) to knuckles or key points
            thumb_to_knuckle_5 = dist_2d(lm[4], lm[5]) / hand_scale
            thumb_to_knuckle_6 = dist_2d(lm[4], lm[6]) / hand_scale
            thumb_to_knuckle_9 = dist_2d(lm[4], lm[9]) / hand_scale
            thumb_to_knuckle_10 = dist_2d(lm[4], lm[10]) / hand_scale
            thumb_to_knuckle_14 = dist_2d(lm[4], lm[14]) / hand_scale
            thumb_to_knuckle_15 = dist_2d(lm[4], lm[15]) / hand_scale
            thumb_to_point_0 = dist_2d(lm[4], lm[0]) / hand_scale

            # distance from fingertips to thumb tip
            i_tip_to_thumb = dist_2d(lm[8], lm[4]) / hand_scale
            m_tip_to_thumb = dist_2d(lm[12], lm[4]) / hand_scale
            r_tip_to_thumb = dist_2d(lm[16], lm[4]) / hand_scale
            p_tip_to_thumb = dist_2d(lm[20], lm[4]) / hand_scale

            #INDEX
            # distance index to middle
            i_tip_to_mid_tip = dist_2d(lm[8], lm[12]) / hand_scale
            # distance index to point 2
            index_to_point_2 = dist_2d(lm[8], lm[2]) / hand_scale
            # index hook for X (tip close to PIP, not extended)
            index_tip_to_pip = dist_2d(lm[8], lm[6]) / hand_scale
            index_tip_to_mcp = dist_2d(lm[8], lm[5]) / hand_scale
            i_hook = (not i_ext) and index_tip_to_pip < 0.25 and index_tip_to_mcp < 0.55

            #MIDDLE
            # distance middle to index points (7-6)
            mid_to_index_knuckle_6 = dist_2d(lm[9], lm[6]) / hand_scale
            mid_to_index_kuckle_7 = dist_2d(lm[9], lm[7]) / hand_scale
            # distance middle to point 3
            middle_to_point_3 = dist_2d(lm[12], lm[3]) / hand_scale


            # RING
            # distance ring 0 - 1
            ring_to_point_1 = dist_2d(lm[16], lm[1]) / hand_scale
            ring_to_point_0 = dist_2d(lm[16], lm[0]) / hand_scale



            # --- LSC TROUBLESHOOTING LOGIC ---

            # LETTER Q: All extended but touching and the top 
            if (i_ext and m_ext and r_ext and p_ext and 
            (thumb_to_index_tip < 0.25 or thumb_to_mid_tip < 0.25)):
                label = "LSC: Q"

            # LETTER B: All extended and touching
            elif i_ext and m_ext and r_ext and p_ext and thumb_to_point_0 < 0.8:
                label = "LSC: B"
            

            # LETTER D: Only index up (make sure index is really extended)
            elif (i_ext and not m_ext and not r_ext and not p_ext and
            index_tip_to_mcp > 0.65 and index_tip_to_pip > 0.35 and
            (m_tip_to_thumb < 0.4 and r_tip_to_thumb < 0.4 and p_tip_to_thumb < 0.4)):
                label = "LSC: D"
            
            #LETTER T: index and thumb touching, other fingers down 
            elif (p_ext and m_ext and r_ext and (i_tip_to_thumb < 0.5)):
                label = "LSC: T"

            #empezo a fallar/ ya no reconoce bien
            #LETTER P: pinky and ring down, middle touch index at 7-6
            elif (not p_ext and not r_ext and i_ext and (mid_to_index_knuckle_6 < 0.4 or mid_to_index_kuckle_7 < 0.4) and not m_ext):
                label = "LSC: P"

            #control of H, R, k, V: index and middle up
            elif i_ext and m_ext and not r_ext and not p_ext:

                is_horizontal = abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y)
                # LETTER H
                if is_horizontal or mov_x > 0.04:
                    label = "LSC: H"
                #LETTER R: index and middle touch  up
                elif i_tip_to_mid_tip < 0.35:
                    label = "LSC: R"
                #LETTER K: thumb up close to middle and index (6-10)
                elif(thumb_to_knuckle_6 < 0.4 or thumb_to_knuckle_10 < 0.4):
                    label = "LSC: K"

                #LETTER V: thumb touches ring knuckle (14-15)
                elif (thumb_to_knuckle_14 < 0.3 or thumb_to_knuckle_15 < 0.3):
                    label = "LSC: V"

            # LETTER U: Index and Pinky up
            elif i_ext and p_ext and not m_ext and not r_ext:
                label = "LSC: U"
            
            #LETER W: Three fingers up (Index, Middle, Ring)
            elif r_ext and m_ext and i_ext and not p_ext:
                label = "LSC: W"

            # LETTER Y: Thumb and Pinky out
            elif p_ext and thumb_to_knuckle_5 > 0.5 and not i_ext and not m_ext and not r_ext:
                label = "LSC: Y"

            # LETTER I: Only pinky up
            elif p_ext and not i_ext and not m_ext and not r_ext:

                is_j_motion = False 

                if len(pinky_history) == pinky_history.maxlen:
                    y_coords = [p[1] for p in pinky_history]
                    x_coords = [p[0] for p in pinky_history]
                    
                    max_y = max(y_coords)
                    min_y = min(y_coords)
                    max_y_idx = y_coords.index(max_y)

                    vertical_travel = max_y - min_y
                    horizontal_travel = max(x_coords) - min(x_coords)

                    if vertical_travel > 0.05 and horizontal_travel > 0.03:
                        if 4 < max_y_idx < 16:  
                            is_j_motion = True

                if is_j_motion or j_cooldown > 0:
                    label = "LSC: J"
                    if is_j_motion:
                        j_cooldown = 18

                else:
                    label = "LSC: I"

            #LETTER F: only index up and thumb close to index
            elif (not m_ext and not r_ext and not p_ext and i_ext and
            (thumb_to_knuckle_5 < 0.35 or thumb_to_knuckle_6 < 0.35)):
                label = "LSC: F"

            # LETTER X: index hook + thumb near middle knuckle 
            elif (i_hook and not m_ext and not r_ext and not p_ext and
            thumb_to_knuckle_9 < 0.5 and thumb_to_point_0 > 0.45 and thumb_to_index_tip > 0.25):
                label = "LSC: X"


            # --- THE "A" vs "O" vs "C" vs "E" ZONE ---
            # Condition: All long fingers closed (include the pinky to ensure a fist)
            elif not i_ext and not m_ext and not r_ext and not p_ext:

                # distances for E 
                i_tip_to_mcp = dist_2d(lm[8], lm[5]) / hand_scale
                m_tip_to_mcp = dist_2d(lm[12], lm[9]) / hand_scale
                r_tip_to_mcp = dist_2d(lm[16], lm[13]) / hand_scale
                p_tip_to_mcp = dist_2d(lm[20], lm[17]) / hand_scale

                # Thumb X positions relative to the fingers' MCP joints
                thumb_x = lm[4].x
                idx_mcp_x = lm[5].x
                mid_mcp_x = lm[9].x
                ring_mcp_x = lm[13].x
                
                # LETTER O: Circle (Thumb tip touches Index and/or Middle tip)
                if thumb_to_index_tip < 0.25 or thumb_to_mid_tip < 0.2:
                    label = "LSC: O"
                
                #LETTER E: all fingers curled, thumb tucked across palm
                elif (i_tip_to_mcp < 0.5 and m_tip_to_mcp < 0.5 and r_tip_to_mcp < 0.5 
                    and p_tip_to_mcp < 0.45 and thumb_to_point_0 < 0.6 and
                    thumb_to_knuckle_5 < 0.45):
                    label = "LSC: E"


                # LETTER N/M: index near point 2 and middle near point 3
                if (index_to_point_2 < 0.35 and middle_to_point_3 < 0.35):
                    if (ring_to_point_1 < 0.45 or ring_to_point_0 < 0.45):
                        label = "LSC: M"
                    else:
                        label = "LSC: N"
                
                # LETTER A: Thumb is very close to point 5 or point 6
                elif (thumb_to_knuckle_5 < 0.35 or thumb_to_knuckle_6 < 0.35) and not i_hook:
                    label = "LSC: A"
                
                # LETTER C: Claw (Fingers curved, but a wide gap)
                elif thumb_to_index_tip > 0.45 and thumb_to_index_tip < 0.8:
                    label = "LSC: C"
                
                else:
                    label = "LSC: E"

            # LETTER L: Index up + Thumb out
            elif i_ext and thumb_to_knuckle_5 > 0.5 and not m_ext:
                label = "LSC: L"

            
            

    # Display results
    cv2.putText(img, label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    cv2.imshow("LSC Troubleshooter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
