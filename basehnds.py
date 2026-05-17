import cv2
import mediapipe as mp
import math
"""
letras con movimiento G,H,J,Ñ,S,Z
puede mejorar: R, Q, M, N 
"""

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
                knuckle_dist = dist_2d(lm[tip_idx-2], lm[0])
                tip_dist = dist_2d(lm[tip_idx], lm[0])
                return tip_dist / knuckle_dist if knuckle_dist != 0 else 0

            i_ext = get_ext_ratio(8) > 1.2
            m_ext = get_ext_ratio(12) > 1.2
            r_ext = get_ext_ratio(16) > 1.2
            p_ext = get_ext_ratio(20) > 1.2

            # --- ESCALA DE LA MANO (Para normalizar distancias) ---
            # Usamos la distancia del nudillo del índice a la muñeca como referencia de tamaño
            hand_scale = dist_2d(lm[5], lm[0]) if dist_2d(lm[5], lm[0]) != 0 else 1.0

            # 2. Key Landmark Distances (Normalizadas con la escala de la mano)
            thumb_to_index_tip = dist_2d(lm[4], lm[8]) / hand_scale
            thumb_to_mid_tip = dist_2d(lm[4], lm[12]) / hand_scale
            
            # distancias especificas de pulgar(4) a nudillos o puntos clave
            thumb_to_knuckle_5 = dist_2d(lm[4], lm[5]) / hand_scale
            thumb_to_knuckle_6 = dist_2d(lm[4], lm[6]) / hand_scale
            thumb_to_knuckle_9 = dist_2d(lm[4], lm[9]) / hand_scale
            thumb_to_knuckle_10 = dist_2d(lm[4], lm[10]) / hand_scale
            thumb_to_knuckle_14 = dist_2d(lm[4], lm[14]) / hand_scale
            thumb_to_knuckle_15 = dist_2d(lm[4], lm[15]) / hand_scale
            thumb_to_point_0 = dist_2d(lm[4], lm[0]) / hand_scale

            #distancia punta de los dedos a punta pulgar 
            i_tip_to_thumb = dist_2d(lm[8], lm[4]) / hand_scale
            m_tip_to_thumb = dist_2d(lm[12], lm[4]) / hand_scale
            r_tip_to_thumb = dist_2d(lm[16], lm[4]) / hand_scale
            p_tip_to_thumb = dist_2d(lm[20], lm[4]) / hand_scale

            #distancia index y midle
            i_tip_to_mid_tip = dist_2d(lm[8], lm[12]) / hand_scale

            #distancia middle y index puntos (7-6)
            mid_to_index_knuckle_6 = dist_2d(lm[9], lm[6]) / hand_scale
            mid_to_index_kuckle_7 = dist_2d(lm[9], lm[7]) / hand_scale

            #index gancho para X
            i_hook = (lm[8].y < lm[6].y) and (not i_ext)

            #distancia index a punto 2
            index_to_point_2 = dist_2d(lm[8], lm[2]) / hand_scale
        
            #distancia middle a punto 3
            middle_to_point_3 = dist_2d(lm[12], lm[3]) / hand_scale

            #distancia ring 0 - 1
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
            
            # LETTER X: index hook 
            elif i_hook and not m_ext and not r_ext and not p_ext:
                label = "LSC: X"


            # LETTER D: Only index up
            elif (i_ext and not m_ext and not r_ext and not p_ext and
            (m_tip_to_thumb < 0.4 and r_tip_to_thumb < 0.4 and p_tip_to_thumb < 0.4)):
                label = "LSC: D"
            
            #LETTER T: index and thumb touching, other fingers down 
            elif (p_ext and m_ext and r_ext and (i_tip_to_thumb < 0.5)):
                label = "LSC: T"

            #LETTER P: pinky and ring down, middle touch index at 7-6
            elif (not p_ext and not r_ext and i_ext and (mid_to_index_knuckle_6 < 0.4 or mid_to_index_kuckle_7 < 0.4) and not m_ext):
                label = "LSC: P"


            # LETTER V vs R: Index and Middle up
            elif i_ext and m_ext and not r_ext and not p_ext:
                #LETTER R: index and middle touch  up
                if i_tip_to_mid_tip < 0.35:
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
                label = "LSC: I"

            #LETTER F: only index up and thumb close to index
            elif (not m_ext and not r_ext and not p_ext and i_ext and
            (thumb_to_knuckle_5 < 0.35 or thumb_to_knuckle_6 < 0.35)):
                label = "LSC: F"


            # --- THE "A" vs "O" vs "C" vs "E" ZONE ---
            # Condición: Todos los dedos largos cerrados (incluyendo el meñique para asegurar el puño)
            elif not i_ext and not m_ext and not r_ext and not p_ext:

                #distancias para E 
                i_tip_to_mcp = dist_2d(lm[8], lm[5]) / hand_scale
                m_tip_to_mcp = dist_2d(lm[12], lm[9]) / hand_scale
                r_tip_to_mcp = dist_2d(lm[16], lm[13]) / hand_scale
                p_tip_to_mcp = dist_2d(lm[20], lm[17]) / hand_scale

                #Posiciones X del pulgar respecto a las falanges de los dedos
                thumb_x = lm[4].x
                idx_mcp_x = lm[5].x
                mid_mcp_x = lm[9].x
                ring_mcp_x = lm[13].x
                
                # LETTER O: Circle (Thumb tip touches Index and/or Middle tip)
                if thumb_to_index_tip < 0.25 or thumb_to_mid_tip < 0.2:
                    label = "LSC: O"
                
                #LETTER E: all fingers curled
                elif (i_tip_to_mcp < 0.4 and m_tip_to_mcp < 0.4 and r_tip_to_mcp < 0.4 
                    and p_tip_to_mcp < 0.35 and (thumb_to_index_tip < 0.4 or thumb_to_mid_tip < 0.4 or
                    thumb_to_point_0 < 1)):
                    label = "LSC: E"

                # LETTER N: index cerca de 2 y middle cerca de 3
                if (index_to_point_2 < 0.3 and middle_to_point_3 < 0.3):
                    if (ring_to_point_1 < 0.5 or ring_to_point_0 < 0.5):
                        label = "LSC: M"
                    else:
                        label = "LSC: N"

                
                
                # LETTER A: El pulgar está muy cerca del punto 5 O del punto 6
                elif thumb_to_knuckle_5 < 0.35 or thumb_to_knuckle_6 < 0.35:
                    label = "LSC: A"
                
                # LETTER C: Claw (Fingers curved, but a wide gap)
                elif thumb_to_index_tip > 0.5 and thumb_to_index_tip < 1.2:
                    label = "LSC: C"
                
                else:
                    label = "Closed Fist"

            # LETTER L: Index up + Thumb out
            elif i_ext and (dist_2d(lm[4], lm[5]) / hand_scale) > 0.5 and not m_ext:
                label = "LSC: L"

            
            

    # Display results
    cv2.putText(img, label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    cv2.imshow("LSC Troubleshooter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()