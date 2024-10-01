import pickle
import cv2
import mediapipe as mp
import numpy as np
import time  # Thêm thư viện time để sử dụng thời gian

with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

labels_dict = {0: 'Hello', 1: 'I Love You', 2: 'No', 3: 'Clear', 4: 'Thank You', 5: 'I', 6: 'Love', 7: 'Phenikaa', 8: "I'm just kidding"}

text_output = ""
previous_character_left = ""  
previous_character_right = ""  

last_update_time_left = 0  # Lưu thời gian cập nhật ký tự cho tay trái
last_update_time_right = 0  # Lưu thời gian cập nhật ký tự cho tay phải
update_delay = 1  # Đặt độ trễ 1 giây

white_color = (255, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    current_time = time.time()  # Lấy thời gian hiện tại để so sánh với lần cập nhật trước
    
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 1:
            # Xử lý khi chỉ có 1 tay
            for hand_landmarks, hand_type in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame,  
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing.DrawingSpec(color=white_color, thickness=2, circle_radius=3), 
                    mp_drawing.DrawingSpec(color=white_color, thickness=2)  
                )

                # Xác định tay trái hoặc phải
                label = hand_type.classification[0].label 
                data_aux = []
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)
                if data_aux:
                    x_coords = data_aux[::2]
                    y_coords = data_aux[1::2]
                    min_x = min(x_coords)
                    min_y = min(y_coords)
                    normalized_data = []
                    for i in range(len(x_coords)):
                        normalized_data.append(x_coords[i] - min_x)
                        normalized_data.append(y_coords[i] - min_y)
                    
                    if len(normalized_data) == 42:
                        normalized_data += [0] * 42  

                    normalized_data = np.asarray(normalized_data).reshape(1, -1)

                    prediction = model.predict(normalized_data)[0]
                    predicted_character = labels_dict.get(int(prediction), "Unknown")

                    if label == "Left":
                        if predicted_character == "Clear":
                            text_output = ""  
                        # Kiểm tra nếu đã qua 1 giây kể từ lần cập nhật cuối
                        elif predicted_character != previous_character_left and current_time - last_update_time_left > update_delay:
                            text_output += predicted_character + " "
                            previous_character_left = predicted_character
                            last_update_time_left = current_time  # Cập nhật thời gian sau khi thêm ký tự

                        # Vẽ bounding box cho tay trái
                        x_coords_pixels = [int(lm.x * W) for lm in hand_landmarks.landmark]
                        y_coords_pixels = [int(lm.y * H) for lm in hand_landmarks.landmark]
                        x1 = min(x_coords_pixels) - 10
                        y1 = min(y_coords_pixels) - 10
                        x2 = max(x_coords_pixels) + 10
                        y2 = max(y_coords_pixels) + 10

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

                    elif label == "Right":
                        if predicted_character == "Clear":
                            text_output = "" 
                        # Kiểm tra nếu đã qua 1 giây kể từ lần cập nhật cuối
                        elif predicted_character != previous_character_right and current_time - last_update_time_right > update_delay:
                            text_output += predicted_character + " "
                            previous_character_right = predicted_character
                            last_update_time_right = current_time  # Cập nhật thời gian sau khi thêm ký tự

                        # Vẽ bounding box cho tay phải
                        x_coords_pixels = [int(lm.x * W) for lm in hand_landmarks.landmark]
                        y_coords_pixels = [int(lm.y * H) for lm in hand_landmarks.landmark]
                        x1 = min(x_coords_pixels) - 10
                        y1 = min(y_coords_pixels) - 10
                        x2 = max(x_coords_pixels) + 10
                        y2 = max(y_coords_pixels) + 10

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        # Xử lý khi có 2 tay
        elif len(results.multi_hand_landmarks) == 2:
            all_x_coords = []
            all_y_coords = []
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=white_color, thickness=2, circle_radius=3), 
                    mp_drawing.DrawingSpec(color=white_color, thickness=2)  
                )
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                all_x_coords.extend(x_coords)
                all_y_coords.extend(y_coords)

                min_x = min(x_coords)
                min_y = min(y_coords)
                for i in range(len(x_coords)):
                    data_aux.append(x_coords[i] - min_x)
                    data_aux.append(y_coords[i] - min_y)

            # Dự đoán ký hiệu cho cả hai tay
            normalized_data = np.asarray(data_aux).reshape(1, -1)
            prediction = model.predict(normalized_data)[0]
            predicted_character = labels_dict.get(int(prediction), "Unknown")

            if predicted_character == "Clear":
                text_output = ""
            # Kiểm tra nếu đã qua 1 giây kể từ lần cập nhật cuối
            elif predicted_character != previous_character_left and predicted_character != previous_character_right and current_time - last_update_time_left > update_delay and current_time - last_update_time_right > update_delay:
                text_output += predicted_character + " "
                previous_character_left = predicted_character
                previous_character_right = predicted_character
                last_update_time_left = current_time
                last_update_time_right = current_time

            # Vẽ bounding box cho cả hai tay
            x1 = int(min(all_x_coords) * W) - 10
            y1 = int(min(all_y_coords) * H) - 10
            x2 = int(max(all_x_coords) * W) + 10
            y2 = int(max(all_y_coords) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    max_text_length = 40
    if len(text_output) > max_text_length:
        text_output = ""
    cv2.rectangle(frame, (0, H - 50), (W, H), (255, 255, 255), -1)  
    cv2.putText(frame, text_output, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
