import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load mô hình đã huấn luyện
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Chấp nhận 2 tay nhưng chỉ dùng 1 tay cho dự đoán
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# Các nhãn ký hiệu
labels_dict = {0: 'Hello', 1: 'I Love You', 2: 'No', 3: 'Clear'}

# Chuỗi để hiển thị câu chữ
text_output = ""
previous_character = ""  # Biến để lưu ký tự dự đoán trước đó

# Màu trắng cho các điểm và đường nối
white_color = (255, 255, 255)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xử lý ảnh để tìm các landmarks
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Lấy landmarks của bàn tay đầu tiên
        hand_landmarks = results.multi_hand_landmarks[0]

        # Vẽ các landmarks
        mp_drawing.draw_landmarks(
            frame,  # ảnh để vẽ
            hand_landmarks,  # đầu ra từ mô hình
            mp_hands.HAND_CONNECTIONS,  # các kết nối tay
            mp_drawing.DrawingSpec(color=white_color, thickness=2, circle_radius=3),  # điểm màu trắng
            mp_drawing.DrawingSpec(color=white_color, thickness=2)  # đường nối màu trắng
        )

        # Lưu tọa độ các landmarks
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        # Chuẩn hóa tọa độ
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Dự đoán ký hiệu bằng mô hình
        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        # Nếu ký hiệu là "Clear", xóa toàn bộ text_output
        if predicted_character == "Clear":
            text_output = ""
        # Kiểm tra nếu ký hiệu khác với ký hiệu trước đó, thì mới thêm vào
        elif predicted_character != previous_character:
            text_output += predicted_character + " "
            previous_character = predicted_character

        # Kiểm tra nếu text_output dài quá thì xóa hết
        max_text_length = 50  # Giới hạn chiều dài chuỗi
        if len(text_output) > max_text_length:
            text_output = ""

        # Vẽ khung chữ nhật và ký tự dự đoán lên khung hình
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Vẽ một ô text ở phía dưới màn hình để hiển thị câu, sát lề trái và phải
    cv2.rectangle(frame, (0, H - 50), (W, H), (255, 255, 255), -1)  # Full chiều rộng của frame
    cv2.putText(frame, text_output, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # Hiển thị khung hình
    cv2.imshow('SIGN LANGUAGE', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
