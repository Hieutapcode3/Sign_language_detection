import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  
        for img_path in os.listdir(dir_path):
            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                all_data_aux = [] 
                all_x_coords = []
                all_y_coords = []

                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x)
                        data_aux.append(landmark.y)

                    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                    all_x_coords.extend(x_coords)
                    all_y_coords.extend(y_coords)
                    all_data_aux.extend(data_aux)

                if all_data_aux:
                    # Tạo bounding box lớn bao quanh cả hai tay
                    min_x = min(all_x_coords)
                    min_y = min(all_y_coords)

                    normalized_data = []
                    for i in range(0, len(all_data_aux), 2):
                        normalized_data.append(all_data_aux[i] - min_x)
                        normalized_data.append(all_data_aux[i + 1] - min_y)

                    data.append(normalized_data)
                    labels.append(dir_)  

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Number of images collected per class:")
for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(class_dir):
        print(f'Class {dir_}: {len(os.listdir(class_dir))} images collected')
    else:
        print(f'{class_dir} is not a directory')
