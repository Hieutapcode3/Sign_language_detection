import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dữ liệu
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

processed_data = []
for sample in data_dict['data']:
    if len(sample) == 42:
        sample += [0] * 42
    processed_data.append(sample)

data = np.asarray(processed_data)
labels = np.asarray(data_dict['labels'])

print(f'Loaded {len(data)} samples with {data.shape[1]} features per sample.')
print(f'Total {len(set(labels))} classes.')

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

# In kết quả độ chính xác
print(f'{score * 100:.2f}% of samples were classified correctly!')

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
