import numpy as np # xử lý mảng đa chiều (ma trận)
import pandas as pd # xử lý dữ liệu dạng bảng
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 # xử lý ảnh, video: đọc, resize, làm mờ ... ảnh
import os

df_train = pd.read_csv('Fall/train_labels.csv', index_col='images')
df_train

test_df = pd.read_csv('Fall/test_labels.csv', index_col='images')
test_df

import os
import cv2
import numpy as np

dataset_folder = 'Fall'

# Khởi tạo danh sách trống
train_images = []
train_labels = []
test_images = []
test_labels = []

IMG_SIZE = 96  # Đặt kích thước ảnh đích

for folder in os.listdir(dataset_folder):  # Lặp qua tất cả các thư mục trong "Fall"
    folder_path = os.path.join(dataset_folder, folder)
    if folder == 'train_images':  # Đọc ảnh huấn luyện
        for file in os.listdir(folder_path):
            if file.endswith(('jpg', 'png')):  # Đọc cả jpg và png
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)  # Đọc ảnh (dưới dạng BGR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize về 96x96
                    train_images.append(img)
                    train_labels.append(df_train.loc[file, 'labels'])  # Lấy nhãn từ df_train
    elif folder == 'test_images':  # Đọc ảnh kiểm tra
        for file in os.listdir(folder_path):
            if file.endswith(('jpg', 'png')):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    test_images.append(img)
                    test_labels.append(test_df.loc[file, 'labels'])

# Chuyển sang numpy array
train_images = np.array(train_images, dtype='float32') / 255.0  # Normalize về [0,1]
train_labels = np.array(train_labels)
test_images = np.array(test_images, dtype='float32') / 255.0
test_labels = np.array(test_labels)

# In kích thước dữ liệu
print('Shape of stacked train images:', train_images.shape)
print('Shape of train labels:', train_labels.shape)
print('Shape of stacked test images:', test_images.shape)
print('Shape of test labels:', test_labels.shape)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, stratify=train_labels, test_size=0.2)

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.regularizers import l2

# FallNet architecture
model_input = Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), name='inputs')

conv1 = Conv2D(16, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(0.), bias_regularizer=l2(0.), name='convolution_1')(model_input)
pool1 = MaxPooling2D(pool_size=(2, 2), name='pooling_1')(conv1)

conv2 = Conv2D(16, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(0.), bias_regularizer=l2(0.), name='convolution_2')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2), name='pooling_2')(conv2)

conv3 = Conv2D(32, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(0.), bias_regularizer=l2(0.), name='convolution_3')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2), name='pooling_3')(conv3)

conv4 = Conv2D(32, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(0.), bias_regularizer=l2(0.), name='convolution_4')(pool3)
pool4 = MaxPooling2D(pool_size=(2, 2), name='pooling_4')(conv4)

conv5 = Conv2D(64, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(0.), bias_regularizer=l2(0.), name='convolution_5')(pool4)
pool5 = MaxPooling2D(pool_size=(2, 2), name='pooling_5')(conv5)

conv6 = Conv2D(64, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(0.), bias_regularizer=l2(0.), name='convolution_6')(pool5)
pool6 = MaxPooling2D(pool_size=(2, 2), name='pooling_6')(conv6)

flat = Flatten(name='flatten')(pool6)
dense1 = Dense(32, activation='relu', name='dense1')(flat)
output = Dense(1, activation='sigmoid', name='output')(dense1)

model = Model(inputs=[model_input], outputs=[output])
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

model.save('model.h5')

import time
start = time.time()
predictions = model.predict(test_images)
end = time.time()

# Đánh giá
predicted_labels = (predictions > 0.5).astype('int').flatten()
accuracy = np.mean(predicted_labels == test_labels)

print(f"Inference time (CPU): {end - start:.2f} seconds")
print(f"Test Accuracy: {accuracy:.4f}")

# Đường dẫn tới video cần kiểm tra
video_path = 'test_video.mp4'

# Mở video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video.")
    exit()

IMG_SIZE = 96  # Kích thước đầu vào của mô hình
frame_count = 0
total_inference_time = 0

# Tùy chọn: hiển thị video
display = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize và tiền xử lý
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension

    # Inference
    start_time = time.time()
    prediction = model.predict(img, verbose=0)
    end_time = time.time()

    inference_time = end_time - start_time
    total_inference_time += inference_time
    frame_count += 1

    label = "Fall" if prediction[0][0] > 0.5 else "No Fall"
    label_color = (0, 0, 255) if label == "Fall" else (0, 255, 0)

    if display:
        # Ghi nhãn lên frame
        cv2.putText(frame, f"{label} ({prediction[0][0]:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
        cv2.imshow("Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Kết quả thống kê
if frame_count > 0:
    avg_inference_time = total_inference_time / frame_count
    fps = 1.0 / avg_inference_time

    print(f"Tổng số frame: {frame_count}")
    print(f"Thời gian inference trung bình mỗi frame: {avg_inference_time:.4f} giây")
    print(f"FPS ước tính: {fps:.2f} khung hình/giây")
else:
    print("Không có frame nào được xử lý.")