import cv2
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# Load sample image
img_array = cv2.imread("Training/0/Training_314.jpg")

# Define data path
Datadirectory = "Training/"
Classes = ["0", "1", "2", "3", "4", "5", "6"]

# Show one image from each category (optional preview)
for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.title(f"Category {category}")
        plt.show()
        break
    break

# Resize for uniformity
img_size = 224
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()

# Create training data
training_data = []
def create_training_data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
print("Total training samples:", len(training_data))

# Shuffle and split features and labels
random.shuffle(training_data)
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)
X = X / 255.0
Y = np.array(y)

# Load MobileNetV2 as base model
base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224,224,3), pooling='avg')
base_output = base_model.output

# Add custom layers
final_output = tf.keras.layers.Dense(128, activation='relu')(base_output)
final_output = tf.keras.layers.Dense(64, activation='relu')(final_output)
final_output = tf.keras.layers.Dense(7, activation='softmax')(final_output)

new_model = tf.keras.Model(inputs=base_model.input, outputs=final_output)
new_model.summary()

# Compile model
new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
new_model.fit(X, Y, epochs=25)

# Save model
new_model.save('final_model_95p07.h5')
new_model = tf.keras.models.load_model('final_model_95p07.h5')


# Load the face cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        backtorgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        final_image = cv2.resize(backtorgb, (224, 224))
        final_image = np.expand_dims(final_image, axis=0) / 255.0

        Predictions = new_model.predict(final_image)
        predicted_class = np.argmax(Predictions)
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        Status = emotions[predicted_class]

        # Draw the rectangle and text
        x1, y1, box_w, box_h = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (0, 0, 0), -1)
        cv2.putText(frame, Status, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Facial Emotion Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
