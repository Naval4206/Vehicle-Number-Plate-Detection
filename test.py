import numpy as np
import cv2
import imutils
import pytesseract
import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Point to the Tesseract executable if it's not in the PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if needed

# Characters we want to recognize (A-Z and 0-9)
characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
num_classes = len(characters)
img_size = 28
dataset_path = "character_dataset"

# Step 1: Generate Synthetic Data for Training the CNN
os.makedirs(dataset_path, exist_ok=True)
for char in characters:
    char_path = os.path.join(dataset_path, char)
    os.makedirs(char_path, exist_ok=True)
    for i in range(100):  # Generate 100 images per character
        img = np.ones((img_size, img_size), dtype=np.uint8) * 255  # White background
        cv2.putText(img, char, (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0), 2, cv2.LINE_AA)
        img_path = os.path.join(char_path, f"{char}_{i}.png")
        cv2.imwrite(img_path, img)

# Step 2: Define a CNN Model for Character Recognition
def create_character_recognition_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_character_recognition_model()

# Step 3: Train the CNN on Synthetic Data
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32
)
model.fit(train_generator, epochs=50)
model.save("character_recognition_model.h5")

# Load the trained model
character_recognition_model = tf.keras.models.load_model("character_recognition_model.h5")

# Function to predict a character from an image
def predict_character(image, model):
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    prediction = model.predict(image)
    char_index = np.argmax(prediction)
    return characters[char_index]

# Function to segment characters in the license plate image
def segment_characters(license_plate_img):
    gray = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    character_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < w < 50 and 15 < h < 50:
            char_img = binary_img[y:y + h, x:x + w]
            character_images.append(char_img)
    character_images = sorted(character_images, key=lambda img: cv2.boundingRect(img)[0])
    return character_images

# Load the image
image = cv2.imread('img 1.jpg')
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show grayscale image
cv2.imshow("Grayscale Image", gray)

# Apply bilateral filter and show filtered image
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Bilateral Filtered Image", gray)

# Edge Detection using Canny and show edge-detected image
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("Canny Edges", edged)

# Find contours
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCnt = None

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        NumberPlateCnt = approx
        break

# Mask the license plate area and show masked image
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
new_image = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Masked License Plate", new_image)

# Attempt OCR
config = ('-l eng --oem 1 --psm 3')
text = pytesseract.image_to_string(new_image, config=config)

# Segment characters and recognize using CNN
character_images = segment_characters(new_image)
recognized_text = ""
for char_img in character_images:
    recognized_text += predict_character(char_img, character_recognition_model)

# Save both OCR and CNN recognized text with timestamp
raw_data = {
    'date': [time.asctime(time.localtime(time.time()))],
    'v_number_OCR': [text.strip()],
    'v_number_CNN': [recognized_text.strip()]
}
df = pd.DataFrame(raw_data, columns=['date', 'v_number_OCR', 'v_number_CNN'])
df.to_csv('recognized_license_plate_data.csv', index=False)

print("Tesseract OCR Text:", text.strip())
print("CNN Recognized License Plate Text:", recognized_text.strip())

# Show segmented characters
for i, char_img in enumerate(character_images, start=1):
    cv2.imshow(f"Character Segment {i}", char_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
