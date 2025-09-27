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
model.fit(train_generator, epochs=10)
model.save("character_recognition_model.h5")