import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------
# 1. Set paths
# ----------------------------
clips_path = "/Users/sincymol/Downloads/CNN_PROJECT/TRAINING/clip"
toothbrush_path = "/Users/sincymol/Downloads/CNN_PROJECT/TRAINING/toothbrush"
shoes_path = "/Users/sincymol/Downloads/CNN_PROJECT/TRAINING/shoes123"

class_paths = {
    "shoes": shoes_path,
    "clips": clips_path,
    "toothbrush": toothbrush_path
}

# ----------------------------
# 2. Load images and labels
# ----------------------------
IMG_SIZE = (128, 128)
images, labels = [], []

for label, folder in class_paths.items():
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}, skipping this class")
        continue
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                img_path = os.path.join(folder, file)
                img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
                img_array = tf.keras.utils.img_to_array(img)
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Skipped {file}: {e}")

images = np.array(images)
labels = np.array(labels)

if len(images) == 0:
    raise ValueError("No images loaded! Check your folder paths and image files.")

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# ----------------------------
# 3. Train / Validation split
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    images, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# Normalize images
X_train = X_train / 255.0
X_val = X_val / 255.0

# ----------------------------
# 4. Data Augmentation
# ----------------------------
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

datagen.fit(X_train)

# Save some augmented images
augmented_output = "augmented_images"
os.makedirs(augmented_output, exist_ok=True)

save_gen = datagen.flow(
    X_train, y_train,
    batch_size=10,
    save_to_dir=augmented_output,
    save_prefix="aug",
    save_format="jpg"
)

for i in range(30):  # Save 30 augmented images
    next(save_gen)

# ----------------------------
# 5. Build CNN model (MobileNetV2)
# ----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights=None
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_paths), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# 6. Train the model
# ----------------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=8),
    validation_data=(X_val, y_val),
    epochs=15,
    verbose=1
)

# ----------------------------
# 7. Fine-tuning 
# ----------------------------
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    datagen.flow(X_train, y_train, batch_size=8),
    validation_data=(X_val, y_val),
    epochs=10,
    verbose=1
)

# ----------------------------
# 8. Prediction Function
# ----------------------------
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"⚠️ File not found: {img_path}")
        return
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    class_name = le.inverse_transform([class_idx])[0]
    print(f"Prediction: {class_name}, Confidence: {confidence:.2f}")

# Example usage
predict_image("/Users/sincymol/Downloads/CNN_PROJECT/TESTING/IMG_4890 Medium.jpeg")
