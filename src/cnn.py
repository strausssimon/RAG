import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.callbacks import TqdmCallback

class BinaryMushroomCNN:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.model = None
        self.class_names = ["Amanita_muscaria", "Armillaria_mellea"]
        
    def load_data(self):
        print("\n🔄 Loading image data...")
        images = []
        labels = []
        base_path = Path("/Users/celineotten/Documents/Git/SmallLanguageModels/Webscraper/data/images_mushrooms")
        
        # Get total number of images first
        amanita_files = list((base_path / "Amanita_muscaria").glob("*.jpg"))
        armillaria_files = list((base_path / "Armillaria_mellea").glob("*.jpg"))
        
        print("📸 Loading Amanita muscaria images...")
        for img_path in tqdm(amanita_files, desc="Loading Amanita"):
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, self.image_size)
                img = img / 255.0  # Normalize
                images.append(img)
                labels.append(0)
        print(f"   ✓ Loaded {labels.count(0)} Amanita muscaria images")
        
        print("📸 Loading Armillaria mellea images...")
        for img_path in tqdm(armillaria_files, desc="Loading Armillaria"):
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, self.image_size)
                img = img / 255.0  # Normalize
                images.append(img)
                labels.append(1)
        print(f"   ✓ Loaded {labels.count(1)} Armillaria mellea images")
        
        print("🔄 Converting data to numpy arrays...")
        X = np.array(images)
        y = np.array(labels)
        
        # Convert labels to categorical
        y = tf.keras.utils.to_categorical(y, 2)
        
        print("📊 Splitting data into train and test sets...")
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
    def build_model(self):
        print("\n🏗️  Building CNN model...")
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),  # Added dropout to prevent overfitting
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),   # Added dropout
            layers.Dense(2, activation='softmax')  # 2 classes
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✓ Model built successfully")
        
    def train(self, epochs=20):
        print("\n🚀 Starting training process...")
        X_train, X_test, y_train, y_test = self.load_data()
        
        if self.model is None:
            self.build_model()
        
        # Use TqdmCallback for training progress
        print(f"\n📈 Training for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            batch_size=32,
            callbacks=[TqdmCallback(verbose=1)],
            verbose=0  # Disable default progress bar
        )
        
        print("\n📊 Evaluating model performance...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"📈 Final test accuracy: {test_accuracy:.4f}")
        
        return history
        
    def predict(self, image_path):
        print(f"\n🔍 Predicting image: {Path(image_path).name}")
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, self.image_size)
        img = img / 255.0
        
        prediction = self.model.predict(np.array([img]))
        predicted_class = self.class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        print(f"✓ Prediction complete")
        return predicted_class, confidence

if __name__ == "__main__":
    print("\n🍄 Starting Mushroom Classification CNN\n" + "="*40)
    # Create and train the model
    cnn = BinaryMushroomCNN()
    history = cnn.train(epochs=20)
    
    print("\n🧪 Testing model with sample images...")
    base_path = Path("/Users/celineotten/Documents/Git/SmallLanguageModels/Webscraper/data/images_mushrooms")
    
    # Test with one image from each class
    test_images = [
        base_path / "Amanita_muscaria" / "some_image.jpg",  # Replace with actual image name
        base_path / "Armillaria_mellea" / "some_image.jpg"   # Replace with actual image name
    ]
    
    for image_path in test_images:
        if image_path.exists():
            predicted_class, confidence = cnn.predict(image_path)
            print(f"\nImage: {image_path.name}")
            print(f"Predicted: {predicted_class} with confidence: {confidence:.2f}")
