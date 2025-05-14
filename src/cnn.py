import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

class MushroomCNN:
    def __init__(self, image_dir, image_size=(224, 224)):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.model = None
        self.class_names = None
        
    def load_data(self):
        images = []
        labels = []
        
        for img_path in self.image_dir.glob("*.jpg"):
            # Load and preprocess image
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, self.image_size)
            img = img / 255.0  # Normalize
            images.append(img)
            
            # Get label from filename
            label = img_path.stem.replace('_', ' ')
            labels.append(label)
            
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Get unique class names
        self.class_names = np.unique(y)
        
        # Convert labels to categorical
        y = pd.get_dummies(y).values
        
        # Split data
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
    def build_model(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, epochs=10):
        X_train, X_test, y_train, y_test = self.load_data()
        
        if self.model is None:
            self.build_model()
            
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_test, y_test)
        )
        
        return history
        
    def predict(self, image_path):
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, self.image_size)
        img = img / 255.0
        
        prediction = self.model.predict(np.array([img]))
        predicted_class = self.class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        return predicted_class, confidence

if __name__ == "__main__":
    cnn = MushroomCNN("../data/images")
    history = cnn.train(epochs=10)
    
    # Test prediction
    test_image = "../data/images/Radulomyces_molaris.jpg"
    predicted_class, confidence = cnn.predict(test_image)
    print(f"Predicted: {predicted_class} with confidence: {confidence:.2f}")