"""
====================================================
Programmname : CNN: ResNet
Datum        : 24.08.2025
Version      : 1.0
Beschreibung : CNN-Modell basierend auf der ResNet-Architektur für die Bildklassifikation.

====================================================
"""
# TensorFlow 2.19.1 & Keras 3.11.2 Konfiguration - EXKLUSIV
import os
# Moderne TensorFlow/Keras 3 Konfiguration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # OneDNN Optimierungen falls problematisch

# EXKLUSIV Keras 3.11.2 - Keine Fallbacks
try:
    import keras
    from keras import layers, models, optimizers, callbacks
    import tensorflow as tf
    
    # Überprüfe Keras-Version - nur 3.x erlaubt
    keras_version = keras.__version__
    major_version = int(keras_version.split('.')[0])
    
    if major_version < 3:
        raise ImportError(f"Keras {keras_version} ist nicht unterstützt. Mindestens Keras 3.x erforderlich!")
    
    print(f"Keras 3.x erfolgreich geladen (Version: {keras.__version__})")
    print(f"TensorFlow erfolgreich geladen (Version: {tf.__version__})")
    
except ImportError as e:
    print(f"ERROR: KRITISCHER FEHLER: Keras 3.x ist erforderlich!")
    print(f"Fehlerdetails: {e}")
    print(f"HINWEIS:Installieren Sie Keras 3.x mit: pip install keras>=3.11.2")
    print("ERROR: Script wird beendet - Keras 3.x ist zwingend erforderlich!")
    exit(1)
except Exception as e:
    print(f"ERROR: UNERWARTETER FEHLER beim Keras 3.x Import: {e}")
    print("ERROR: Script wird beendet!")
    exit(1)

import numpy as np
from pathlib import Path
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
from collections import defaultdict, Counter
import sys

# Log-Ausgabe in Datei UND Terminal
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

logfile = open("cnn_resnet50_output_log.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, logfile)
sys.stderr = Tee(sys.stderr, logfile)

# Verwendete Datenpfade in diesem Skript:
# - data/randomized_mushrooms/inaturalist: Pfad zu den randomisierten Bilddaten für 80/20 Train/Test Split
# - best_mushroom_model.keras, mushroom_4class_cnn_external_test.keras, mushroom_cnn_model.keras: Speicherorte für trainierte Modelle

class MushroomCNN:
    def build_resnet50_transfer_model(self, trainable_layers=30):
        """
        Erstellt ein Transfer-Learning-Modell auf Basis von ResNet50 (ImageNet), angepasst auf self.image_size und self.num_classes.
        trainable_layers: Anzahl der letzten ResNet50-Layer, die mittrainiert werden (Rest bleibt eingefroren)
        """
        print("\nBaue Transfer-Learning Modell mit ResNet50 (ImageNet)...")
        from keras.applications import ResNet50
        from keras.applications.resnet import preprocess_input
        # Basis-Modell (ohne Top, mit ImageNet-Gewichten)
        base_model = ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(self.image_size[0], self.image_size[1], 3),
            pooling="avg"
        )
        # Nur die letzten trainable_layers Layer trainieren
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
        inputs = layers.Input(shape=(self.image_size[0], self.image_size[1], 3), name="input_image")
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0005),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.model = model
        print("ResNet50-Transfermodell erstellt und kompiliert!")
    def __init__(self, image_size=(200, 200)):
        self.image_size = image_size
        self.model = None
        self.class_names = ["Phallus_impudicus", "Amanita_muscaria", "Boletus_edulis", "Cantharellus_cibarius", "Armillaria_mellea"]
        self.num_classes = len(self.class_names)
        
    def extract_base_number(self, filename):
        """
        Extrahiert die Basisnummer aus dem Dateinamen.
        Beispiel: "Armillaria_mellea_0_zoom_110.jpg" -> "0"
        Beispiel: "Boletus_edulis_42_bright_plus.jpg" -> "42"
        """
        match = re.search(r'_(\d+)_', filename)
        if match:
            return match.group(1)
        return None
        
    def load_data_with_split(self):
        """
        Lädt alle Daten aus data/randomized_mushrooms/inaturalist und macht einen 80/20 Train/Test Split
        """
        print("\nLade randomisierte Pilzdaten für 80/20 Train/Test Split...")
        # Pfad zu den randomisierten Bilddaten
        # data_path = Path("data/randomized_mushrooms/inaturalist")
        data_path = Path("data/resized_mushrooms/inaturalist")

        if not data_path.exists():
            print(f"Fehler: Randomisierte Daten nicht gefunden in {data_path}")
            return None, None, None, None, None, None

        print(f"Lade Daten aus: {data_path.absolute()}")

        print(f"Konfigurierte Klassen: {self.class_names}")
        print(f"Anzahl Klassen: {self.num_classes}")

        print("\n1. Sammle alle Bilder aus allen Klassen...")
        
        all_files = []
        all_labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = data_path / class_name
            if not class_path.exists():
                print(f"   Warnung: Klasse {class_name} nicht gefunden!")
                continue
                
            # Alle Bilddateien finden (ohne Duplikate)
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            
            for file_path in class_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
            
            print(f"   {class_name}: {len(image_files)} Dateien gefunden")
            
            # Alle Dateien zur Gesamtliste hinzufügen
            for img_file in image_files:
                all_files.append(img_file)
                all_labels.append(class_idx)
        
        print(f"\n   Gesamt gesammelt: {len(all_files)} Dateien")

        # 80/20 Split der Dateilisten (stratifiziert nach Klassen)
        print("\n2. Erstelle 80/20 Train/Test Split...")
        train_files, test_files, train_labels, test_labels = train_test_split(
            all_files, 
            all_labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=all_labels  # Sicherstellt gleiche Klassenverteilung in Train/Test
        )
        
        print(f"   Training: {len(train_files)} Dateien ({len(train_files)/len(all_files)*100:.1f}%)")
        print(f"   Test: {len(test_files)} Dateien ({len(test_files)/len(all_files)*100:.1f}%)")

        # Zeige Klassenverteilung für Train/Test
        print("\n   Klassenverteilung im Training:")
        train_counter = Counter(train_labels)
        for i, class_name in enumerate(self.class_names):
            count = train_counter.get(i, 0)
            print(f"     {class_name}: {count} Bilder")
            
        print("\n   Klassenverteilung im Test:")
        test_counter = Counter(test_labels)
        for i, class_name in enumerate(self.class_names):
            count = test_counter.get(i, 0)
            print(f"     {class_name}: {count} Bilder")

        # Lädt die tatsächlichen Bilddaten
        print("\n3. Lade Bilddaten...")
        def load_images(file_list, label_list, desc):
            images = []
            labels = []
            # Lädt und normalisiert alle Bilder aus der übergebenen Dateiliste
            for img_file, label in tqdm(zip(file_list, label_list), desc=desc, total=len(file_list)):
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.resize(img, self.image_size)
                        img = img / 255.0  # Normalisierung auf Wertebereich 0-1
                        images.append(img)
                        labels.append(label)
                    else:
                        print(f"   Warnung: Kann {img_file.name} nicht laden")
                except Exception as e:
                    print(f"   Fehler bei {img_file.name}: {e}")
            return np.array(images), np.array(labels)
        
        X_train, y_train = load_images(train_files, train_labels, "Loading training images")
        X_test, y_test = load_images(test_files, test_labels, "Loading test images")
        
        if len(X_train) == 0 or len(X_test) == 0:
            print("Fehler: Keine Bilder erfolgreich geladen!")
            return None, None, None, None, None, None
            
        # Labels werden in One-Hot-Encoding umgewandelt
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        # Berechnet die Klassengewichte für das Training
        from sklearn.utils.class_weight import compute_class_weight
        unique_classes = np.unique(np.argmax(y_train, axis=1))
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=np.argmax(y_train, axis=1)
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"\n=== FINALE STATISTIKEN ===")
        print(f"Training shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")
        print(f"Klassengewichte: {class_weight_dict}")
        
        return X_train, y_train, X_test, y_test, class_weight_dict

    def build_model(self):
        print("\nBuilding 7-class ResNet-inspired CNN model for mushroom classification (Keras 3.x EXKLUSIV)...")
        
        # Keras 3.x Functional API
        inputs = layers.Input(shape=(200, 200, 3), name='input_image')
        
        # Initial Convolution Block
        x = layers.Conv2D(32, (7, 7), strides=2, padding='same', activation='relu', name='initial_conv')(inputs)
        x = layers.BatchNormalization(name='initial_bn')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='initial_pool')(x)
        
        # Residual Block 1 (32 filters)
        residual1 = x
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='res1_conv1')(x)
        x = layers.BatchNormalization(name='res1_bn1')(x)
        x = layers.Dropout(0.3, name='res1_dropout1')(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='res1_conv2')(x)
        x = layers.BatchNormalization(name='res1_bn2')(x)
        # Skip connection
        x = layers.Add(name='res1_add')([x, residual1])
        x = layers.Activation('relu', name='res1_activation')(x)
        x = layers.MaxPooling2D((2, 2), name='res1_pool')(x)
        
        # Residual Block 2 (64 filters) - with projection shortcut
        residual2 = layers.Conv2D(64, (1, 1), padding='same', name='res2_projection')(x)
        residual2 = layers.BatchNormalization(name='res2_proj_bn')(residual2)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='res2_conv1')(x)
        x = layers.BatchNormalization(name='res2_bn1')(x)
        x = layers.Dropout(0.3, name='res2_dropout1')(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='res2_conv2')(x)
        x = layers.BatchNormalization(name='res2_bn2')(x)
        # Skip connection with projection
        x = layers.Add(name='res2_add')([x, residual2])
        x = layers.Activation('relu', name='res2_activation')(x)
        x = layers.MaxPooling2D((2, 2), name='res2_pool')(x)
        
        # Residual Block 3 (128 filters) - with projection shortcut
        residual3 = layers.Conv2D(128, (1, 1), padding='same', name='res3_projection')(x)
        residual3 = layers.BatchNormalization(name='res3_proj_bn')(residual3)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='res3_conv1')(x)
        x = layers.BatchNormalization(name='res3_bn1')(x)
        x = layers.Dropout(0.3, name='res3_dropout1')(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='res3_conv2')(x)
        x = layers.BatchNormalization(name='res3_bn2')(x)
        # Skip connection with projection
        x = layers.Add(name='res3_add')([x, residual3])
        x = layers.Activation('relu', name='res3_activation')(x)
        x = layers.MaxPooling2D((2, 2), name='res3_pool')(x)
        
        # Residual Block 4 (256 filters) - with projection shortcut
        residual4 = layers.Conv2D(256, (1, 1), padding='same', name='res4_projection')(x)
        residual4 = layers.BatchNormalization(name='res4_proj_bn')(residual4)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='res4_conv1')(x)
        x = layers.BatchNormalization(name='res4_bn1')(x)
        x = layers.Dropout(0.3, name='res4_dropout1')(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='res4_conv2')(x)
        x = layers.BatchNormalization(name='res4_bn2')(x)
        # Skip connection with projection
        x = layers.Add(name='res4_add')([x, residual4])
        x = layers.Activation('relu', name='res4_activation')(x)
        
        # GlobalAveragePooling2D statt Flatten - reduziert Parameter drastisch
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Classifier mit deutlich weniger Parametern
        x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='dense1')(x)
        x = layers.BatchNormalization(name='classifier_bn')(x)
        x = layers.Dropout(0.5, name='classifier_dropout')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Modell mit Keras 3.x Functional API erstellen
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='mushroom_resnet_cnn')
        
        # Keras 3.x Optimizer mit reduzierter Lernrate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"ResNet-inspiriertes Modell erfolgreich erstellt für {self.num_classes} Klassen mit Keras 3.x Functional API")
        print("Verwendet: Residual Connections + GlobalAveragePooling2D + reduzierte Parameter")
        
    def train(self, epochs=30):
        print("\nStarting training process with 80/20 split from randomized data...")
        data_result = self.load_data_with_split()
        
        if data_result is None or any(x is None for x in data_result):
            print("Fehler: Keine Daten verfügbar!")
            return None
            
        X_train, y_train, X_test, y_test, class_weights = data_result
        
        print(f"\n--- Verwende 80/20 Split aus data/randomized_mushrooms/inaturalist ---")
        print(f"Training: {X_train.shape[0]} Bilder")
        print(f"Test: {X_test.shape[0]} Bilder")
        
        if self.model is None:
            self.build_model()
        
        # Keras 3.x Callbacks für Training
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            min_delta=0.005
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            'best_mushroom_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        print(f"\nTraining for max {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            batch_size=32,
            class_weight=class_weights,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=2
        )
        
        print("\nEvaluating model performance...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Final test accuracy: {test_accuracy:.4f}")
        
        # Detaillierte Vorhersageanalyse
        predictions = self.model.predict(X_test, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Klassenweise Accuracy
        print(f"\n=== KLASSENWEISE PERFORMANCE ===")
        for i, class_name in enumerate(self.class_names):
            class_mask = (true_classes == i)
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(pred_classes[class_mask] == true_classes[class_mask])
                print(f"{class_name}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(true_classes, pred_classes)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        print(f"\nClassification Report:")
        print(classification_report(true_classes, pred_classes, target_names=self.class_names))
        
        return history
        
    def predict(self, image_path):
        """Vorhersage für ein einzelnes Bild"""
        if self.model is None:
            print("Fehler: Modell nicht trainiert!")
            return None, None
            
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Fehler: Kann Bild {image_path} nicht laden")
            return None, None
            
        img = cv2.resize(img, self.image_size)
        img = img / 255.0
        
        prediction = self.model.predict(np.array([img]), verbose=0)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = np.max(prediction)
        
        return predicted_class, confidence
    
    def save_model(self, filepath="mushroom_resnet50_transfer.keras"):
        """Speichert das trainierte Modell im modernen .keras Format (Keras 3.x kompatibel)"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Modell gespeichert im Keras 3.x Format: {filepath}")
            
            # Erstelle auch .h5 Version für Rückwärtskompatibilität
            if filepath.endswith('.keras'):
                h5_filepath = filepath.replace('.keras', '.h5')
                try:
                    self.model.save(h5_filepath)
                    print(f"Rückwärtskompatible .h5 Version erstellt: {h5_filepath}")
                except Exception as e:
                    print(f".h5 Version konnte nicht erstellt werden: {e}")
        else:
            print("ERROR: Fehler: Kein Modell zum Speichern vorhanden!")
    
    def load_model(self, filepath="mushroom_resnet50_transfer.keras"):
        """Lädt ein gespeichertes Modell (Keras 3.x EXKLUSIV)"""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"Modell erfolgreich geladen mit Keras 3.x: {filepath}")
        except Exception as e:
            print(f"ERROR: Fehler beim Laden des Modells: {e}")

if __name__ == "__main__":
    print("\nMUSHROOM CLASSIFICATION CNN - 5- CLASSES + RESIDUAL CONNECTIONS (TensorFlow 2.19.1 & Keras 3.x EXKLUSIV)")
    print("=" * 90)
    print("\nMUSHROOM CLASSIFICATION CNN - RESNET50 TRANSFER LEARNING (TensorFlow 2.19.1 & Keras 3.x EXKLUSIV)")
    print("=" * 100)
    print("Classes: Amanita_muscaria, Boletus_edulis, Armillaria_mellea, Phallus_impudicus, Cantharellus_cibarius")
    print("Architecture: ECHTES ResNet50 (ImageNet Pretrained) + Transfer Learning")
    print("Improvements: Fine-Tuning der letzten Layer, Dense-Top, Dropout, Adam Optimizer")
    print("Data: 80/20 split from data/randomized_mushrooms/inaturalist")
    print("Training: Randomized 200x200 images (80%)")
    print("Testing: Randomized 200x200 images (20%)")
    print("Output: Modern .keras format (with .h5 fallback for compatibility)")
    print("KERAS 3.x ERFORDERLICH - Keine Fallbacks!")
    print("=" * 100)

    # Modell erstellen
    cnn = MushroomCNN()
    cnn.build_resnet50_transfer_model(trainable_layers=30)

    # Trainieren
    history = cnn.train(epochs=30)

    if history is not None:
        # Speichern
        cnn.save_model("models/mushroom_resnet50_transfer_80_20.keras")
        print("\nTraining erfolgreich abgeschlossen mit Keras 3.x!")
        print("Echtes ResNet50-Transfermodell gespeichert im modernen .keras Format")
    else:
        print("\nERROR: Training fehlgeschlagen!")

    # Erstellt und trainiert das Modell mit Keras 3.x und 80/20 Split
    cnn = MushroomCNN()
    history = cnn.train(epochs=30)

    if history is not None:
        # Speichert das trainierte Modell im modernen .keras Format + .h5 Fallback
        cnn.save_model("models/mushroom_5class_resnet_cnn_80_20_split.keras")
        print("\nTraining erfolgreich abgeschlossen mit Keras 3.x!")
        print("ResNet-inspiriertes 5-Klassen Modell mit 80/20 Split gespeichert im modernen .keras Format")
    else:
        print("\nERROR: Training fehlgeschlagen!")
