import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
from collections import defaultdict, Counter

# Verwendete Datenpfade in diesem Skript:
# - data/augmented_mushrooms/resized: Pfad zu den augmentierten und auf 200x200 Pixel skalierten Trainings- und Testbildern
# - data/test_mushrooms: Pfad zu externen Testbildern
# - best_mushroom_model.h5, mushroom_4class_cnn_external_test.h5, mushroom_cnn_model.h5: Speicherorte für trainierte Modelle

class MushroomCNN:
    def __init__(self, image_size=(200, 200)):
        self.image_size = image_size
        self.model = None
        self.class_names = ["Amanita_phalloides", "Armillaria_mellea", "Boletus_edulis", "Cantharellus_cibarius"]
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
        
    def load_data(self):
        print("\nLade augmentierte Pilzdaten mit intelligenter Train/Test-Aufteilung...")
        # Pfad zu den augmentierten und skalierten Bilddaten
        data_path = Path("data/augmented_mushrooms/resized")

        if not data_path.exists():
            print(f"Fehler: Augmentierte Daten nicht gefunden in {data_path}")
            return None, None, None, None, None

        print(f"Lade Daten aus: {data_path.absolute()}")

        # Gruppiert alle Bilder nach Basisnummer und Klasse
        image_groups = defaultdict(lambda: defaultdict(list))  # {class: {base_number: [files]}}

        print("\n1. Analysiere Dateistruktur...")
        
        for class_name in self.class_names:
            class_path = data_path / class_name
            if not class_path.exists():
                print(f"   Warnung: Klasse {class_name} nicht gefunden!")
                continue
                
            # Alle Bilddateien finden
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(class_path.glob(ext)))
                image_files.extend(list(class_path.glob(ext.upper())))
            
            print(f"   {class_name}: {len(image_files)} Dateien gefunden")
            
            # Gruppiere nach Basisnummer
            for img_file in image_files:
                base_number = self.extract_base_number(img_file.name)
                if base_number is not None:
                    image_groups[class_name][base_number].append(img_file)
                else:
                    # Fallback für Dateien ohne erkennbare Nummer
                    print(f"   Warnung: Keine Nummer erkannt in {img_file.name}")
        
        # Statistiken über Gruppierung
        print("\n2. Gruppierungsstatistiken:")  # Ausgabe der Gruppierungsstatistiken
        total_base_images = 0
        for class_name in self.class_names:
            num_groups = len(image_groups[class_name])
            total_base_images += num_groups
            if num_groups > 0:
                avg_augmentations = np.mean([len(files) for files in image_groups[class_name].values()])
                print(f"   {class_name}: {num_groups} Basis-Bilder, {avg_augmentations:.1f} Augmentationen pro Bild")
        print(f"   Gesamt: {total_base_images} verschiedene Basis-Bilder")

        # Intelligente 80/20-Aufteilung der Basisnummern in Trainings- und Testdaten
        print("\n3. Erstelle intelligente 80/20 Aufteilung...")
        train_files = []
        test_files = []
        train_labels = []
        test_labels = []
        for class_idx, class_name in enumerate(self.class_names):
            if class_name not in image_groups or len(image_groups[class_name]) == 0:
                print(f"   Überspringe {class_name} - keine Daten")
                continue
            # Alle Basisnummern für diese Klasse
            base_numbers = list(image_groups[class_name].keys())
            base_numbers.sort(key=int)  # Numerisch sortieren
            # 80/20 Split der Basisnummern
            split_point = int(len(base_numbers) * 0.8)
            train_base_numbers = base_numbers[:split_point]
            test_base_numbers = base_numbers[split_point:]
            print(f"   {class_name}: {len(train_base_numbers)} Basis-Bilder für Training, {len(test_base_numbers)} für Test")
            # Alle Dateien der Train-Basisnummern zum Training hinzufügen
            for base_num in train_base_numbers:
                for img_file in image_groups[class_name][base_num]:
                    train_files.append(img_file)
                    train_labels.append(class_idx)
            # Alle Dateien der Test-Basisnummern zum Test hinzufügen
            for base_num in test_base_numbers:
                for img_file in image_groups[class_name][base_num]:
                    test_files.append(img_file)
                    test_labels.append(class_idx)
        print(f"\n   Training: {len(train_files)} Dateien")
        print(f"   Test: {len(test_files)} Dateien")
        print(f"   Verhältnis: {len(train_files)/(len(train_files)+len(test_files))*100:.1f}% Training")

        # Lädt die tatsächlichen Bilddaten
        print("\n4. Lade Bilddaten...")
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
            return None, None, None, None, None
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
        # Gibt die Verteilung der Klassen im Trainings- und Testdatensatz aus
        train_distribution = Counter(np.argmax(y_train, axis=1))
        test_distribution = Counter(np.argmax(y_test, axis=1))
        print(f"\n=== FINALE STATISTIKEN ===")
        print(f"Training shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")
        print("Training Klassenverteilung:")
        for i, class_name in enumerate(self.class_names):
            count = train_distribution.get(i, 0)
            print(f"   {class_name}: {count} Bilder")
        print("Test Klassenverteilung:")
        for i, class_name in enumerate(self.class_names):
            count = test_distribution.get(i, 0)
            print(f"   {class_name}: {count} Bilder")
        print(f"Klassengewichte: {class_weight_dict}")
        return X_train, X_test, y_train, y_test, class_weight_dict
    
    def load_external_test_data(self):
        """
        Lädt externe Testdaten aus dem Verzeichnis data/test_mushrooms.
        Die Klassenzuordnung erfolgt anhand des Dateinamens.
        """
        print("\nLade externe Testdaten aus data/test_mushrooms...")

        test_path = Path("data/test_mushrooms")

        if not test_path.exists():
            print(f"Fehler: Test-Ordner {test_path} existiert nicht!")
            return None, None

        print(f"Lade Testdaten aus: {test_path.absolute()}")

        # Sammelt alle Testbilder und ordnet sie den Klassen zu
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        test_files = []
        test_labels = []

        for file_path in test_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                filename = file_path.name.lower()
                class_idx = None
                for i, class_name in enumerate(self.class_names):
                    class_name_lower = class_name.lower()
                    if filename.startswith(class_name_lower):
                        class_idx = i
                        break
                if class_idx is not None:
                    test_files.append(file_path)
                    test_labels.append(class_idx)
                else:
                    print(f"   Warnung: Kann Klasse für {filename} nicht bestimmen")

        if not test_files:
            print("Keine gültigen Testbilder gefunden!")
            return None, None

        print(f"Gefunden: {len(test_files)} Testbilder")

        # Gibt die Verteilung der Klassen im Testdatensatz aus
        from collections import Counter
        class_distribution = Counter(test_labels)
        print("Test Klassenverteilung:")
        for i, class_name in enumerate(self.class_names):
            count = class_distribution.get(i, 0)
            print(f"   {class_name}: {count} Bilder")

        # Lädt und normalisiert die Testbilder
        print("\nLade Test-Bilddaten...")

        images = []
        labels = []

        for img_file, label in tqdm(zip(test_files, test_labels), desc="Loading test images", total=len(test_files)):
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

        if len(images) == 0:
            print("Fehler: Keine Testbilder erfolgreich geladen!")
            return None, None

        X_test = np.array(images)
        y_test = tf.keras.utils.to_categorical(np.array(labels), self.num_classes)

        print(f"Erfolgreich geladen: {X_test.shape}")

        return X_test, y_test
    
    def build_model(self):
        print("\nBuilding 4-class CNN model for mushroom classification...")
        self.model = models.Sequential([
            # Erste Convolution Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Zweiter Convolution Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dritter Convolution Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Vierter Convolution Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Optimizer mit angepasster Learning Rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"Model built successfully for {self.num_classes} classes")
        
    def train(self, epochs=30, use_external_test=True):
        print("\nStarting training process...")
        data_result = self.load_data()
        
        if data_result is None or any(x is None for x in data_result):
            print("Fehler: Keine Trainingsdaten verfügbar!")
            return None
            
        X_train, X_test_internal, y_train, y_test_internal, class_weights = data_result
        
        # Entscheide ob externe oder interne Testdaten verwendet werden sollen
        if use_external_test:
            print("\n--- Verwende externe Testdaten aus test_mushrooms ---")
            external_test_result = self.load_external_test_data()
            if external_test_result[0] is not None:
                X_test, y_test = external_test_result
                print("Externe Testdaten erfolgreich geladen!")
            else:
                print("Fallback zu internen Testdaten...")
                X_test, y_test = X_test_internal, y_test_internal
        else:
            print("\n--- Verwende interne Testdaten (80/20 Split) ---")
            X_test, y_test = X_test_internal, y_test_internal
        
        if self.model is None:
            self.build_model()
        
        # Callbacks für Training
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            min_delta=0.005
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_mushroom_model.h5',
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
            verbose=1
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
    
    def save_model(self, filepath="mushroom_cnn_model.h5"):
        """Speichert das trainierte Modell"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Modell gespeichert: {filepath}")
        else:
            print("Fehler: Kein Modell zum Speichern vorhanden!")
    
    def load_model(self, filepath="mushroom_cnn_model.h5"):
        """Lädt ein gespeichertes Modell"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Modell geladen: {filepath}")
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")

if __name__ == "__main__":
    print("\nMUSHROOM CLASSIFICATION CNN - 4 CLASSES")
    print("=" * 60)
    print("Classes: Amanita_phalloides, Armillaria_mellea, Boletus_edulis, Cantharellus_cibarius")
    print("Training: Augmented 200x200 images")
    print("Testing: External test data from data/test_mushrooms")
    print("=" * 60)

    # Erstellt und trainiert das Modell
    cnn = MushroomCNN()
    history = cnn.train(epochs=30, use_external_test=True)

    if history is not None:
        # Speichert das trainierte Modell
        cnn.save_model("models/mushroom_4class_cnn_external_test.h5")
        print("\nTraining abgeschlossen! Modell gespeichert.")
    else:
        print("\nTraining fehlgeschlagen!")
