"""
====================================================
Programmname : Keras Model Tester
Beschreibung : Testet ein gespeichertes Keras-Modell mit unabhängigen Testdaten.
               Speziell für das 5-Klassen Pilzklassifikationsmodell.
               Zweck: Performance-Evaluation auf echten, ungesehenen Testdaten
====================================================
"""

import os
# Moderne TensorFlow/Keras 3 Konfiguration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# EXKLUSIV Keras 3.11.2 - Keine Fallbacks
try:
    import keras
    import tensorflow as tf
    
    # Überprüfe Keras-Version - nur 3.x erlaubt
    keras_version = keras.__version__
    major_version = int(keras_version.split('.')[0])
    
    if major_version < 3:
        raise ImportError(f"Keras {keras_version} ist nicht unterstützt. Mindestens Keras 3.x erforderlich!")
    
    print(f"Keras 3.x erfolgreich geladen (Version: {keras.__version__})")
    print(f"TensorFlow erfolgreich geladen (Version: {tf.__version__})")
    
except ImportError as e:
    print(f"KRITISCHER FEHLER: Keras 3.x ist erforderlich!")
    print(f"Fehlerdetails: {e}")
    print(f"Installieren Sie Keras 3.x mit: pip install keras>=3.11.2")
    exit(1)

import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTester:
    def __init__(self, model_path="models/mushroom_resnet50_transfer_80_20_2.keras"):
        self.model_path = Path(model_path)
        self.test_data_path = Path("data/test_mushrooms/test_mushrooms")
        self.image_size = (200, 200)
        
        # 5 Hauptklassen für den Test
        self.class_names = [
            
            "Phallus_impudicus", "Amanita_muscaria", "Boletus_edulis", "Cantharellus_cibarius", "Armillaria_mellea"
        ]
        self.num_classes = len(self.class_names)
        
        # Unterstützte Bildformate
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        self.model = None
        
    def load_model(self):
        """Lädt das gespeicherte Keras-Modell"""
        print(f"\nLade Modell von: {self.model_path}")
        
        if not self.model_path.exists():
            print(f"Fehler: Modell {self.model_path} existiert nicht!")
            return False
            
        try:
            self.model = keras.models.load_model(str(self.model_path))
            print(f"Modell erfolgreich geladen!")
            
            # Modell-Informationen anzeigen
            print(f"Modell-Architektur:")
            print(f"Input Shape: {self.model.input_shape}")
            print(f"Output Shape: {self.model.output_shape}")
            print(f"Total Parameters: {self.model.count_params():,}")
            
            return True
            
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")
            return False
    
    def load_test_data(self):
        """Lädt die Testdaten aus data/test_mushrooms_randomized/inaturalist"""
        print(f"\nLade Testdaten von: {self.test_data_path}")
        
        if not self.test_data_path.exists():
            print(f"Fehler: Testdaten-Verzeichnis {self.test_data_path} existiert nicht!")
            return None, None
        
        test_files = []
        test_labels = []
        
        print(f"🔍 Suche nach Testbildern in {len(self.class_names)} Klassen...")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.test_data_path / class_name
            
            if not class_dir.exists():
                print(f"Warnung: Klasse {class_name} nicht gefunden!")
                continue
            
            # Sammle alle Bilddateien
            class_files = []
            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                    class_files.append(file_path)
            
            print(f"{class_name}: {len(class_files)} Testbilder gefunden")
            
            # Füge zur Gesamtliste hinzu
            for img_file in class_files:
                test_files.append(img_file)
                test_labels.append(class_idx)
        
        print(f"\nGesamt: {len(test_files)} Testbilder aus {len(set(test_labels))} Klassen")
        
        # Zeige Klassenverteilung
        label_counter = Counter(test_labels)
        print(f"\nKlassenverteilung in Testdaten:")
        for i, class_name in enumerate(self.class_names):
            count = label_counter.get(i, 0)
            percentage = (count / len(test_files)) * 100 if test_files else 0
            print(f"{class_name}: {count} Bilder ({percentage:.1f}%)")
        
        if not test_files:
            print("Keine Testbilder gefunden!")
            return None, None
        
        # Lade und verarbeite Bilder
        print(f"\nLade und verarbeite Bilder...")
        X_test = []
        y_test = []
        
        for img_file, label in tqdm(zip(test_files, test_labels), desc="Loading test images", total=len(test_files)):
            try:
                # Bild laden
                img = cv2.imread(str(img_file))
                if img is not None:
                    # Auf 200x200 skalieren
                    img_resized = cv2.resize(img, self.image_size)
                    # Normalisieren (0-1)
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    
                    X_test.append(img_normalized)
                    y_test.append(label)
                else:
                    print(f"Warnung: Kann {img_file.name} nicht laden")
                    
            except Exception as e:
                print(f"Fehler bei {img_file.name}: {e}")
        
        if not X_test:
            print("Keine Bilder konnten erfolgreich geladen werden!")
            return None, None
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"Testdaten erfolgreich geladen!")
        print(f"Shape: {X_test.shape}")
        print(f"Labels: {y_test.shape}")
        
        return X_test, y_test
    
    def evaluate_model(self, X_test, y_test):
        """Evaluiert das Modell auf den Testdaten"""
        print(f"\nEvaluiere Modell-Performance...")
        
        if self.model is None:
            print("Fehler: Kein Modell geladen!")
            return
        
        # Vorhersagen machen
        print("Mache Vorhersagen...")
        predictions = self.model.predict(X_test, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Gesamtgenauigkeit
        accuracy = accuracy_score(y_test, predicted_classes)
        print(f"\nGESAMTGENAUIGKEIT: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Klassenweise Performance
        print(f"\nKLASSENWEISE PERFORMANCE:")
        print("=" * 60)
        
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_test == i)
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(predicted_classes[class_mask] == y_test[class_mask])
                class_count = np.sum(class_mask)
                print(f"{class_name:20}: {class_accuracy:.4f} ({class_accuracy*100:.1f}%) - {class_count} samples")
            else:
                print(f"{class_name:20}: Keine Testdaten")
        
        # Confusion Matrix
        print(f"\nCONFUSION MATRIX:")
        print("=" * 40)
        cm = confusion_matrix(y_test, predicted_classes)
        print(cm)
        
        # Classification Report
        print(f"\nCLASSIFICATION REPORT:")
        print("=" * 50)
        report = classification_report(y_test, predicted_classes, target_names=self.class_names)
        print(report)
        
        # Speichere Ergebnisse
        self.save_results(accuracy, cm, report, y_test, predicted_classes)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': y_test
        }
    
    def save_results(self, accuracy, cm, report, y_true, y_pred):
        """Speichert die Testergebnisse in einer Datei"""
        results_file = "test_results_mushroom_model.txt"
        
        print(f"\nSpeichere Ergebnisse in: {results_file}")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("MUSHROOM MODEL TEST RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Modell: {self.model_path}\n")
            f.write(f"Testdaten: {self.test_data_path}\n")
            f.write(f"Klassen: {', '.join(self.class_names)}\n")
            f.write(f"Anzahl Testbilder: {len(y_true)}\n\n")
            
            f.write(f"GESAMTGENAUIGKEIT: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            
            f.write("KLASSENWEISE PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            for i, class_name in enumerate(self.class_names):
                class_mask = (y_true == i)
                if np.sum(class_mask) > 0:
                    class_accuracy = np.mean(y_pred[class_mask] == y_true[class_mask])
                    class_count = np.sum(class_mask)
                    f.write(f"{class_name}: {class_accuracy:.4f} ({class_accuracy*100:.1f}%) - {class_count} samples\n")
            
            f.write(f"\nCONFUSION MATRIX:\n")
            f.write("-" * 20 + "\n")
            f.write(str(cm) + "\n\n")
            
            f.write(f"CLASSIFICATION REPORT:\n")
            f.write("-" * 30 + "\n")
            f.write(report)
        
        print(f"Ergebnisse gespeichert!")
    
    def run_test(self):
        """Führt den kompletten Test durch"""
        print("MUSHROOM MODEL TESTER")
        print("=" * 50)
        print(f"Modell: {self.model_path}")
        print(f"Testdaten: {self.test_data_path}")
        print(f"Klassen: {', '.join(self.class_names)}")
        print("=" * 50)
        
        # 1. Modell laden
        if not self.load_model():
            return False
        
        # 2. Testdaten laden
        X_test, y_test = self.load_test_data()
        if X_test is None:
            return False
        
        # 3. Modell evaluieren
        results = self.evaluate_model(X_test, y_test)
        
        print(f"\nTest erfolgreich abgeschlossen!")
        print(f"Gesamtgenauigkeit: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
        return True

def main():
    print("KERAS MODEL TESTER - Mushroom Classification")
    print("=" * 60)
    
    # Erstelle Tester-Instanz
    tester = ModelTester(
        model_path="models\mushroom_5class_resnet_cnn_80_20_split_2.keras" #"models/mushroom_resnet50_transfer_80_20.keras"
    )
    
    # Führe Test durch
    success = tester.run_test()
    
    if not success:
        print("\nTest fehlgeschlagen!")
        return

    print("\nTest erfolgreich abgeschlossen!")
    print("Detaillierte Ergebnisse in: test_results_mushroom_model.txt")

if __name__ == "__main__":
    main()
