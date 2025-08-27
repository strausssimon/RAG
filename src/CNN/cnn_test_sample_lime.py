"""
====================================================
Programmname : Model Test
Beschreibung : Skript zum Test des CNN-Modells zur Pilzklassifikation eines einzelnen Bildes inkl. Lime.
Vorbereitung : Einzelnes Bild aus data\inaturalist_samples in den Ordner src\CNN legen

====================================================
"""
import numpy as np
import tensorflow as tf
import sys
import os
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

modell_pfad = os.path.join(os.path.dirname(__file__), "..", "..", "models", "mushroom_resnet50_transfer_80_20.keras")#mushroom_5class_resnet_cnn_80_20.keras")

# === Pilzklassen (entsprechend Modell-Ausgabe) ===
pilzklassen = [
    "Phallus_impudicus", "Amanita_muscaria", "Boletus_edulis", "Cantharellus_cibarius", "Armillaria_mellea"
]

def bereite_bild_vor(pfad_zum_bild):
    with open(pfad_zum_bild, 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Kann Bild nicht laden: {pfad_zum_bild}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (200, 200))
    img_normalized = img_resized.astype(np.float32) / 255.0
    x = np.expand_dims(img_normalized, axis=0)
    return x, img_resized  # Rückgabe auch Original-200x200-Bild für LIME

def finde_bild_im_ordner(ordner_pfad):
    bild_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    bilder = [datei for datei in os.listdir(ordner_pfad) if any(datei.lower().endswith(ext) for ext in bild_extensions)]
    bilder.sort()
    if len(bilder) == 0:
        return None
    return [os.path.join(ordner_pfad, b) for b in bilder]

if __name__ == "__main__":
    ordner_pfad = os.path.dirname(__file__)
    bilder = finde_bild_im_ordner(ordner_pfad)

    if not bilder or len(bilder) == 0:
        print(f"Keine Bilddatei im Ordner gefunden: {ordner_pfad}")
        sys.exit(1)

    print("Lade Modell...")
    modell = tf.keras.models.load_model(modell_pfad)

    # Nimm erstes Bild für LIME-Analyse
    bild_pfad = bilder[0]
    print(f"Analysiere Bild mit LIME: {bild_pfad}")
    eingabe, img_200 = bereite_bild_vor(bild_pfad)

    vorhersage = modell.predict(eingabe)
    klasse_index = np.argmax(vorhersage)
    erkannter_pilz = pilzklassen[klasse_index]
    print("\nVorhersage:")
    for i, klasse in enumerate(pilzklassen):
        print(f"  {klasse:25s}: {vorhersage[0][i]:.4f}")
    print(f"==> Erkannter Pilz: {erkannter_pilz}")

    # ==== LIME Erklärung ====
    def predict_fn(images):
        images = np.array(images)
        return modell.predict(images)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_200,                 # Originalbild (200x200 RGB)
        predict_fn,              # Vorhersagefunktion
        top_labels=len(pilzklassen),            # nur Top-Klasse erklären
        hide_color=0,
        num_samples=1000         # Anzahl Samples für Approximation
    )

        # Falls die vorhergesagte Klasse nicht in explanation enthalten ist → fallback
    if klasse_index not in explanation.top_labels:
        print(f"Warnung: Klasse {klasse_index} nicht in LIME-Labels enthalten. Nutze Top-Klasse {explanation.top_labels[0]}")
        klasse_index = explanation.top_labels[0]

    # Maske für Top-Klasse
    temp, mask = explanation.get_image_and_mask(
        klasse_index,
        positive_only=False,
        num_features=10,
        hide_rest=False
    )

# Visualisierung
plt.figure(figsize=(6,6))
plt.imshow(temp)  # Originalbild als Basis
plt.imshow(mask, alpha=0.5)  # Heatmap drüberlegen
#plt.imshow(mask, cmap='bwr', alpha=0.5)  # Heatmap drüberlegen
plt.title(f"LIME-Erklärung für: {erkannter_pilz}")
plt.axis("off")
plt.show()