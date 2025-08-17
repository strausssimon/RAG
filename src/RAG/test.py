import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import sys
import os

modell_pfad = os.path.join(os.path.dirname(__file__), "mushroom_4class_cnn_external_test.h5")

# === Pilzklassen (entsprechend Modell-Ausgabe) ===
pilzklassen = [
    "Gemeiner Steinpilz",
    "Hallimasch",
    # ... hier weitere Pilze ergänzen ...
]

def bereite_bild_vor(pfad_zum_bild):
    img = image.load_img(pfad_zum_bild, target_size=(224, 224))  # ggf. anpassen
    x = image.img_to_array(img)
    x = x / 255.0  # Normalisierung, je nach Modell
    x = np.expand_dims(x, axis=0)
    return x

def klassifiziere_pilz(modell, bild_pfad):
    eingabe = bereite_bild_vor(bild_pfad)
    vorhersage = modell.predict(eingabe)
    klasse_index = np.argmax(vorhersage)
    wahrscheinlichkeit = vorhersage[0][klasse_index]
    erkannter_pilz = pilzklassen[klasse_index]
    return erkannter_pilz, wahrscheinlichkeit

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python modell_test.py <pfad_zum_bild>")
        sys.exit(1)

    bild_pfad = sys.argv[1]

    if not os.path.exists(bild_pfad):
        print(f"Bilddatei nicht gefunden: {bild_pfad}")
        sys.exit(1)


    print("Lade Modell...")
    modell = tf.keras.models.load_model(modell_pfad)

    print(f"Klassifiziere Bild: {bild_pfad}")
    pilz, wahrscheinlichkeit = klassifiziere_pilz(modell, bild_pfad)

    print(f"Erkannter Pilz: {pilz}")
    print(f"Wahrscheinlichkeit: {wahrscheinlichkeit:.2f}")
