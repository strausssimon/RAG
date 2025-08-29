# RAG-System zur automatisierten Klassifikation von Pilzen und Wissensbereitstellung durch ein LLM

## Projektübersicht

Dieses Repository enthält ein modulares System zur automatisierten Klassifikation von Pilzarten und zur Wissensbereitstellung mittels Retrieval-Augmented Generation (RAG). Die Lösung kombiniert eine effiziente Webscraping-Komponente mit einer modernen Bildverarbeitung (CNN) sowie einen Large Language Model (LLM) zur nutzerfreundlichen Interaktion und Wissensausgabe. Das Projekt richtet sich an Forschende und Entwickler aus den Bereichen Computer Vision, maschinelles Lernen und Data Mining mit Fokus auf Pilzbestimmung und angewandte KI.

---

## Hauptfunktionen

- **Webscraping:**  
  Extraktion und Aktualisierung von Pilzdaten aus externen Quellen mittels anpassbarem Scraper.

- **Automatische Pilzklassifikation:**  
  Implementierung eines Convolutional Neural Networks (CNN) für die Bildklassifikation von Pilzarten.

- **RAG-System:**  
  Integration eines Retrieval-Augmented Generation Workflows zur Kontextanreicherung und erklärenden Wissensausgabe durch ein Sprachmodell.

- **Evaluations- und Benchmark-Tools:**  
  Skripte zur Modellbewertung (z.B. mit RAGAS) sowie Vergleich und Visualisierung von Ergebnissen.

---

## Projektstruktur

```plaintext
smalllanguagemodels/
├── data/                       # Raw und Prepared Datasets
│   ├── csv
│   │   └── inaturalist         # CSV für Scraper
│   │       ├── inaturalist_amanita_muscaria.csv
│   │       │── ...
│   │       └── inaturalist_phallus_impudicus.csv
│   │ 
│   ├── inaturalist_mushrooms   # leer aufgrund Speicherbegrenzung FOM 256 MB
│   │   ├── Amanita_muscaria
│   │   │   └── ...
│   │   ├── Armillaria_mellea
│   │   │   └── ...
│   │   ├── Amanita_muscaria
│   │   │   └── ...
│   │   ├── Boletus_edulis
│   │   │   └── ...
│   │   ├── Cantharellus_cibarius
│   │   │   └── ...
│   │   └── Phallus_impudicus
│   │ 
│   ├── inaturalist_samples   # leer aufgrund Speicherbegrenzung FOM 256 MB
│   │   ├── amanita_muscaria_7488.jpg
│   │   ├── ...
│   │   └── phallus_impudicus_545748746.jpg
│   │ 
│   ├── randomized_mushrooms    # 1500 Bilder je Pilz # leer aufgrund Speicherbegrenzung FOM 256 MB
│   │   └── inaturalist
│   │       ├── Amanita_muscaria
│   │       │   └── ...
│   │       ├── Armillaria_mellea
│   │       │   └── ...
│   │       ├── Amanita_muscaria
│   │       │   └── ...
│   │       ├── Boletus_edulis
│   │       │   └── ...
│   │       ├── Cantharellus_cibarius
│   │       │   └── ...
│   │       └── Phallus_impudicus
│   │ 
│   ├── resized_mushrooms     # 200 x 200 # leer aufgrund Speicherbegrenzung FOM 256 MB
│   │   └── inaturalist
│   │       ├── Amanita_muscaria
│   │       │   └── ...
│   │       ├── Armillaria_mellea
│   │       │   └── ...
│   │       ├── Amanita_muscaria
│   │       │   └── ...
│   │       ├── Boletus_edulis
│   │       │   └── ...
│   │       ├── Cantharellus_cibarius
│   │       │   └── ...
│   │       └── Phallus_impudicus
│   │ 
│   └── test_mushrooms        # Testdatensatz # leer aufgrund Speicherbegrenzung FOM 256 MB
│       ├── Amanita_muscaria
│       │   └── ...
│       ├── Armillaria_mellea
│       │   └── ...
│       ├── Amanita_muscaria
│       │   └── ...
│       ├── Boletus_edulis
│       │   └── ...
│       ├── Cantharellus_cibarius
│       │   └── ...
│       └── Phallus_impudicus
│
├── models/                   # Vorgefertigte Modelle (CNN, etc.)// Über GIT LFS (siehe oben)
│   └── mushroom_5class_resnet_cnn_80_20_split_2.keras 
│
├── src/                      # Hauptmodule und Kernlogik
│   ├── CNN/
│   │   ├── cnn_resnet.py
│   │   ├── cnn_test_model_keras.py
│   │   └── cnn_test_sample_lime.py
│   │
│   ├── GUI/
│   │   └──  GUI.py
│   │
│   ├── helpers/              # Hilfsfunktionen und Utilities für Daten und Modelle
│   │   ├── count_files_in_path.py
│   │   ├── find_min_image_size.py
│   │   ├── randomize_and_move_images.py 
│   │   ├── rename_test_clean.py
│   │   ├── rename.py
│   │   ├── resize.py
│   │   └── robust_test_set.py
│   │ 
│   ├── RAG/
│   │   ├── ragas/
│   │   │   ├── ragas_demo.py
│   │   │   ├── ragas_evaluation.py
│   │   │   └── ragas_setup.py
│   │   │ 
│   │   ├── results/                  # RAGAS results
│   │   │   ├── ollama_category_summary.csv
│   │   │   ├── ollama_per_question_analysis.csv
│   │   │   ├── ollama_summary.json
│   │   │   └── ollama_worst_cases.csv
│   │   │
│   │   ├── Informationen_RAG.json
│   │   └── rag.py
│   │
│   └── Webscraper/
│       └── inaturalist_scraper.py    
│
├── README.md                             # Dieses Dokument
├── .env                                  # Umgebungsvariablen
├── cnn_resnet50_output_log_27082025.txt  # Log des CNN-Model
├── best_mushroom_model.keras             # Backup Keras-Model
└── requirements.txt                      # Python-Abhängigkeiten
```

---

## Installation

1. **Repository klonen:**
git clone https://github.com/strausssimon/SmallLanguageModels.git
cd SmallLanguageModels

2. **Git LFS initialisieren und Modelle herunterladen**
git lfs install          # nur einmal nötig
git lfs pull             # lädt alle großen Dateien herunter


3. **Python-Umgebung einrichten (empfohlen):**
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


4. **Abhängigkeiten installieren:**
pip install -r requirements.txt

---

## Nutzung

- **Pilzklassifikation ausführen:**  
  Beispielskript zum Klassifizieren von Pilzbildern befindet sich unter `src/rag/rag.py`.

- **Webscraping starten:**  
  Scraper-Skripte im `src/webscraper/`-Verzeichnis.

- **Evaluation:**  
  Das Skript in `ragas_evaluation.py` ist für das Benchmarking.

- **Interaktive Wissensabfrage:**  
  Über das SLM können, basierend auf Klassifikationsergebnissen, Erklärungen und Zusatzinformationen zu Pilzarten abgerufen werden.

---
