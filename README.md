# SmallLanguageModels

## Projektübersicht

Dieses Repository enthält ein modulares System zur automatisierten Klassifikation von Pilzarten und zur Wissensbereitstellung mittels Retrieval-Augmented Generation (RAG). Die Lösung kombiniert eine effiziente Webscraping-Komponente mit einer modernen Bildverarbeitung (CNN) sowie einen Small Language Model (SLM) zur nutzerfreundlichen Interaktion und Wissensausgabe. Das Projekt richtet sich an Forschende und Entwickler aus den Bereichen Computer Vision, maschinelles Lernen und Data Mining mit Fokus auf Pilzbestimmung und angewandte KI.

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
│   ├── inaturalist_mushrooms   # Scraped Datensatz
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
│   ├── randomized_mushrooms    # 1500 Bilder je Pilz
│   │   └── inaturalist
│   │   │   ├── Amanita_muscaria
│   │   │   │   └── ...
│   │   │   ├── Armillaria_mellea
│   │   │   │   └── ...
│   │   │   ├── Amanita_muscaria
│   │   │   │   └── ...
│   │   │   ├── Boletus_edulis
│   │   │   │   └── ...
│   │   │   ├── Cantharellus_cibarius
│   │   │   │   └── ...
│   │   │   └── Phallus_impudicus
│   ├── recolored_mushrooms   # grün -> grau 
│   │   └── inaturalist
│   │   │   ├── Amanita_muscaria
│   │   │   │   └── ...
│   │   │   ├── Armillaria_mellea
│   │   │   │   └── ...
│   │   │   ├── Amanita_muscaria
│   │   │   │   └── ...
│   │   │   ├── Boletus_edulis
│   │   │   │   └── ...
│   │   │   ├── Cantharellus_cibarius
│   │   │   │   └── ...
│   │   │   └── Phallus_impudicus
│   ├── resized_mushrooms     # 200 x 200
│   │   └── inaturalist
│   │   │   ├── Amanita_muscaria
│   │   │   │   └── ...
│   │   │   ├── Armillaria_mellea
│   │   │   │   └── ...
│   │   │   ├── Amanita_muscaria
│   │   │   │   └── ...
│   │   │   ├── Boletus_edulis
│   │   │   │   └── ...
│   │   │   ├── Cantharellus_cibarius
│   │   │   │   └── ...
│   │   │   └── Phallus_impudicus
│   └── test_mushrooms        # Testdatensatz
│       └── ...
│
├── models/                   # Vorgefertigte Modelle (CNN, etc.)
│   └── mushroom_resnet50_transfer_80_20.keras  # Über GIT LFS (siehe oben) 
│
├── results/                  # Evaluationsergebnisse, CSVs, Visualisierungen
│   ├── ollama_rag_evaluation_simple.csv
│   ├── ollama_rag_evaluation_detailed.csv
│   └── ...
│
├── src/                      # Hauptmodule und Kernlogik
│   ├── CNN/
│   │   ├── cnn_resnet.py
│   │   └── cnn_test.py
│   ├── GUI/
│   │   └──  gui.py
│   ├── helpers/              # Hilfsfunktionen und Utilities für Daten und Modelle

│   │   ├── convert_model_fixed.py
│   │   ├── count_files_in_path.py
│   │   ├── crop_mushrooms.py
│   │   ├── find_min_image_size.py
│   │   ├── rename_test_clean.py
│   │   ├── rename.py
│   │   ├── resize.py
│   │   └── robust_test_set.py
│   ├── RAG/
│   │   ├── ragas/
│   │   │   ├── ragas_demo.py
│   │   │   ├── ragas_evaluation.py
│   │   │   └── ragas_setup.py
│   │   ├── Informationen_RAG.json
│   │   └── RAG_mit_CNN.py
│   └── Webscraper/
│       └── inaturalist_scraper.py
│
├── README.md                 # Dieses Dokument
├── .env                      # Umgebungsvariablen
└── requirements.txt          # Python-Abhängigkeiten
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

5. **Umgebungsvariablen anpassen:**  
Trage sensible Informationen (API-Keys, Pfade etc.) in die `.env`-Datei ein (nicht mit Git tracken!).

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

## Beispielaufruf

```sh
python src/rag/rag.py --image data/beispiel_pilz.jpg
```

---

## Kontakt

Fragen, Anregungen oder Beiträge bitte via [GitHub Issues](https://github.com/strausssimon/SmallLanguageModels/issues) einreichen.
