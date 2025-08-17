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
├── data/                     # Raw und Prepared Datasets
│   ├── all_mushrooms
│   │   ├── Agaricus_arvensis
│   │   │   └── ...
│   │   ├── ... 
│   │   │   └── ...
│   │   └── Volvariella_volacea
│   │       └── ...
│   ├── augmented_mushrooms
│   │   ├── example
│   │   │   └── ...
│   │   └── resized
│   │       ├── Amanita_phalloides
│   │       │   └── ...
│   │       ├── Armillaria_mellea
│   │       │   └── ...
│   │       ├── Boletus_edulis
│   │       │   └── ...
│   │       └── Cantharellus_cibarius
│   │           └── ...
│   ├── cropped_mushrooms
│   │   ├── Cropped_Armillaria_mellea
│   │   │   └── ...
│   │   └── Cropped_Boletus_edulis
│   │       └── ...
│   ├── images_mushrooms
│   │   ├── Amanita_muscaria
│   │   │   └── ...
│   │   ├── ...
│   │   │   └── ...
│   │   └── Tylopilus_felleus
│   │       └── ...
│   ├── resized_mushrooms 
│   │   ├── Amanita_phalloides
│   │   │   └── ...
│   │   ├── Armillaria_mellea
│   │   │   └── ...
│   │   ├── Boletus_edulis
│   │   │   └── ...
│   │   └── Cantharellus_cibarius
│   │       └── ...
│   └── test_mushrooms        # Testdatensatz
│       └── ...
│
├── models/                   # Vorgefertigte Modelle (CNN, etc.)
│   ├── mushroom_4class_cnn_external_test.h5
│   ├── mushroom_4class_cnn_external_test.keras
│   └── ...
│
├── results/                  # Evaluationsergebnisse, CSVs, Visualisierungen
│   ├── ollama_rag_evaluation_simple.csv
│   ├── ollama_rag_evaluation_detailed.csv
│   └── ...
│
├── src/                      # Hauptmodule und Kernlogik
│   ├── CNN/
│   │   └──  cnn.py
│   ├── helpers/              # Hilfsfunktionen und Utilities für Daten und Modelle
│   │   ├── augment_images.py
│   │   ├── augment.py
│   │   ├── augmentation.py
│   │   ├── convert_model_fixed.py
│   │   ├── convert_model.py
│   │   ├── create_augmentation_examples.py
│   │   ├── create_test_set.py
│   │   ├── crop_mushrooms.py
│   │   ├── dataset.py
│   │   ├── find_min_image_size.py
│   │   ├── rename_2_fixed.py
│   │   ├── rename_2.py
│   │   ├── rename_test_clean.py
│   │   ├── rename.py
│   │   ├── resize.py
│   │   ├── restore_test_set.py
│   │   └── robust_test_set.py
│   ├── RAG/
│   │   ├── ragas/
│   │   │   ├── ragas_demo.py
│   │   │   ├── ragas_evaluation.py
│   │   │   └── ragas_setup.py
│   │   ├── Informationen_RAG.json
│   │   ├── RAG.py
│   │   └── test.py
│   └── webscraper/
│       ├── mushroom_scraper_2.py
│       ├── mushroom_scraper.py
│       └── scraper_test.py
│
├── requirements.txt          # Python-Abhängigkeiten
├── .env                      # Umgebungsvariablen
└── README.md                 # Dieses Dokument
```

---

## Installation

1. **Repository klonen:**
   ```sh
   git clone https://github.com/strausssimon/SmallLanguageModels.git
   cd SmallLanguageModels
   ```

2. **Python-Umgebung einrichten (empfohlen):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Abhängigkeiten installieren:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Umgebungsvariablen anpassen:**  
   Trage sensible Informationen (API-Keys, Pfade etc.) in die `.env`-Datei ein (nicht mit Git tracken!).

---

## Nutzung

- **Pilzklassifikation ausführen:**  
  Beispielskript zum Klassifizieren von Pilzbildern befindet sich unter `src/rag/rag.py`.

- **Webscraping starten:**  
  Scraper-Skripte im `src/webscraper/`-Verzeichnis.

- **Evaluation:**  
  Nutze die Skripte in `src/rag/evaluation.py` oder die bereitgestellten Notebooks für das Benchmarking.

- **Interaktive Wissensabfrage:**  
  Über das SLM können, basierend auf Klassifikationsergebnissen, Erklärungen und Zusatzinformationen zu Pilzarten abgerufen werden.

---

## Beispielaufruf

```sh
python src/rag/rag.py --image data/beispiel_pilz.jpg
```

---

## Weiterführende Hinweise

- **Tests:**  
  Alle Testskripte befinden sich im Ordner `tests/` und können via `pytest` ausgeführt werden.

- **Eigene Modelle:**  
  Zum Austausch des CNN-Modells einfach die entsprechende `.h5`-Datei im `models/`-Verzeichnis ersetzen.

- **Datenquellen:**  
  Verwendete Bilder und externe Datenquellen sind im Ordner `data/` dokumentiert.

---

## Kontakt

Fragen, Anregungen oder Beiträge bitte via [GitHub Issues](https://github.com/strausssimon/SmallLanguageModels/issues) einreichen.
