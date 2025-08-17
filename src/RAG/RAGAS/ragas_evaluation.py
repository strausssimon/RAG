"""
RAGAS Evaluation Script für RAG.py mit Ollama
Testet die Qualität des RAG-Systems mit lokalen Modellen
"""


import json
import pandas as pd
import numpy as np
import os
import sys
import subprocess
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# RAGAS Imports and Availability Check
try:
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness, answer_similarity
    from ragas import evaluate
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"RAGAS oder Abhängigkeiten konnten nicht importiert werden: {e}")
    RAGAS_AVAILABLE = False


# Vereinfachte RAGAS-ähnliche Evaluation ohne externe APIs
import sys
import os
# Add parent directory to sys.path to allow import of RAG.py
rag_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if rag_dir not in sys.path:
    sys.path.insert(0, rag_dir)
try:
    from RAG import PILZ_NAME, pilzdaten, frage_beantworten
    RAG_AVAILABLE = True
    print("RAG.py erfolgreich importiert")
except ImportError as e:
    print(f"Fehler beim Importieren von RAG.py: {e}")
    RAG_AVAILABLE = False

class RAGASEvaluator:
    """RAGAS-basierte Evaluation für RAG.py"""
    
    def __init__(self, json_path="Informationen_RAG.json"):
        self.json_path = json_path
        # Pilzname explizit auf "Gemeiner Steinpilz" setzen
        self.pilz_name = "Gemeiner Steinpilz"
        self.pilzdaten = pilzdaten if RAG_AVAILABLE else []
        self.test_cases = []
        self.results = {}
        
    def load_test_cases(self):
        """Erstellt Testfälle basierend auf den Pilzdaten"""
        print(f"Erstelle Testfälle für Pilz: {self.pilz_name}")

        # Finde den spezifischen Pilz in den Daten
        pilz_info = None
        for pilz in self.pilzdaten:
            if pilz.get("name") == self.pilz_name:
                pilz_info = pilz
                break

        if not pilz_info:
            print(f"Pilz '{self.pilz_name}' nicht in den Daten gefunden!")
            return

        # Verschiedene Fragetypen generieren
        self.test_cases = [
            {
                "question": "Ist dieser Pilz essbar?",
                "expected_answer": f"Der {self.pilz_name} ist {pilz_info.get('essbar', 'unbekannt')} essbar.",
                "context": json.dumps(pilz_info, ensure_ascii=False),
                "category": "Essbarkeit"
            },
            {
                "question": "Wie sieht dieser Pilz aus?",
                "expected_answer": pilz_info.get('beschreibung', 'Keine Beschreibung verfügbar'),
                "context": json.dumps(pilz_info, ensure_ascii=False),
                "category": "Aussehen"
            },
            {
                "question": "Wo findet man diesen Pilz?",
                "expected_answer": pilz_info.get('vorkommen', 'Vorkommen unbekannt'),
                "context": json.dumps(pilz_info, ensure_ascii=False),
                "category": "Vorkommen"
            },
            {
                "question": "Wann ist die beste Zeit zum Sammeln?",
                "expected_answer": pilz_info.get('saison', 'Saison unbekannt'),
                "context": json.dumps(pilz_info, ensure_ascii=False),
                "category": "Saison"
            },
            {
                "question": "Wie bereitet man diesen Pilz zu?",
                "expected_answer": pilz_info.get('zubereitung', 'Zubereitung unbekannt'),
                "context": json.dumps(pilz_info, ensure_ascii=False),
                "category": "Zubereitung"
            },
            {
                "question": "Mit welchen Pilzen kann man ihn verwechseln?",
                "expected_answer": str(pilz_info.get('verwechslungsgefahr', [])),
                "context": json.dumps(pilz_info, ensure_ascii=False),
                "category": "Verwechslung"
            },
            {
                "question": "Wie riecht dieser Pilz?",
                "expected_answer": pilz_info.get('geruch', 'Geruch unbekannt'),
                "context": json.dumps(pilz_info, ensure_ascii=False),
                "category": "Sinneseindruck"
            },
            {
                "question": "Wie schmeckt dieser Pilz?",
                "expected_answer": pilz_info.get('geschmack', 'Geschmack unbekannt'),
                "context": json.dumps(pilz_info, ensure_ascii=False),
                "category": "Sinneseindruck"
            },
            {
                "question": "Gibt es Gefahrstoffe in diesem Pilz?",
                "expected_answer": pilz_info.get('gefahrstoffe', 'Keine Angaben zu Gefahrstoffen'),
                "context": json.dumps(pilz_info, ensure_ascii=False),
                "category": "Sicherheit"
            },
            {
                "question": "Wie lautet der lateinische Name?",
                "expected_answer": pilz_info.get('lateinisch', 'Lateinischer Name unbekannt'),
                "context": json.dumps(pilz_info, ensure_ascii=False),
                "category": "Taxonomie"
            }
        ]
        
        print(f"✅ {len(self.test_cases)} Testfälle erstellt")
        
        # Zeige Beispiel-Testfall
        print("\nBeispiel-Testfall:")
        example = self.test_cases[0]
        print(f"   Frage: {example['question']}")
        print(f"   Erwartete Antwort: {example['expected_answer'][:100]}...")
        print(f"   Kategorie: {example['category']}")
    
    def get_rag_answers(self):
        """Holt Antworten vom RAG System"""
        print("\nHole Antworten vom RAG-System...")
        
        if not RAG_AVAILABLE:
            print("RAG.py nicht verfügbar!")
            return
        
        for i, test_case in enumerate(self.test_cases):
            try:
                print(f"   Frage {i+1}/{len(self.test_cases)}: {test_case['question']}")
                
                # Hole Antwort von RAG
                answer = frage_beantworten(test_case['question'])
                test_case['rag_answer'] = answer
                
                print(f"   Antwort erhalten: {answer[:80]}...")
                
            except Exception as e:
                print(f"   Fehler bei Frage {i+1}: {e}")
                test_case['rag_answer'] = f"Fehler: {str(e)}"
        print("Alle RAG-Antworten gesammelt")
    
    def prepare_ragas_dataset(self):
        """Bereitet Dataset für RAGAS vor"""
        if not RAGAS_AVAILABLE:
            print("RAGAS ist nicht verfügbar")
            return None
        
        print("\nVorbereitung RAGAS Dataset.")
        
        # Erstelle Dataset im RAGAS-Format
        data = {
            "question": [tc["question"] for tc in self.test_cases],
            "answer": [tc["rag_answer"] for tc in self.test_cases],
            "contexts": [[tc["context"]] for tc in self.test_cases],  # RAGAS erwartet Liste von Kontexten
            "ground_truth": [tc["expected_answer"] for tc in self.test_cases]
        }
        
        dataset = Dataset.from_dict(data)
        print(f"Dataset mit {len(dataset)} Einträgen erstellt")
        
        return dataset
    
    def run_ragas_evaluation(self, dataset):
        """Führt RAGAS Evaluation durch"""
        if not RAGAS_AVAILABLE:
            print("RAGAS ist nicht verfügbar.")
            return {}
        
        print("\nDurchführung RAGAS Evaluation.")

        # Definiere Metriken
        metrics = [
            faithfulness,          # Wie treu ist die Antwort zum Kontext?
            answer_relevancy,      # Wie relevant ist die Antwort zur Frage?
            context_precision,     # Wie präzise ist der Kontext?
            context_recall,        # Wie vollständig ist der Kontext?
            answer_correctness,    # Wie korrekt ist die Antwort?
            answer_similarity      # Wie ähnlich ist die Antwort zur Ground Truth?
        ]
        
        try:
            # Führe Evaluation durch
            result = evaluate(
                dataset,
                metrics=metrics,
            )
            
            print("RAGAS Evaluation abgeschlossen.")
            return result
            
        except Exception as e:
            print(f"RAGAS Evaluation fehlgeschlagen: {e}")
            print("Stelle sicher, dass OpenAI API Key gesetzt ist oder verwende lokale Modelle")
            return {}
    
    def analyze_results(self, ragas_result):
        """Analysiert die RAGAS Ergebnisse"""
        print("\nRAGAS EVALUATION ERGEBNISSE")
        print("=" * 50)
        
        if not ragas_result:
            print("Keine Ergebnisse verfügbar.")
            return
        
        # Zeige Metriken
        for metric, score in ragas_result.items():
            if isinstance(score, (int, float)):
                print(f"Metrik {metric}: {score:.4f}")
        
        # Erstelle DataFrame für detaillierte Analyse
        df_results = pd.DataFrame({
            'Frage': [tc['question'] for tc in self.test_cases],
            'Kategorie': [tc['category'] for tc in self.test_cases],
            'RAG_Antwort': [tc['rag_answer'] for tc in self.test_cases],
            'Erwartete_Antwort': [tc['expected_answer'] for tc in self.test_cases]
        })
        
        # Speichere Ergebnisse
        output_file = "ragas_evaluation_results.csv"
        df_results.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nDetaillierte Ergebnisse gespeichert: {output_file}")
        
        # Kategorien-Analyse
        print(f"\nANALYSE NACH KATEGORIEN:")
        category_counts = df_results['Kategorie'].value_counts()
        for category, count in category_counts.items():
            print(f"   {category}: {count} Fragen")
        
        return df_results
    
    def run_simple_evaluation(self):
        """Führt vereinfachte Evaluation ohne RAGAS durch"""
        print("\nDurchführung vereinfachte Evaluation.")
        
        results = []
        
        for test_case in self.test_cases:
            question = test_case['question']
            rag_answer = test_case['rag_answer']
            expected = test_case['expected_answer']
            category = test_case['category']
            
            # Einfache Metriken
            answer_length = len(rag_answer) if rag_answer else 0
            has_answer = bool(rag_answer.strip()) and "keine Information" not in rag_answer.lower()
            
            # Keyword-basierte Relevanz
            expected_keywords = set(expected.lower().split())
            answer_keywords = set(rag_answer.lower().split())
            keyword_overlap = len(expected_keywords.intersection(answer_keywords))
            
            results.append({
                'question': question,
                'rag_answer': rag_answer,
                'expected_answer': expected,
                'has_answer': has_answer,
                'answer_length': answer_length,
                'relevant_keywords': keyword_overlap,
                'appropriate_length': answer_length > 10
            })
        
        df_simple = pd.DataFrame(results)
        
        # Zusammenfassung
        print("\nVEREINFACHTE EVALUATION ERGEBNISSE")
        print("=" * 50)
        if 'has_answer' in df_simple:
            print(f"Antwort-Rate: {df_simple['has_answer'].mean():.2%}")
        if 'answer_length' in df_simple:
            print(f"Durchschnittliche Antwortlänge: {df_simple['answer_length'].mean():.0f} Zeichen")
        if 'relevant_keywords' in df_simple:
            print(f"Durchschnittliche relevante Keywords: {df_simple['relevant_keywords'].mean():.2f}")
        if 'appropriate_length' in df_simple:
            print(f"Anteil angemessener Länge: {df_simple['appropriate_length'].mean():.2%}")
        
        # Speichere Ergebnisse
        simple_output = "simple_evaluation_results.csv"
        df_simple.to_csv(simple_output, index=False, encoding='utf-8')
        print(f"\nErgebnisse gespeichert: {simple_output}")
        
        return df_simple
    
    def run_evaluation(self):
        """Hauptfunktion für die Evaluation"""
        print("RAGAS EVALUATION FÜR RAG.py")
        print("=" * 60)
        print(f"Pilz: {self.pilz_name}")
        print(f"JSON-Datei: {self.json_path}")
        print("=" * 60)
        
        # Schritt 1: Testfälle laden
        self.load_test_cases()
        
        # Schritt 2: RAG-Antworten holen
        self.get_rag_answers()
        
        # Schritt 3: RAGAS Evaluation (falls verfügbar)
        if RAGAS_AVAILABLE:
            dataset = self.prepare_ragas_dataset()
            if dataset:
                ragas_result = self.run_ragas_evaluation(dataset)
                df_detailed = self.analyze_results(ragas_result)
        
        # Schritt 4: Vereinfachte Evaluation
        df_simple = self.run_simple_evaluation()
        
        print("\nEvaluation abgeschlossen.")
        print("Überprüfe die CSV-Dateien für detaillierte Ergebnisse")
        
        return df_simple

def main():
    """Hauptfunktion"""
    print("Start RAGAS Evaluation.")
    
    # Prüfe Abhängigkeiten
    if not RAG_AVAILABLE:
        print("RAG.py konnte nicht importiert werden.")
        print("Stelle sicher, dass RAG.py im gleichen Ordner liegt.")
        return
    
    # Erstelle Evaluator
    evaluator = RAGASEvaluator()
    
    # Führe Evaluation durch
    results = evaluator.run_evaluation()
    
    print("\nRAGAS Evaluation fertig.")

if __name__ == "__main__":
    main()
