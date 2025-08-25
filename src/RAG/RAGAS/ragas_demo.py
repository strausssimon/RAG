"""
====================================================
Programmname : RAGAS Evaluation
Datum        : 17.08.2025
Version      : 1.0
Beschreibung : Demonstration und Anleitung zur Nutzung der RAGAS Evaluation.
               Dieses Skript zeigt die wichtigsten Evaluationsmetriken und den Ablauf der Auswertung.
====================================================
"""

def show_demo():
    """
    Zeigt eine Übersicht der wichtigsten Evaluationsmetriken und den Ablauf der RAGAS-Auswertung.
    """
    print("RAGAS EVALUATION DEMO")
    print("=" * 50)

    print("\nGetestete Metriken:")
    print("1. Faithfulness - Übereinstimmung der Antwort mit dem Kontext")
    print("2. Answer Relevancy - Relevanz der Antwort zur Frage")
    print("3. Context Precision - Präzision des genutzten Kontexts")
    print("4. Context Recall - Vollständigkeit des Kontexts")
    print("5. Answer Correctness - Korrektheit der Antwort")
    print("6. Answer Similarity - Ähnlichkeit zur Ground Truth")

    print("\nSetup-Schritte:")
    print("1. python setup_ragas.py  # Installation der RAGAS-Abhängigkeiten")
    print("2. python ragas_evaluation.py  # Durchführung der Evaluation")

    print("\nAusgabe:")
    print("- ragas_evaluation_results.csv (detaillierte RAGAS-Metriken)")
    print("- simple_evaluation_results.csv (vereinfachte Metriken)")

    print("\nBeispiel-Testfälle:")
    example_questions = [
        "Ist dieser Pilz essbar?",
        "Wie sieht dieser Pilz aus?",
        "Wo findet man diesen Pilz?",
        "Wann ist die beste Zeit zum Sammeln?",
        "Wie bereitet man diesen Pilz zu?"
    ]
    for i, question in enumerate(example_questions, 1):
        print(f"{i}. {question}")

    print("\nErwartete Metriken:")
    print("- Faithfulness: 0.8+ (gut)")
    print("- Answer Relevancy: 0.7+ (gut)")
    print("- Context Precision: 0.9+ (sehr gut)")
    print("- Answer Correctness: 0.6+ (akzeptabel)")

    print("\nStart der Evaluation mit:")
    print("python setup_ragas.py")

if __name__ == "__main__":
    show_demo()
