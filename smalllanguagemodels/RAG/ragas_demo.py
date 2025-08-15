"""
Demo und Anleitung für RAGAS Evaluation
"""

def show_demo():
    """Zeigt Demo-Verwendung der RAGAS Evaluation"""
    
    print("🎯 RAGAS EVALUATION DEMO")
    print("=" * 50)
    
    print("\n📋 Was wird getestet:")
    print("1. Faithfulness - Wie treu ist die Antwort zum Kontext?")
    print("2. Answer Relevancy - Wie relevant ist die Antwort zur Frage?")
    print("3. Context Precision - Wie präzise ist der Kontext?")
    print("4. Context Recall - Wie vollständig ist der Kontext?")
    print("5. Answer Correctness - Wie korrekt ist die Antwort?")
    print("6. Answer Similarity - Wie ähnlich ist die Antwort zur Ground Truth?")
    
    print("\n🔧 Setup-Schritte:")
    print("1. python setup_ragas.py  # Installiert RAGAS")
    print("2. python ragas_evaluation.py  # Führt Evaluation durch")
    
    print("\n📊 Ausgabe:")
    print("- ragas_evaluation_results.csv (Detaillierte RAGAS Metriken)")
    print("- simple_evaluation_results.csv (Vereinfachte Metriken)")
    
    print("\n🎯 Beispiel-Testfälle:")
    example_questions = [
        "Ist dieser Pilz essbar?",
        "Wie sieht dieser Pilz aus?", 
        "Wo findet man diesen Pilz?",
        "Wann ist die beste Zeit zum Sammeln?",
        "Wie bereitet man diesen Pilz zu?"
    ]
    
    for i, question in enumerate(example_questions, 1):
        print(f"{i}. {question}")
    
    print("\n📈 Erwartete Metriken:")
    print("- Faithfulness: 0.8+ (gut)")
    print("- Answer Relevancy: 0.7+ (gut)")
    print("- Context Precision: 0.9+ (sehr gut)")
    print("- Answer Correctness: 0.6+ (akzeptabel)")
    
    print("\n🚀 Jetzt starten mit:")
    print("python setup_ragas.py")

if __name__ == "__main__":
    show_demo()
