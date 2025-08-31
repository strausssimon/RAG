"""
====================================================
Programmname : konvertierung h5-keras
Beschreibung : Konvertiert ein Keras-Modell im .h5 Format in das .keras Format.

====================================================
"""
import tensorflow as tf
import os

def konvertiere_modell():
    """Konvertiert das .h5 Modell zu .keras Format"""
    
    # Korrekte Pfade zum RAG-Ordner
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    rag_ordner = os.path.join(project_root, "RAG")
    
    h5_pfad = os.path.join(rag_ordner, "mushroom_4class_cnn_external_test.h5")
    keras_pfad = os.path.join(rag_ordner, "mushroom_4class_cnn_external_test.keras")
    
    print("🔄 Konvertiere H5-Modell zu Keras-Format...")
    print(f"📁 RAG-Ordner: {rag_ordner}")
    print(f"📂 Quell-Pfad: {h5_pfad}")
    print(f"💾 Ziel-Pfad: {keras_pfad}")
    
    if not os.path.exists(h5_pfad):
        print(f"❌ H5-Modell nicht gefunden: {h5_pfad}")
        print("📋 Verfügbare Dateien im RAG-Ordner:")
        if os.path.exists(rag_ordner):
            for file in os.listdir(rag_ordner):
                print(f"   - {file}")
        return False
    
    try:
        # Lade das H5-Modell
        print(f"📂 Lade H5-Modell...")
        model = tf.keras.models.load_model(h5_pfad, compile=False)
        
        # Speichere als Keras-Format
        print(f"💾 Speichere als Keras-Format...")
        model.save(keras_pfad)
        
        # Vergleiche Dateigrößen
        h5_size = os.path.getsize(h5_pfad) / (1024*1024)
        keras_size = os.path.getsize(keras_pfad) / (1024*1024)
        
        print(f"✅ Konvertierung erfolgreich!")
        print(f"   📊 H5-Datei: {h5_size:.1f} MB")
        print(f"   📊 Keras-Datei: {keras_size:.1f} MB")
        
        # Teste das konvertierte Modell
        print("🧪 Teste konvertiertes Modell...")
        test_model = tf.keras.models.load_model(keras_pfad, compile=False)
        print(f"   🎯 Input Shape: {test_model.input_shape}")
        print(f"   🎯 Output Shape: {test_model.output_shape}")
        print(f"   🔢 Anzahl Layer: {len(test_model.layers)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Konvertierung fehlgeschlagen: {e}")
        print(f"💡 Tipp: Das H5-Modell könnte ein Kompatibilitätsproblem haben")
        return False

if __name__ == "__main__":
    print("🔧 MODELL-KONVERTER: H5 → KERAS")
    print("=" * 50)
    erfolg = konvertiere_modell()
    
    if erfolg:
        print("\n🎉 Konvertierung abgeschlossen!")
        print("Das .keras Modell sollte jetzt besser kompatibel sein.")
    else:
        print("\n😞 Konvertierung fehlgeschlagen.")
        print("Das Original .h5 Modell hat wahrscheinlich Kompatibilitätsprobleme.")
