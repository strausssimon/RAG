import tensorflow as tf
import os

def konvertiere_modell():
    """Konvertiert das .h5 Modell zu .keras Format"""
    
    # Pfade angepasst für RAG-Ordner
    h5_pfad = os.path.join("..", "..", "RAG", "mushroom_4class_cnn_external_test.h5")
    keras_pfad = os.path.join("..", "..", "RAG", "mushroom_4class_cnn_external_test.keras")
    
    # Absolute Pfade für bessere Klarheit
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h5_pfad_abs = os.path.join(script_dir, h5_pfad)
    keras_pfad_abs = os.path.join(script_dir, keras_pfad)
    
    print("🔄 Konvertiere H5-Modell zu Keras-Format...")
    print(f"📂 Quelle: {h5_pfad_abs}")
    print(f"📂 Ziel: {keras_pfad_abs}")
    
    if not os.path.exists(h5_pfad_abs):
        print(f"❌ H5-Modell nicht gefunden: {h5_pfad_abs}")
        return False
    
    try:
        # Lade das H5-Modell mit verschiedenen Methoden
        print(f"📂 Lade H5-Modell...")
        print(f"   Dateigröße: {os.path.getsize(h5_pfad_abs) / (1024*1024):.1f} MB")
        
        try:
            model = tf.keras.models.load_model(h5_pfad_abs, compile=False)
            print("✅ Standard-Lademethode erfolgreich")
        except Exception as e1:
            print(f"❌ Standard-Methode fehlgeschlagen: {e1}")
            print("🔄 Versuche mit Custom Objects...")
            try:
                model = tf.keras.models.load_model(h5_pfad_abs, compile=False, custom_objects={})
                print("✅ Custom Objects Methode erfolgreich")
            except Exception as e2:
                print(f"❌ Custom Objects fehlgeschlagen: {e2}")
                return False
        
        # Speichere als Keras-Format
        print(f"💾 Speichere als Keras-Format...")
        model.save(keras_pfad_abs)
        
        # Vergleiche Dateigrößen
        h5_size = os.path.getsize(h5_pfad_abs) / (1024*1024)
        keras_size = os.path.getsize(keras_pfad_abs) / (1024*1024)
        
        print(f"✅ Konvertierung erfolgreich!")
        print(f"   H5-Datei: {h5_size:.1f} MB")
        print(f"   Keras-Datei: {keras_size:.1f} MB")
        
        # Teste das konvertierte Modell
        print("🧪 Teste konvertiertes Modell...")
        test_model = tf.keras.models.load_model(keras_pfad_abs, compile=False)
        print(f"   Input Shape: {test_model.input_shape}")
        print(f"   Output Shape: {test_model.output_shape}")
        print(f"   Anzahl Layer: {len(test_model.layers)}")
        
        print("\n🎯 Beide Dateien sind jetzt verfügbar:")
        print(f"   H5-Format: RAG/mushroom_4class_cnn_external_test.h5")
        print(f"   Keras-Format: RAG/mushroom_4class_cnn_external_test.keras")
        
        return True
        
    except Exception as e:
        print(f"❌ Konvertierung fehlgeschlagen: {e}")
        return False

if __name__ == "__main__":
    konvertiere_modell()
