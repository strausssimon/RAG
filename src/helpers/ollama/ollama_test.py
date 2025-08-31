import subprocess
import shutil
import os
import re

def clean_output(text):
    """Entfernt ANSI-Escape-Codes und andere Formatierungen aus der Ausgabe"""
    if not text:
        return ""
    
    # ANSI-Escape-Codes entfernen
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_escape.sub('', text)
    
    # Zusätzliche Bereinigungen
    cleaned = cleaned.replace('\r', '')  # Carriage Returns entfernen
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)  # Steuerzeichen entfernen
    
    return cleaned.strip()

def check_ollama_installed():
    """Prüft ob Ollama installiert ist"""
    # Verschiedene mögliche Pfade für Ollama prüfen
    possible_paths = [
        "ollama",  # Standard PATH
        r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(os.environ.get('USERNAME', '')),
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe"
    ]
    
    for path in possible_paths:
        if shutil.which(path) or (path.endswith('.exe') and os.path.exists(path)):
            print(f"✅ Ollama gefunden: {path}")
            return path
    
    print("❌ Ollama ist nicht installiert oder nicht im PATH!")
    print("Installieren Sie Ollama von: https://ollama.ai/download")
    print("Starten Sie Ollama über das Startmenü und starten Sie PowerShell neu")
    return None

def query_ollama(prompt, model="phi3:mini"):
    # Erst prüfen ob Ollama verfügbar ist
    ollama_path = check_ollama_installed()
    if not ollama_path:
        return
    
    try:
        # Der Befehl, um das Modell mit dem angegebenen Prompt auszuführen
        print(f"🤖 Verwende Modell: {model}")
        result = subprocess.run(
            [ollama_path, "run", model, prompt],  # Vollständiger Pfad zu Ollama
            text=True,  # Damit die Ausgabe als String behandelt wird
            capture_output=True,  # Um die Ausgabe von stdout zu fangen
            check=True,  # Falls der Befehl fehlschlägt, wird eine Ausnahme ausgelöst
            encoding='utf-8',  # UTF-8 Encoding für Unicode-Zeichen
            errors='replace'   # Ersetze unbekannte Zeichen statt Fehler
        )
        
        # Die Ausgabe des Modells anzeigen
        print("✅ Antwort vom Modell:")
        print("-" * 50)
        cleaned_output = clean_output(result.stdout)
        print(cleaned_output)
        print("-" * 50)
        
    except subprocess.CalledProcessError as e:
        if "pull" in str(e.stderr):
            print(f"📥 Modell '{model}' wird heruntergeladen...")
            print(f"Führen Sie aus: ollama pull {model}")
        else:
            print(f"❌ Fehler beim Abfragen des Modells: {e}")
            print(f"Fehlerausgabe: {e.stderr}")

# Der Prompt, den wir dem Modell stellen möchten
if __name__ == "__main__":
    # Erste Test: Service-Status prüfen
    print("=== Ollama Service Test ===")
    ollama_path = check_ollama_installed()
    if ollama_path:
        try:
            # Test ob Service läuft mit 'list' command
            result = subprocess.run(
                [ollama_path, "list"],
                text=True,
                capture_output=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                print(f"✅ Ollama Service läuft!")
                cleaned_models = clean_output(result.stdout)
                print(f"Verfügbare Modelle:\n{cleaned_models}")
                
                # Beispiel-Prompt für RAG
                prompt = """
                Du bist ein Experte für Pilze. Beantworte folgende Frage kurz und präzise:
                Welche Merkmale unterscheiden giftige von essbaren Pilzen?
                """
                
                # Test mit Phi3:mini (optimiert für RAG)
                print("\n=== Pilz-Expertise Test ===")
                query_ollama(prompt, "phi3:mini")
            else:
                print(f"❌ Ollama Service läuft nicht. Fehler: {result.stderr}")
                print("💡 Versuche Modell zu installieren...")
                
                # Versuche Phi3:mini zu installieren
                try:
                    install_result = subprocess.run(
                        [ollama_path, "pull", "phi3:mini"],
                        text=True,
                        capture_output=True,
                        timeout=300,  # 5 Minuten für Download
                        encoding='utf-8',
                        errors='replace'
                    )
                    
                    if install_result.returncode == 0:
                        print("✅ Phi3:mini erfolgreich installiert!")
                        print("🔄 Teste erneut...")
                        
                        # Nochmal testen
                        prompt = "In welchem Land steht der schiefe Turm von Pisa?"
                        query_ollama(prompt, "phi3:mini")
                    else:
                        print(f"❌ Installation fehlgeschlagen: {install_result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print("⏰ Download-Timeout (Modell zu groß oder langsame Verbindung)")
                except Exception as e:
                    print(f"❌ Installations-Fehler: {e}")
                
        except subprocess.TimeoutExpired:
            print("⏰ Ollama antwortet nicht (Timeout)")
        except Exception as e:
            print(f"❌ Fehler beim Testen: {e}")
    else:
        print("❌ Ollama nicht gefunden")
