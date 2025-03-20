import subprocess

def query_ollama(prompt):
    try:
        # Der Befehl, um das Modell mit dem angegebenen Prompt auszuführen
        result = subprocess.run(
            ["ollama", "run", "gemma2", prompt],  # Der Prompt wird direkt als Argument übergeben
            text=True,  # Damit die Ausgabe als String behandelt wird
            capture_output=True,  # Um die Ausgabe von stdout zu fangen
            check=True  # Falls der Befehl fehlschlägt, wird eine Ausnahme ausgelöst
        )
        
        # Die Ausgabe des Modells anzeigen
        print("Antwort vom Modell:")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Abfragen des Modells: {e}")
        print(f"Fehlerausgabe: {e.stderr}")

# Der Prompt, den wir dem Modell stellen möchten
prompt = "In welchem Land steht der schiefe Turm von Pisa?"

# Modell abfragen
query_ollama(prompt)
