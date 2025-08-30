"""
====================================================
Programmname : RAGAS Evaluation
Beschreibung : RAGAS Evaluation Script für rag.py mit Ollama
               Testet die Qualität des RAG-Systems mit lokalen Modellen
====================================================
"""

import os
import re
import json
import shutil
import subprocess
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')

# === Automatische Ollama-Konfiguration für RAGAS (falls RAGAS verwendet wird) ===
if os.environ.get("RAGAS_LLM_PROVIDER", "").lower() != "ollama":
    os.environ["RAGAS_LLM_PROVIDER"] = "ollama"
    print("[INFO] Setze RAGAS_LLM_PROVIDER=ollama für lokale LLM-Auswertung.")

if "RAGAS_LLM_MODEL" not in os.environ:
    os.environ["RAGAS_LLM_MODEL"] = "llama2"
    print("[INFO] Setze RAGAS_LLM_MODEL=llama2 (Standardmodell für Ollama).")

print("RAGAS_LLM_PROVIDER:", os.environ.get("RAGAS_LLM_PROVIDER"))
print("RAGAS_LLM_MODEL:", os.environ.get("RAGAS_LLM_MODEL"))

# === RAGAS Imports (optional) ===
try:
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision,
        context_recall, answer_correctness, answer_similarity
    )
    from ragas import evaluate
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] RAGAS oder Abhängigkeiten konnten nicht importiert werden: {e}")
    RAGAS_AVAILABLE = False

# === rag.py laden (optional – wird hier nicht aktiv genutzt, aber geprüft) ===
try:
    base_dir = Path(__file__).resolve().parent
except NameError:
    base_dir = Path.cwd()

rag_path = base_dir.parent / "rag.py"
RAG_AVAILABLE = False
if rag_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rag", rag_path)
    rag = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(rag)
        RAG_AVAILABLE = True
        print("[INFO] rag.py erfolgreich importiert.")
        # Optionaler Testkontext
        if hasattr(rag, "setze_test_pilz"):
            rag.setze_test_pilz("Gemeiner Steinpilz")
    except Exception as e:
        print(f"[ERROR] Fehler beim Importieren von rag.py: {e}")
else:
    print(f"[WARN] rag.py nicht gefunden unter: {rag_path}")


# ===========================
# Hilfsfunktionen (Analyse)
# ===========================
def _normalize_text(s: str) -> str:
    """Einfache Normalisierung: lower, Sonderzeichen/Mehrfachwhitespace entfernen."""
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False)
    s = s.lower()
    s = re.sub(r"[^\wäöüß]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_set(s: str) -> set:
    """Tokenisierung in einfache Wortmenge."""
    return set(_normalize_text(s).split())


def _safe_json_dump(v) -> str:
    """Beliebigen Wert sicher als String erzeugen."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _nested_get(d: dict, path: list, default=None):
    """Sicherer Zugriff auf verschachtelte Dicts."""
    cur = d
    try:
        for p in path:
            if isinstance(cur, dict):
                cur = cur.get(p, default)
            else:
                return default
        return cur
    except Exception:
        return default


# =========================================
# Hauptklasse: Ollama-basierte Evaluation
# =========================================
class RAGASEvaluator:
    """Ollama-basierte Evaluation + (optional) RAGAS-Gesamtmetriken."""

    def __init__(self, json_path=None, output_dir=None, pilz_name="Gemeiner Steinpilz"):
        # Pfade
        self.json_path = Path(json_path) if json_path else (base_dir.parent / "Informationen_RAG.json")
        self.output_dir = Path(output_dir) if output_dir else (base_dir.parent / "results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Konfiguration
        self.pilz_name = pilz_name
        self.model = "phi3:mini"  # Default; kann im Aufruf überschrieben werden

        # Daten laden
        self.pilzdaten = []
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                self.pilzdaten = json.load(f)
        except Exception as e:
            print(f"[ERROR] Fehler beim Laden der Pilzdaten: {e}")

        # pilz_info extrahieren
        self.pilz_info = None
        for pilz in self.pilzdaten or []:
            if pilz.get("bezeichnung", {}).get("name") == self.pilz_name:
                self.pilz_info = pilz
                break

        if self.pilz_info:
            print(f"[INFO] Pilzdaten für '{self.pilz_name}' geladen.")
        else:
            print(f"[WARN] '{self.pilz_name}' nicht in JSON gefunden.")

        # Testfälle (Frage, erwartete Antwort, Kategorie, Kontext)
        self.test_cases = self._build_test_cases()
        self.results = []           # Liste dicts: question, category, context, expected, answer
        self.analysis_df = None     # per-Question Analyse
        self.ragas_scores = None    # Aggregierte RAGAS-Metriken (falls verfügbar)

    # -----------------------------
    # Ollama Interaktion
    # -----------------------------
    def check_ollama(self):
        """Prüfe Ollama Installation (Linux/macOS/Windows)."""
        possible_paths = [
            "ollama",
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            rf"C:\Users\{os.environ.get('USERNAME', '')}\AppData\Local\Programs\Ollama\ollama.exe",
            r"C:\Program Files\Ollama\ollama.exe",
            r"C:\Program Files (x86)\Ollama\ollama.exe",
        ]
        for path in possible_paths:
            if shutil.which(path) or (path.endswith(".exe") and Path(path).exists()):
                return path
        return None

    def query_ollama(self, prompt: str, model: str = None) -> str:
        """Direkte Ollama-Abfrage (CLI)."""
        model = model or self.model
        ollama_path = self.check_ollama()
        if not ollama_path:
            return "❌ Ollama nicht gefunden"

        try:
            # Prompt als Argument (funktioniert plattformübergreifend)
            result = subprocess.run(
                [ollama_path, "run", model, prompt],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=120
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return f"❌ Ollama Fehler: {result.stderr.strip() or 'Unbekannter Fehler'}"
        except subprocess.TimeoutExpired:
            return "❌ Ollama Timeout (120s)"
        except Exception as e:
            return f"❌ Unerwarteter Ollama Fehler: {str(e)}"

    # -----------------------------
    # Testfall-Erzeugung
    # -----------------------------
    def _build_test_cases(self):
        """Erzeugt standardisierte Testfälle aus self.pilz_info."""
        if not self.pilz_info:
            return []

        def exp(path, default=""):
            return _safe_json_dump(_nested_get(self.pilz_info, path, default))

        context = _safe_json_dump(self.pilz_info)

        cases = [
            {
                "question": "Ist dieser Pilz essbar?",
                "expected": exp(["verzehr", "essbar"], "unbekannt"),
                "category": "Essbarkeit",
                "context": context,
            },
            {
                "question": "Wie sieht dieser Pilz aus?",
                "expected": exp(["aussehen"], {}),
                "category": "Aussehen",
                "context": context,
            },
            {
                "question": "Wo findet man diesen Pilz?",
                "expected": exp(["vorkommen"], {}),
                "category": "Vorkommen",
                "context": context,
            },
            {
                "question": "Wann ist die beste Zeit zum Sammeln?",
                "expected": exp(["saison"], "Saison unbekannt"),
                "category": "Saison",
                "context": context,
            },
            {
                "question": "Mit welchen Pilzen kann man ihn verwechseln?",
                "expected": exp(["verwechslungsgefahr"], []),
                "category": "Verwechslung",
                "context": context,
            },
            {
                "question": "Wie lautet der lateinische Name?",
                "expected": exp(["bezeichnung", "lateinischer_name"], "unbekannt"),
                "category": "Taxonomie",
                "context": context,
            },
        ]
        print(f"[INFO] {len(cases)} Testfälle erstellt.")
        return cases

    # -----------------------------
    # Ausführung & Datensammlung
    # -----------------------------
    def run_ollama_evaluation(self, model: str = None):
        """Fragt alle Testfälle bei Ollama an und speichert Antworten."""
        if not self.test_cases:
            print("[ERROR] Keine Testfälle vorhanden (keine Pilzdaten?).")
            return []

        model = model or self.model
        print(f"\n🚀 Starte Ollama-Evaluation mit Modell: {model}")

        results = []
        for i, tc in enumerate(self.test_cases, 1):
            q = tc["question"]
            print(f"\n[{i}/{len(self.test_cases)}] 📝 Frage: {q}")
            prompt = (
                "Beantworte die folgende Frage faktengetreu basierend auf diesen Pilzdaten:\n\n"
                f"{tc['context']}\n\nFrage: {q}\nAntwort (knapp, präzise):"
            )
            answer = self.query_ollama(prompt, model=model)
            print(f"💡 Antwort: {answer[:200]}...")
            results.append({
                "question": q,
                "category": tc["category"],
                "expected": tc["expected"],
                "context": tc["context"],
                "answer": answer
            })

        self.results = results
        return results

    # -----------------------------
    # Detaillierte Analyse
    # -----------------------------
    def detailed_analysis(self):
        """Berechnet pro Frage einfache, robuste Metriken und erstellt Zusammenfassungen.
        - has_answer: ob eine nicht-leere, nicht-offensichtliche Fehlerantwort vorliegt
        - answer_length: Zeichenanzahl
        - jaccard_overlap: Wortmengen-Jaccard zwischen Antwort und Ground Truth
        - contains_gt_substring: Ground-Truth-Teilstring in Antwort (normalisiert)
        - exact_latin_name_match: Spezialfall für 'Lateinischer Name'
        Zusätzlich (falls RAGAS verfügbar): Aggregierte RAGAS-Metriken.
        """
        if not self.results:
            print("[WARN] Keine Ergebnisse zum Analysieren. Bitte zuerst run_ollama_evaluation() aufrufen.")
            return None

        rows = []
        for r in self.results:
            ans = r.get("answer", "") or ""
            exp = r.get("expected", "") or ""

            # Flags / Längen
            has_answer = bool(_normalize_text(ans)) and not any(k in ans.lower() for k in [
                "❌", "error", "fehler", "nicht gefunden", "keine information"
            ])
            answer_length = len(ans)

            # Overlap
            s_ans = _token_set(ans)
            s_exp = _token_set(exp)
            jaccard = (len(s_ans & s_exp) / len(s_ans | s_exp)) if (s_ans or s_exp) else 0.0

            # Substring (normalisiert)
            norm_ans = _normalize_text(ans)
            norm_exp = _normalize_text(exp)
            contains_sub = norm_exp[:80] in norm_ans if len(norm_exp) >= 10 else (norm_exp in norm_ans and len(norm_exp) > 0)

            # Spezial: Lateinischer Name exakt?
            exact_latin = None
            if r.get("category") == "Taxonomie":
                # Häufig liegen in expected nur der Name, daher exakte Gleichheit zulassen
                exact_latin = _normalize_text(exp) in norm_ans

            rows.append({
                "question": r.get("question"),
                "category": r.get("category"),
                "answer": ans,
                "expected": exp,
                "answer_length": answer_length,
                "has_answer": has_answer,
                "jaccard_overlap": round(jaccard, 4),
                "contains_gt_substring": bool(contains_sub),
                "exact_latin_name_match": exact_latin if exact_latin is not None else "",
            })

        df = pd.DataFrame(rows)
        self.analysis_df = df

        # Speichern per-Question
        perq_path = self.output_dir / "ollama_per_question_analysis.csv"
        df.to_csv(perq_path, index=False, encoding="utf-8")
        print(f"\n[OK] Per-Question-Analyse gespeichert unter: {perq_path}")

        # Aggregierte Übersicht
        summary = {
            "num_questions": len(df),
            "answer_rate": float(df["has_answer"].mean()) if len(df) else 0.0,
            "avg_answer_length": float(df["answer_length"].mean()) if len(df) else 0.0,
            "mean_jaccard_overlap": float(df["jaccard_overlap"].mean()) if len(df) else 0.0,
            "contains_gt_substring_rate": float(df["contains_gt_substring"].mean()) if len(df) else 0.0,
        }

        # Kategorie-Zusammenfassung
        cat_summary = (
            df.groupby("category")[["jaccard_overlap", "has_answer", "contains_gt_substring"]]
            .mean()
            .reset_index()
            .sort_values("jaccard_overlap", ascending=False)
        )
        cat_path = self.output_dir / "ollama_category_summary.csv"
        cat_summary.to_csv(cat_path, index=False, encoding="utf-8")
        print(f"[OK] Kategorie-Zusammenfassung gespeichert unter: {cat_path}")

        # Top-Fehlerfälle (niedrigster Overlap)
        worst = df.sort_values("jaccard_overlap", ascending=True).head(min(3, len(df)))
        worst_path = self.output_dir / "ollama_worst_cases.csv"
        worst.to_csv(worst_path, index=False, encoding="utf-8")
        print(f"[OK] Schlechteste Fälle gespeichert unter: {worst_path}")

        # Optional: RAGAS-Gesamtmetriken (aggregiert)
        ragas_scores = None
        if RAGAS_AVAILABLE:
            try:
                dataset = Dataset.from_dict({
                    "question": df["question"].astype(str).tolist(),
                    "answer": df["answer"].astype(str).tolist(),
                    "contexts": [[c] for c in self.analysis_df["expected"].astype(str).tolist()],  # konservativ: GT als Kontext
                    "ground_truth": df["expected"].astype(str).tolist(),
                })
                metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness, answer_similarity]
                ragas_result = evaluate(dataset, metrics=metrics)

                # Versuche, auf aggregierte Scores zuzugreifen
                if isinstance(ragas_result, dict):
                    ragas_scores = {k: float(v) for k, v in ragas_result.items() if isinstance(v, (int, float))}
                else:
                    # Fallback: einige Versionen bieten .to_pandas() oder .scores
                    scores = getattr(ragas_result, "scores", None)
                    if isinstance(scores, dict):
                        ragas_scores = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
                    else:
                        df_r = getattr(ragas_result, "to_pandas", lambda: None)()
                        if df_r is not None and "score" in df_r.columns and "metric" in df_r.columns:
                            ragas_scores = {m: float(df_r[df_r["metric"] == m]["score"].mean()) for m in df_r["metric"].unique()}

                if ragas_scores:
                    self.ragas_scores = ragas_scores
                    ragas_path = self.output_dir / "ragas_aggregate_scores.json"
                    with open(ragas_path, "w", encoding="utf-8") as f:
                        json.dump(ragas_scores, f, ensure_ascii=False, indent=2)
                    print(f"[OK] RAGAS-Gesamtmetriken gespeichert unter: {ragas_path}")
                else:
                    print("[WARN] Konnte keine RAGAS-Gesamtscores extrahieren.")

            except Exception as e:
                print(f"[WARN] RAGAS Evaluation fehlgeschlagen: {e}")

        # Zusammenfassung speichern
        summary_path = self.output_dir / "ollama_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "ragas_scores": self.ragas_scores if self.ragas_scores else {},
            }, f, ensure_ascii=False, indent=2)
        print(f"[OK] Zusammenfassung gespeichert unter: {summary_path}")

        # Kurzbericht in Konsole
        print("\n===== KURZBERICHT =====")
        print(f"Fragen gesamt: {summary['num_questions']}")
        print(f"Antwort-Rate: {summary['answer_rate']:.2%}")
        print(f"Ø Antwortlänge: {summary['avg_answer_length']:.0f} Zeichen")
        print(f"Ø Jaccard-Overlap: {summary['mean_jaccard_overlap']:.3f}")
        print(f"GT-Substring-Rate: {summary['contains_gt_substring_rate']:.2%}")
        if self.ragas_scores:
            print("RAGAS (aggregiert): " + ", ".join(f"{k}={v:.3f}" for k, v in self.ragas_scores.items()))

        return df, summary, self.ragas_scores

    # -----------------------------
    # Komfort: Komplettlauf
    # -----------------------------
    def run_all(self, model: str = None):
        """Kompletter Durchlauf: Abfrage + Analyse."""
        self.run_ollama_evaluation(model=model)
        return self.detailed_analysis()


# ===========================
# Skripteintritt
# ===========================
def main():
    print("=== Starte Ollama/RAGAS Evaluation ===")

    evaluator = RAGASEvaluator(
        json_path=base_dir.parent / "Informationen_RAG.json",
        output_dir=base_dir.parent / "results",
        pilz_name="Gemeiner Steinpilz",  # ggf. anpassen
    )

    # 1) Ollama-Abfragen + 2) Detaillierte Analyse (+ optional 3) RAGAS-Aggregate)
    evaluator.run_all(model="phi3:mini")

    print("\nFertig. Ergebnisse liegen im 'results' Ordner.")


if __name__ == "__main__":
    main()
