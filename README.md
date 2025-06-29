# Training Progress Projector

Ein Python-Tool zur Analyse und Prognose deines Trainings- und Gewichtsverlaufs.

## Features
- Erfasse Trainingseinheiten (Art, Dauer, Distanz, Intensität)
- Prognose deines Kalorienverbrauchs und Gewichtsverlaufs
- Visualisierung von realen und prognostizierten Werten
- Berücksichtigung echter Gewichtsmessungen für realistische Prognosen
- Automatische Korrektur des Kalorienmodells anhand deines echten Fortschritts

## Installation
1. Python 3.11 (empfohlen: Miniconda)
2. Erstelle ein Environment:
   ```
   conda create -n training-progress python=3.11 numpy matplotlib -y
   conda activate training-progress
   ```
3. Klone das Repository und installiere ggf. weitere Pakete:
   ```
   git clone <repo-url>
   cd training-progress-projector
   pip install -r requirements.txt  # falls vorhanden
   ```

## Nutzung
Starte das Tool mit:
```
python training-progress-projector.py
```
Folge den Anweisungen im Terminal, um Trainingsdaten und Gewicht einzutragen.

## Nutzung der Daten-Templates

1. Kopiere `training_data_template.csv` zu `training_data.csv` und trage deine echten Trainingsdaten ein.
2. Kopiere `weight_log_template.csv` zu `weight_log.csv` und trage deine echten Gewichtsverläufe ein.
3. Die Dateien `training_data.csv`, `weight_log.csv` sowie automatisch erzeugte JSON-Dateien werden nicht ins Repository übernommen.

**Hinweis:** Die Templates dienen als Vorlage für die Struktur der Daten. Echte Nutzerdaten bleiben privat.

## Datenstruktur
- `training_data.csv`: Deine Trainingseinheiten (Datum, Art, Dauer, Distanz, kcal, Intensität, Notiz)
- `weight_log.csv`: Deine Gewichtsmessungen (Datum, Gewicht)
- `prognosis_history.json`: Prognose-Verlauf (wird automatisch erstellt, nicht ins Repo!)

## Datenschutz
- Trage keine echten persönlichen Daten ins öffentliche Repository ein!
- Die CSV-Dateien im Repo sind nur Templates/Beispiele.
- Deine echten Daten werden lokal gespeichert und verarbeitet.

## Weiterentwicklung
- Das Tool kann leicht um neue Trainingsarten, weitere Analysen oder eine Web-Oberfläche erweitert werden.
- Pull Requests und Issues sind willkommen! 