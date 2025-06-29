# Training Progress Projector

A Python tool for analyzing and projecting your training and weight progress.

## Features
- Record training sessions (type, duration, distance, intensity)
- Project your calorie consumption and weight progress
- Visualize real and projected values
- Incorporate real weight measurements for realistic projections
- Automatically adjust the calorie model based on your actual progress

## Installation
1. Python 3.11 (recommended: Miniconda)
2. Create an environment:
   ```
   conda create -n training-progress python=3.11 numpy matplotlib -y
   conda activate training-progress
   ```
3. Clone the repository and install additional packages if needed:
   ```
   git clone <repo-url>
   cd training-progress-projector
   pip install -r requirements.txt  # if available
   ```

## Usage
Start the tool with:
```
python training-progress-projector.py
```
Follow the terminal instructions to enter training data and weight.

## Using the Data Templates

1. Copy `training_data_template.csv` to `training_data.csv` and enter your real training data.
2. Copy `weight_log_template.csv` to `weight_log.csv` and enter your real weight records.
3. The files `training_data.csv`, `weight_log.csv`, and automatically generated JSON files are not included in the repository.

**Note:** The templates serve as a guide for the data structure. Your real data remains private.

## Data Structure
- `training_data.csv`: Your training sessions (date, type, duration, distance, kcal, intensity, note)
- `weight_log.csv`: Your weight measurements (date, weight)
- `prognosis_history.json`: Prognosis history (created automatically, not in the repo!)

## Data Privacy
- Do not enter real personal data into the public repository!
- The CSV files in the repo are only templates/examples.
- Your real data is stored and processed locally.

## Further Development
- The tool can easily be extended with new training types, additional analyses, or a web interface.
- Pull requests and issues are welcome!

---

## Deutsche Anleitung

Ein Python-Tool zur Analyse und Prognose deines Trainings- und Gewichtsverlaufs.

### Features
- Erfasse Trainingseinheiten (Art, Dauer, Distanz, Intensität)
- Prognose deines Kalorienverbrauchs und Gewichtsverlaufs
- Visualisierung von realen und prognostizierten Werten
- Berücksichtigung echter Gewichtsmessungen für realistische Prognosen
- Automatische Korrektur des Kalorienmodells anhand deines echten Fortschritts

### Installation
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

### Nutzung
Starte das Tool mit:
```
python training-progress-projector.py
```
Folge den Anweisungen im Terminal, um Trainingsdaten und Gewicht einzutragen.

### Nutzung der Daten-Templates

1. Kopiere `training_data_template.csv` zu `training_data.csv` und trage deine echten Trainingsdaten ein.
2. Kopiere `weight_log_template.csv` zu `weight_log.csv` und trage deine echten Gewichtsverläufe ein.
3. Die Dateien `training_data.csv`, `weight_log.csv` sowie automatisch erzeugte JSON-Dateien werden nicht ins Repository übernommen.

**Hinweis:** Die Templates dienen als Vorlage für die Struktur der Daten. Echte Nutzerdaten bleiben privat.

### Datenstruktur
- `training_data.csv`: Deine Trainingseinheiten (Datum, Art, Dauer, Distanz, kcal, Intensität, Notiz)
- `weight_log.csv`: Deine Gewichtsmessungen (Datum, Gewicht)
- `prognosis_history.json`: Prognose-Verlauf (wird automatisch erstellt, nicht ins Repo!)

### Datenschutz
- Trage keine echten persönlichen Daten ins öffentliche Repository ein!
- Die CSV-Dateien im Repo sind nur Templates/Beispiele.
- Deine echten Daten werden lokal gespeichert und verarbeitet.

### Weiterentwicklung
- Das Tool kann leicht um neue Trainingsarten, weitere Analysen oder eine Web-Oberfläche erweitert werden.
- Pull Requests und Issues sind willkommen! 