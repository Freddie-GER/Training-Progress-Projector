## [1.1.0] - 2024-06-30
### Added
- Automatische wöchentliche Prognose-Snapshots (`prognosis_last_week.json`, `prognosis_two_weeks_ago.json`) und neuen Vergleichsplot für die letzten beiden Wochenprognosen und den realen Verlauf.
- Plot ist robust gegen fehlende oder ungültige Prognosedateien.
- Unterstützung für Dezimaltrennzeichen mit Komma und Punkt bei der Eingabe und beim Einlesen von Distanz und Gewicht (CSV und CLI).

### Changed
- `.gitignore` um Prognose-Snapshots erweitert, damit diese nicht ins Repository gelangen.

### Fixed
- Keine Abstürze mehr bei fehlenden Prognose-Snapshots oder falschem Dezimaltrennzeichen in CSV-Dateien. 