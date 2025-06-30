## [1.1.0] - 2024-06-30
### Added
- Automatische wöchentliche Prognose-Snapshots (`prognosis_last_week.json`, `prognosis_two_weeks_ago.json`) und neuen Vergleichsplot für die letzten beiden Wochenprognosen und den realen Verlauf.
- Plot ist robust gegen fehlende oder ungültige Prognosedateien.
- Unterstützung für Dezimaltrennzeichen mit Komma und Punkt bei der Eingabe und beim Einlesen von Distanz und Gewicht (CSV und CLI).

### Changed
- `.gitignore` um Prognose-Snapshots erweitert, damit diese nicht ins Repository gelangen.

### Fixed
- Keine Abstürze mehr bei fehlenden Prognose-Snapshots oder falschem Dezimaltrennzeichen in CSV-Dateien.

## [1.1.1] - 2024-06-30
### Fixed
- Kombinierter Plot (Kalorienvergleich oben, Gewichtsprognose unten) funktioniert jetzt immer korrekt und zeigt beide Verläufe zuverlässig an.
- Plot-Fenstergröße ist jetzt kompakt (ca. 1/4 Bildschirm, 10x5.5 Zoll). 