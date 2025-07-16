## [1.1.0] - 2024-06-30
### Added
- Automatische wöchentliche Prognose-Snapshots (`prognosis_last_week.json`, `prognosis_two_weeks_ago.json`) und neuen Vergleichsplot für die letzten beiden Wochenprognosen und den realen Verlauf.
- Plot ist robust gegen fehlende oder ungültige Prognosedateien.
- Unterstützung für Dezimaltrennzeichen mit Komma und Punkt bei der Eingabe und beim Einlesen von Distanz und Gewicht (CSV und CLI).

### Changed
- `.gitignore` um Prognose-Snapshots erweitert, damit diese nicht ins Repository gelangen.

### Fixed
- Keine Abstürze mehr bei fehlenden Prognose-Snapshots oder falschem Dezimaltrennzeichen in CSV-Dateien.

## [1.1.2] - 2025-07-12
### Fixed
- **Critical Bug Fix**: Fixed `AttributeError: 'NoneType' object has no attribute 'isoformat'` in prognosis history serialization
- **Critical Bug Fix**: Fixed `ValueError: x and y must have same first dimension` in plotting functions
- Added robust null checks in data serialization to prevent crashes with None values
- Added array length validation in plotting functions to prevent dimension mismatches
- Enhanced weight projection calculation with safety checks for edge cases
- Improved error handling in prognosis generation and data processing

## [1.1.1] - 2024-06-30
### Fixed
- Kombinierter Plot (Kalorienvergleich oben, Gewichtsprognose unten) funktioniert jetzt immer korrekt und zeigt beide Verläufe zuverlässig an.
- Plot-Fenstergröße ist jetzt kompakt (ca. 1/4 Bildschirm, 10x5.5 Zoll). 

## [1.2.0] - 2025-07-15
### Added
- Improved calorie calculation for crosstrainer and crosstrainer_intervall: Now takes both duration and distance into account and uses the lower value to avoid overestimation. This makes calorie output more realistic when distance is much lower for the same time/intensity. 

## [1.2.1] - 2025-07-16
### Fixed
- Daily calorie calculation now sums all training sessions per day (instead of only the first), so multiple activities on the same day are correctly included in the total burn. 