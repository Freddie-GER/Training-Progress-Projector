import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
from shutil import copyfile
import matplotlib.dates as mdates
from json.decoder import JSONDecodeError

@dataclass
class TrainingSession:
    """Represents a single training session"""
    date: datetime
    training_type: str
    duration_minutes: int
    distance_km: float
    kcal: int
    intensity: int
    notes: str = ""

def save_weight_log(date: datetime, weight: float, filename: str = "weight_log.csv"):
    exists = os.path.exists(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not exists:
            writer.writerow(["date", "weight"])
        writer.writerow([date.strftime('%Y-%m-%d'), weight])

def load_weight_log(filename: str = "weight_log.csv") -> List[Tuple[datetime, float]]:
    if not os.path.exists(filename):
        return []
    weights = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            weight_str = row['weight'].replace(',', '.')
            weights.append((datetime.strptime(row['date'], '%Y-%m-%d'), float(weight_str)))
    return weights

class TrainingProgressProjector:
    def __init__(self, bmr: int = 1889, epoc_factor: float = 0.1, 
                 initial_weight: float = 112.0, height_cm: float = 170):
        self.bmr = bmr
        self.epoc_factor = epoc_factor
        self.initial_weight = initial_weight
        self.current_weight = initial_weight
        self.height_cm = height_cm
        
        # Training sessions storage
        self.training_sessions: List[TrainingSession] = []
        
        # Prognosis history for comparison
        self.prognosis_history: List[Dict] = []
        
        # Model parameters
        self.smooth_start = 96
        self.smooth_max = 768
        self.smooth_k = 0.06
        
        # Weight loss model parameters
        self.weight_loss_rate = 0.5  # kg per week for 500 kcal daily deficit
        self.metabolic_adaptation = 0.02  # 2% metabolic slowdown per 10% weight loss
        
        # Training type kcal calculations (kcal per minute)
        self.training_calories = {
            "crosstrainer": 6.0,  # kcal per minute at intensity 1
            "crosstrainer_intervall": 7.0,  # kcal per minute at intensity 1
            "spaziergang": 4.0,  # kcal per minute
            "walking_jogging": 6.5,  # kcal per minute
            "radfahren": 7.0  # kcal per minute
        }
        
        # Load existing data
        self.load_training_data()
        
    def calculate_kcal(self, training_type: str, duration_minutes: int, 
                      intensity: int = 3, distance_km: float = 0) -> int:
        """Calculate kcal based on training type, duration, and intensity"""
        base_kcal_per_min = self.training_calories.get(training_type, 6.0)
        # Intensitätsfaktor: 1.0 + 0.2 * (intensity - 1)
        intensity_factor = 1.0 + (0.2 * (intensity - 1))
        # Calculate base kcal
        kcal = int(base_kcal_per_min * duration_minutes * intensity_factor)
        # Add distance bonus for certain activities
        if training_type in ["spaziergang", "walking_jogging", "radfahren"] and distance_km > 0:
            distance_bonus = int(distance_km * 50)  # 50 kcal per km
            kcal += distance_bonus
        return kcal
    
    def add_training_session(self, date: str, training_type: str, 
                           duration_minutes: int, distance_km: float = 0,
                           intensity: int = 3, notes: str = ""):
        """Add a new training session"""
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d")
        # Falls distance_km als String mit Komma übergeben wird
        if isinstance(distance_km, str):
            distance_km = float(distance_km.replace(',', '.'))
        # Calculate kcal
        kcal = self.calculate_kcal(training_type, duration_minutes, intensity, distance_km)
        session = TrainingSession(
            date=date,
            training_type=training_type,
            duration_minutes=duration_minutes,
            distance_km=distance_km,
            kcal=kcal,
            intensity=intensity,
            notes=notes
        )
        self.training_sessions.append(session)
        self.training_sessions.sort(key=lambda x: x.date)
        # Save to CSV
        self.save_training_data()
        print(f"Training session added: {date.strftime('%Y-%m-%d')} - {training_type} ({duration_minutes}min, {kcal}kcal)")
        return session
    
    def get_daily_calories(self, start_date: datetime, days: int) -> List[Tuple[datetime, float]]:
        """Get daily calorie consumption for a given period"""
        daily_calories = []
        
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            
            # Sum all training kcal for this day
            training_kcal = sum(
                session.kcal for session in self.training_sessions
                if session.date.date() == current_date.date()
            )
            
            total_calories = self.bmr + training_kcal + (self.epoc_factor * training_kcal)
            daily_calories.append((current_date, total_calories))
        
        return daily_calories
    
    def calculate_weight_projection(self, daily_calories: List[Tuple[datetime, float]], 
                                  target_calories: int = 2400, real_weights: List[Tuple[datetime, float]] = None) -> List[Tuple[datetime, float]]:
        """Calculate weight projection based on calorie deficit and fit to real weights if available"""
        weight_projection = []
        current_weight = self.initial_weight
        dates = [date for date, _ in daily_calories]
        
        # If real weights are available, fit a regression
        if real_weights and len(real_weights) >= 2:
            # Fit linear regression to real weights
            x = np.array([(date - real_weights[0][0]).days for date, _ in real_weights])
            y = np.array([w for _, w in real_weights])
            slope, intercept = np.polyfit(x, y, 1)
            for i, date in enumerate(dates):
                days_since_start = (date - real_weights[0][0]).days
                pred_weight = slope * days_since_start + intercept
                weight_projection.append((date, pred_weight))
            return weight_projection
        
        # Default: calorie-based model
        for date, calories in daily_calories:
            deficit = target_calories - calories
            if deficit > 0:
                weight_loss = (deficit * 7) / 7700  # weekly weight loss
                weight_loss_factor = 1 - (self.metabolic_adaptation * (self.initial_weight - current_weight) / self.initial_weight)
                weight_loss *= weight_loss_factor
                current_weight -= weight_loss / 7  # daily weight loss
            weight_projection.append((date, current_weight))
        return weight_projection
    
    def _moving_average(self, values, window=3):
        if len(values) < window:
            return np.mean(values)
        return np.mean(values[-window:])

    def _linear_regression_slope(self, y):
        if len(y) < 2:
            return 0
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return slope

    def generate_prognosis(self, weeks_ahead: int = 25) -> Dict:
        """Generate a new prognosis based on current data (moving average + regression trend)"""
        if not self.training_sessions:
            return {"error": "No training data available"}
        
        start_date = min(session.date for session in self.training_sessions)
        end_date = start_date + timedelta(weeks=weeks_ahead)
        days_total = (end_date - start_date).days
        
        # Get real data
        real_days = (max(session.date for session in self.training_sessions) - start_date).days + 1
        real_calories = self.get_daily_calories(start_date, real_days)
        
        # Prepare session lists by type
        stepper_sessions = [s for s in self.training_sessions if s.training_type == "crosstrainer"]
        interval_sessions = [s for s in self.training_sessions if s.training_type == "crosstrainer_intervall"]
        walking_sessions = [s for s in self.training_sessions if s.training_type == "spaziergang"]
        
        # Moving averages (window=3 for small data, else 7)
        stepper_ma = self._moving_average([s.kcal for s in stepper_sessions], window=min(7, len(stepper_sessions))) if stepper_sessions else 134
        interval_ma = self._moving_average([s.kcal for s in interval_sessions], window=min(7, len(interval_sessions))) if interval_sessions else 343
        walking_ma = self._moving_average([s.kcal for s in walking_sessions], window=min(7, len(walking_sessions))) if walking_sessions else 448
        
        # Linear regression slopes
        stepper_slope = self._linear_regression_slope([s.kcal for s in stepper_sessions]) if stepper_sessions else 0
        interval_slope = self._linear_regression_slope([s.kcal for s in interval_sessions]) if interval_sessions else 0
        walking_slope = self._linear_regression_slope([s.kcal for s in walking_sessions]) if walking_sessions else 0
        
        # Generate prognosis for remaining days
        prognosis_calories = []
        current_date = start_date + timedelta(days=real_days)
        future_stepper_count = 0
        future_interval_count = 0
        future_walking_count = 0
        
        correction_factor = getattr(self, 'correction_factor', 1.0)
        
        for i in range(days_total - real_days):
            weekday = current_date.weekday()  # 0=Monday, 6=Sunday
            if weekday in [0, 2]:  # Monday, Wednesday - Stepper
                future_stepper_count += 1
                kcal = stepper_ma + stepper_slope * future_stepper_count
                # Optional physiologisches Maximum (z.B. 600 kcal)
                kcal = min(kcal, 600)
                kcal = kcal * correction_factor
                total_calories = self.bmr + kcal + (self.epoc_factor * kcal)
            elif weekday == 4:  # Friday - Intervall-Stepper
                future_interval_count += 1
                kcal = interval_ma + interval_slope * future_interval_count
                kcal = min(kcal, 800)
                kcal = kcal * correction_factor
                total_calories = self.bmr + kcal + (self.epoc_factor * kcal)
            elif weekday == 6:  # Sunday - Walking
                future_walking_count += 1
                kcal = walking_ma + walking_slope * future_walking_count
                kcal = min(kcal, 1000)
                kcal = kcal * correction_factor
                total_calories = self.bmr + kcal + (self.epoc_factor * kcal)
            else:
                total_calories = self.bmr
            prognosis_calories.append((current_date, total_calories))
            current_date += timedelta(days=1)
        
        # Combine real and prognosis data
        all_calories = real_calories + prognosis_calories
        
        # Calculate weight projection
        weight_projection = self.calculate_weight_projection(all_calories)
        
        # Calculate moving average
        calories_values = [cal for _, cal in all_calories]
        moving_avg = np.convolve(calories_values, np.ones(7)/7, mode='valid')
        
        prognosis_data = {
            "start_date": start_date,
            "end_date": end_date,
            "real_calories": real_calories,
            "prognosis_calories": prognosis_calories,
            "all_calories": all_calories,
            "weight_projection": weight_projection,
            "moving_average": moving_avg,
            "generated_date": datetime.now(),
            "total_training_sessions": len(self.training_sessions),
            "progress_analysis": {
                "stepper_ma": stepper_ma,
                "interval_ma": interval_ma,
                "walking_ma": walking_ma,
                "stepper_slope": stepper_slope,
                "interval_slope": interval_slope,
                "walking_slope": walking_slope
            }
        }
        return prognosis_data
    
    def weekly_report(self) -> str:
        """Generate a weekly report comparing old vs new prognosis"""
        if len(self.prognosis_history) < 2:
            return "Not enough prognosis history for comparison"
        
        old_prognosis = self.prognosis_history[-2]
        new_prognosis = self.prognosis_history[-1]
        
        # Calculate differences
        old_final_weight = old_prognosis["weight_projection"][-1][1]
        new_final_weight = new_prognosis["weight_projection"][-1][1]
        weight_diff = new_final_weight - old_final_weight
        
        old_final_calories = old_prognosis["all_calories"][-1][1]
        new_final_calories = new_prognosis["all_calories"][-1][1]
        calories_diff = new_final_calories - old_final_calories
        
        report = f"""
=== WÖCHENTLICHER FORTSCHRITTSBERICHT ===
Datum: {datetime.now().strftime('%Y-%m-%d')}

GEWICHTSVERLAUF:
- Alte Prognose (Endgewicht): {old_final_weight:.1f} kg
- Neue Prognose (Endgewicht): {new_final_weight:.1f} kg
- Differenz: {weight_diff:+.1f} kg

KALORIENVERBRAUCH:
- Alte Prognose (Endverbrauch): {old_final_calories:.0f} kcal/Tag
- Neue Prognose (Endverbrauch): {new_final_calories:.0f} kcal/Tag
- Differenz: {calories_diff:+.0f} kcal/Tag

TRAININGSDATEN:
- Gesamte Trainingseinheiten: {new_prognosis['total_training_sessions']}
- Neue Einheiten diese Woche: {new_prognosis['total_training_sessions'] - old_prognosis['total_training_sessions']}

BEWERTUNG:
"""
        
        if weight_diff < 0:
            report += "✅ POSITIV: Deine neue Prognose zeigt besseren Gewichtsverlust!"
        else:
            report += "⚠️  ACHTUNG: Deine neue Prognose zeigt weniger Gewichtsverlust."
        
        return report
    
    def plot_progress(self, save_plot: bool = False):
        """Plot the current progress and prognosis, including real weights if available"""
        if not self.prognosis_history:
            print("No prognosis data available. Run generate_prognosis() first.")
            return
        latest_prognosis = self.prognosis_history[-1]
        dates = [date for date, _ in latest_prognosis["all_calories"]]
        calories = [cal for _, cal in latest_prognosis["all_calories"]]
        weights = [weight for _, weight in latest_prognosis["weight_projection"]]
        real_count = len(latest_prognosis["real_calories"])
        real_dates = dates[:real_count]
        real_calories = calories[:real_count]
        prognosis_dates = dates[real_count:]
        prognosis_calories = calories[real_count:]
        # Load real weights
        real_weights = load_weight_log()
        real_weight_dates = [d for d, _ in real_weights]
        real_weight_values = [w for _, w in real_weights]
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        fig.set_size_inches(10, 5.5)
        # Plot 1: Calorie consumption
        ax1.scatter(real_dates, real_calories, color='orange', s=50, label='Echte Tageswerte', zorder=5)
        num_prognosen = len(self.prognosis_history)
        farben = ['#1f77b4', '#87ceeb']  # blau, hellblau
        for i, idx in enumerate(range(-2, 0)):
            if num_prognosen + idx >= 0:
                old_prognosis = self.prognosis_history[idx]
                real_count_old = len(old_prognosis["real_calories"])
                old_dates = [date for date, _ in old_prognosis["all_calories"]][real_count_old:]
                old_calories = [cal for _, cal in old_prognosis["all_calories"]][real_count_old:]
                # Nur Moving Average der alten Prognose anzeigen
                if len(old_calories) >= 7:
                    old_ma = np.convolve(old_calories, np.ones(7)/7, mode='valid')
                    old_ma_dates = old_dates[6:]
                    ax1.plot(old_ma_dates, old_ma, color=farben[i], linestyle=':', linewidth=2, label=f'MA Prognose Woche {num_prognosen+idx+1}')
        # Aktuelle Prognose
        ax1.plot(prognosis_dates, prognosis_calories, color='green', alpha=0.7, label='Prognose (aktuell)')
        # Moving Average der aktuellen Prognose
        if len(prognosis_calories) >= 7:
            prognosis_ma = np.convolve(prognosis_calories, np.ones(7)/7, mode='valid')
            prognosis_ma_dates = prognosis_dates[6:]
            ax1.plot(prognosis_ma_dates, prognosis_ma, color='green', linestyle=':', linewidth=2, label='MA Prognose (aktuell)')
        ma_dates = dates[6:]
        ma_values = latest_prognosis["moving_average"]
        ax1.plot(ma_dates, ma_values, color='red', linestyle='--', linewidth=2, label='Moving Average (7 Tage, real)')
        ax1.set_title('Tagesverbrauch: Reale Werte & Prognosen', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Tagesverbrauch (kcal)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Plot 2: Weight projection
        ax2.plot(dates, weights, color='blue', linewidth=2, label='Prognostiziertes Gewicht (Kalorienmodell)')
        ax2.axhline(y=self.initial_weight, color='gray', linestyle=':', alpha=0.7, label=f'Startgewicht ({self.initial_weight} kg)')
        if real_weights:
            ax2.scatter(real_weight_dates, real_weight_values, color='red', s=80, zorder=6, label='Echtes Gewicht')
            # Regression durch echte Gewichte
            if len(real_weights) >= 2:
                x = np.array([(d - real_weight_dates[0]).days for d in dates])
                x_real = np.array([(d - real_weight_dates[0]).days for d in real_weight_dates])
                y_real = np.array(real_weight_values)
                slope, intercept = np.polyfit(x_real, y_real, 1)
                y_pred = slope * x + intercept
                ax2.plot(dates, y_pred, color='green', linestyle='--', linewidth=2, label='Prognostiziertes Gewicht (Trend aus echten Werten)')
        ax2.set_title('Gewichtsverlauf: Prognose', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Datum')
        ax2.set_ylabel('Gewicht (kg)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_plot:
            plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'training_progress.png'")
        plt.show()
    
    def save_training_data(self, filename: str = "training_data.csv"):
        """Save training data to CSV file"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['date', 'training_type', 'duration_minutes', 'distance_km', 'kcal', 'intensity', 'notes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for session in self.training_sessions:
                writer.writerow({
                    'date': session.date.strftime('%Y-%m-%d'),
                    'training_type': session.training_type,
                    'duration_minutes': session.duration_minutes,
                    'distance_km': session.distance_km,
                    'kcal': session.kcal,
                    'intensity': session.intensity,
                    'notes': session.notes
                })
        
        print(f"Training data saved to {filename}")
    
    def load_training_data(self, filename: str = "training_data.csv"):
        """Load training data from CSV file"""
        if not os.path.exists(filename):
            print(f"Training data file {filename} not found. Starting with empty data.")
            return
        self.training_sessions = []
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Distanz ggf. mit Komma
                distance_str = row['distance_km'].replace(',', '.')
                session = TrainingSession(
                    date=datetime.strptime(row['date'], '%Y-%m-%d'),
                    training_type=row['training_type'],
                    duration_minutes=int(row['duration_minutes']),
                    distance_km=float(distance_str),
                    kcal=int(row['kcal']),
                    intensity=int(row['intensity']),
                    notes=row['notes']
                )
                self.training_sessions.append(session)
        self.training_sessions.sort(key=lambda x: x.date)
        print(f"Loaded {len(self.training_sessions)} training sessions from {filename}")
    
    def save_prognosis_history(self, filename: str = "prognosis_history.json"):
        """Save prognosis history to JSON file"""
        if not self.prognosis_history:
            print("No prognosis history to save")
            return
            
        # Convert datetime objects to strings for JSON serialization
        serializable_history = []
        for prognosis in self.prognosis_history:
            if "error" in prognosis:
                continue  # Skip error entries
                
            serializable_prognosis = prognosis.copy()
            serializable_prognosis['start_date'] = prognosis['start_date'].isoformat()
            serializable_prognosis['end_date'] = prognosis['end_date'].isoformat()
            serializable_prognosis['generated_date'] = prognosis['generated_date'].isoformat()
            
            # Convert datetime tuples to lists
            serializable_prognosis['real_calories'] = [(d.isoformat(), c) for d, c in prognosis['real_calories']]
            serializable_prognosis['prognosis_calories'] = [(d.isoformat(), c) for d, c in prognosis['prognosis_calories']]
            serializable_prognosis['all_calories'] = [(d.isoformat(), c) for d, c in prognosis['all_calories']]
            serializable_prognosis['weight_projection'] = [(d.isoformat(), w) for d, w in prognosis['weight_projection']]
            
            # Convert numpy array to list
            if 'moving_average' in prognosis:
                ma = prognosis['moving_average']
                if hasattr(ma, 'tolist'):
                    serializable_prognosis['moving_average'] = ma.tolist()
                else:
                    serializable_prognosis['moving_average'] = [ma] if ma is not None else []
            
            serializable_history.append(serializable_prognosis)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        
        print(f"Prognosis history saved to {filename}")
    
    def load_prognosis_history(self, filename: str = "prognosis_history.json"):
        """Load prognosis history from JSON file"""
        if not os.path.exists(filename):
            print(f"Prognosis history file {filename} not found.")
            return
        
        with open(filename, 'r', encoding='utf-8') as f:
            serializable_history = json.load(f)
        
        self.prognosis_history = []
        for prognosis in serializable_history:
            # Convert back to datetime objects
            prognosis['start_date'] = datetime.fromisoformat(prognosis['start_date'])
            prognosis['end_date'] = datetime.fromisoformat(prognosis['end_date'])
            prognosis['generated_date'] = datetime.fromisoformat(prognosis['generated_date'])
            
            # Convert back to datetime tuples
            prognosis['real_calories'] = [(datetime.fromisoformat(d), c) for d, c in prognosis['real_calories']]
            prognosis['prognosis_calories'] = [(datetime.fromisoformat(d), c) for d, c in prognosis['prognosis_calories']]
            prognosis['all_calories'] = [(datetime.fromisoformat(d), c) for d, c in prognosis['all_calories']]
            prognosis['weight_projection'] = [(datetime.fromisoformat(d), w) for d, w in prognosis['weight_projection']]
            
            self.prognosis_history.append(prognosis)
        
        print(f"Loaded {len(self.prognosis_history)} prognosis records from {filename}")

    def _get_last_sundays(self, today=None):
        """Returns the last and the second-to-last Sunday as datetime objects."""
        if today is None:
            today = datetime.now()
        # 6 = Sunday
        days_since_sunday = (today.weekday() - 6) % 7
        last_sunday = today - timedelta(days=days_since_sunday)
        last_sunday = last_sunday.replace(hour=0, minute=0, second=0, microsecond=0)
        prev_sunday = last_sunday - timedelta(days=7)
        return last_sunday, prev_sunday

    def _filter_sessions_until(self, cutoff_date):
        """Returns a list of training sessions up to and including the cutoff_date."""
        return [s for s in self.training_sessions if s.date <= cutoff_date]

    def _filter_weights_until(self, cutoff_date, filename="weight_log.csv"):
        """Returns a list of weight entries up to and including the cutoff_date."""
        weights = load_weight_log(filename)
        return [w for w in weights if w[0] <= cutoff_date]

    def generate_prognosis_for_cutoff(self, cutoff_date, weeks_ahead=25):
        """Generate a prognosis as of a specific cutoff date, using only data up to that date."""
        # Temporär gefilterte Trainingsdaten
        filtered_sessions = self._filter_sessions_until(cutoff_date)
        if not filtered_sessions:
            return {"error": "No training data available up to cutoff date"}
        # Temporär gefilterte Gewichtsdaten
        filtered_weights = self._filter_weights_until(cutoff_date)
        # Temporär ersetzen
        orig_sessions = self.training_sessions
        self.training_sessions = filtered_sessions
        # Prognose generieren
        prognosis = self.generate_prognosis(weeks_ahead=weeks_ahead)
        # Prognose mit gefilterten echten Gewichten
        if filtered_weights:
            prognosis["real_weights"] = [(d.isoformat(), w) for d, w in filtered_weights]
        # Restore
        self.training_sessions = orig_sessions
        return prognosis

    def serialize_prognosis_for_json(self, prognosis):
        """Convert all datetime objects in a prognosis dict to ISO strings for JSON serialization."""
        if "error" in prognosis:
            return prognosis.copy()
        p = prognosis.copy()
        p['start_date'] = p['start_date'].isoformat() if isinstance(p['start_date'], datetime) else p['start_date']
        p['end_date'] = p['end_date'].isoformat() if isinstance(p['end_date'], datetime) else p['end_date']
        p['generated_date'] = p['generated_date'].isoformat() if isinstance(p['generated_date'], datetime) else p['generated_date']
        # Convert lists of tuples with datetimes
        def convert_list(lst):
            return [(d.isoformat() if isinstance(d, datetime) else d, v) for d, v in lst]
        p['real_calories'] = convert_list(p['real_calories'])
        p['prognosis_calories'] = convert_list(p['prognosis_calories'])
        p['all_calories'] = convert_list(p['all_calories'])
        p['weight_projection'] = convert_list(p['weight_projection'])
        # moving_average (numpy array)
        if 'moving_average' in p:
            ma = p['moving_average']
            if hasattr(ma, 'tolist'):
                p['moving_average'] = ma.tolist()
            else:
                p['moving_average'] = [ma] if ma is not None else []
        # real_weights (optional)
        if 'real_weights' in p:
            p['real_weights'] = [(d if isinstance(d, str) else d.isoformat(), w) for d, w in p['real_weights']]
        return p

    def auto_save_weekly_snapshots(self):
        """Automatically save prognosis snapshots for last and previous Sunday if needed."""
        last_sunday, prev_sunday = self._get_last_sundays()
        last_week_file = "prognosis_last_week.json"
        two_weeks_ago_file = "prognosis_two_weeks_ago.json"
        need_update = True
        if os.path.exists(last_week_file):
            with open(last_week_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    file_date = data.get('generated_date')
                    if file_date and file_date.startswith(last_sunday.strftime('%Y-%m-%d')):
                        need_update = False
                except Exception:
                    pass
        if need_update:
            if os.path.exists(last_week_file):
                copyfile(last_week_file, two_weeks_ago_file)
            prognosis = self.generate_prognosis_for_cutoff(last_sunday)
            if 'generated_date' in prognosis:
                prognosis['generated_date'] = last_sunday.isoformat()
            else:
                prognosis['generated_date'] = last_sunday.isoformat()
            # Serialize all datetimes to strings
            prognosis_serialized = self.serialize_prognosis_for_json(prognosis)
            with open(last_week_file, 'w', encoding='utf-8') as f:
                json.dump(prognosis_serialized, f, indent=2, ensure_ascii=False)

def get_user_input(prompt: str, options: List[str] = None) -> str:
    """Get user input with optional choices"""
    while True:
        user_input = input(prompt).strip()
        if options:
            if user_input.lower() in [opt.lower() for opt in options]:
                return user_input
            print(f"Bitte wähle eine der Optionen: {', '.join(options)}")
        else:
            return user_input

def import_original_data(projector):
    """Import the original training data from the user's table with correct distances"""
    print("\n=== IMPORTIERE URSPRÜNGLICHE TRAININGSDATEN ===")
    
    original_data = [
        ("2024-01-01", "crosstrainer", 12, 1.2, 134, 3, "Stepper"),
        ("2024-01-03", "crosstrainer", 13, 1.4, 146, 3, "Stepper"),
        ("2024-01-05", "crosstrainer", 16, 1.4, 179, 3, "Stepper"),
        ("2024-01-07", "crosstrainer_intervall", 23, 2.0, 343, 2, "Intervall-Stepper"),
        ("2024-01-09", "spaziergang", 60, 5.0, 448, 3, "Strammer Spaziergang"),
        ("2024-01-11", "crosstrainer_intervall", 25, 2.3, 373, 2, "Intervall-Stepper"),
        ("2024-01-12", "crosstrainer_intervall", 30, 3.1, 413, 2, "Intervall-Stepper"),
        ("2024-01-14", "crosstrainer_intervall", 48, 4.0, 422, 2, "Intervall-Stepper")
    ]
    
    for date, training_type, duration, distance, kcal, intensity, notes in original_data:
        session = TrainingSession(
            date=datetime.strptime(date, "%Y-%m-%d"),
            training_type=training_type,
            duration_minutes=duration,
            distance_km=distance,
            kcal=kcal,
            intensity=intensity,
            notes=notes
        )
        projector.training_sessions.append(session)
    
    projector.training_sessions.sort(key=lambda x: x.date)
    projector.save_training_data()
    print(f"✅ {len(original_data)} ursprüngliche Trainingseinheiten importiert")

def calculate_correction_factor(weight_log, daily_calories, bmr):
    """Calculate correction factor for calorie consumption based on real weight loss"""
    if len(weight_log) < 2:
        return 1.0, None, None, None
    # Use the last two weight entries
    (date1, w1), (date2, w2) = weight_log[-2], weight_log[-1]
    # Find calories in the period
    cal_sum = 0
    for d, cal in daily_calories:
        if date1 < d <= date2:
            cal_sum += cal - bmr  # Only activity calories
    # Actual deficit
    kg_diff = w1 - w2
    actual_deficit = kg_diff * 7700  # kcal
    estimated_deficit = cal_sum
    if estimated_deficit == 0:
        return 1.0, actual_deficit, estimated_deficit, kg_diff
    factor = actual_deficit / estimated_deficit
    return factor, actual_deficit, estimated_deficit, kg_diff

def interactive_session():
    """Run interactive training session"""
    print("=== TRAINING PROGRESS PROJECTOR ===")
    print("Willkommen beim interaktiven Trainings-Tracker!")
    
    # Initialize projector
    projector = TrainingProgressProjector()
    
    # Load existing prognosis history
    projector.load_prognosis_history()
    
    # If no training data exists, offer to import original data
    if not projector.training_sessions:
        import_choice = get_user_input("\nKeine Trainingsdaten gefunden. Ursprüngliche Daten importieren? (ja/nein): ", ["ja", "nein"])
        if import_choice.lower() == "ja":
            import_original_data(projector)
    
    # Automatische Wochenprognose-Snapshots
    projector.auto_save_weekly_snapshots()
    
    # Get today's date
    today = datetime.now()
    print(f"\nHeute ist: {today.strftime('%A, %d. %B %Y')}")
    
    # Ask if user wants to add new training data
    add_training = get_user_input("\nNeue Trainingsdaten eintragen? (ja/nein): ", ["ja", "nein"])
    
    if add_training.lower() == "ja":
        # Get training details
        print("\n=== NEUE TRAININGSEINHEIT ===")
        
        # Training type
        print("\nWelche Art von Training hast du durchgeführt?")
        print("1) Crosstrainer + Intensitätsstufe")
        print("2) Crosstrainer Intervall + Intensität")
        print("3) Spaziergang")
        print("4) Walking & Jogging")
        print("5) Radfahren")
        
        training_choice = get_user_input("Wähle eine Option (1-5): ", ["1", "2", "3", "4", "5"])
        
        training_types = {
            "1": "crosstrainer",
            "2": "crosstrainer_intervall", 
            "3": "spaziergang",
            "4": "walking_jogging",
            "5": "radfahren"
        }
        
        training_type = training_types[training_choice]
        
        # Get intensity for crosstrainer
        intensity = 1  # default
        if training_type in ["crosstrainer", "crosstrainer_intervall"]:
            intensity_input = get_user_input("Intensitätsstufe (1-5, Standard: 1): ")
            if intensity_input:
                intensity = int(intensity_input)
        
        # Duration
        duration_input = get_user_input("Wie lange dauerte das Training in Minuten? ")
        duration_minutes = int(duration_input)
        
        # Distance
        distance_input = get_user_input("Wie viel Strecke hast du zurückgelegt? (km, 0 wenn nicht relevant): ")
        distance_km = float(distance_input) if distance_input else 0.0
        
        # Add training session
        session = projector.add_training_session(
            today.strftime('%Y-%m-%d'),
            training_type,
            duration_minutes,
            distance_km,
            intensity
        )
        
        print(f"\n✅ Training hinzugefügt: {session.kcal} kcal berechnet")
        
        # Generate new prognosis
        print("\nGeneriere neue Prognose...")
        new_prognosis = projector.generate_prognosis()
        projector.prognosis_history.append(new_prognosis)
        
        # Save prognosis history
        projector.save_prognosis_history()
        
        # Show weekly report if we have enough history
        if len(projector.prognosis_history) >= 2:
            print("\n" + "="*50)
            print(projector.weekly_report())
    else:
        # Gewicht abfragen
        weight_input = get_user_input("Möchtest du dein aktuelles Gewicht eintragen? (kg, leer für überspringen): ")
        if weight_input:
            try:
                weight = float(weight_input.replace(",", "."))
                save_weight_log(today, weight)
                print(f"✅ Gewicht {weight} kg für {today.strftime('%Y-%m-%d')} gespeichert.")
            except ValueError:
                print("Ungültige Eingabe, Gewicht wurde nicht gespeichert.")
    
    # Always show current data and plot
    print("\n=== AKTUELLE DATEN ===")
    print(f"Anzahl Trainingseinheiten: {len(projector.training_sessions)}")
    
    if projector.training_sessions:
        print("\nLetzte 5 Trainingseinheiten:")
        for session in sorted(projector.training_sessions, key=lambda x: x.date)[-5:]:
            print(f"  {session.date.strftime('%Y-%m-%d')}: {session.training_type} ({session.duration_minutes}min, {session.kcal}kcal)")
    
    if projector.prognosis_history:
        latest = projector.prognosis_history[-1]
        final_weight = latest["weight_projection"][-1][1]
        print(f"\nAktuelle Gewichtsprognose (Endgewicht): {final_weight:.1f} kg")
        print("\nGeneriere kombinierten Fortschrittsplot...")
        plot_combined_progress(projector)
    else:
        print("\nKeine Prognosedaten verfügbar. Generiere erste Prognose...")
        initial_prognosis = projector.generate_prognosis()
        if "error" not in initial_prognosis:
            projector.prognosis_history.append(initial_prognosis)
            projector.save_prognosis_history()
            plot_combined_progress(projector)
        else:
            print("Fehler beim Generieren der Prognose:", initial_prognosis["error"])

    # Correction factor output
    real_weights = load_weight_log()
    if len(real_weights) >= 5:
        # Get all daily calories
        all_dates = []
        all_cals = []
        for p in projector.prognosis_history:
            for d, c in p['all_calories']:
                all_dates.append(d)
                all_cals.append(c)
        daily_calories = list(zip(all_dates, all_cals))
        factor, actual_deficit, estimated_deficit, kg_diff = calculate_correction_factor(real_weights, daily_calories, projector.bmr)
        print(f"\n=== KALORIEN-KORREKTUR ===")
        print(f"Zeitraum: {real_weights[-2][0].strftime('%Y-%m-%d')} bis {real_weights[-1][0].strftime('%Y-%m-%d')}")
        print(f"Tatsächlicher Gewichtsverlust: {kg_diff:.2f} kg")
        print(f"Geschätztes Defizit (Training): {estimated_deficit:.0f} kcal")
        print(f"Tatsächliches Defizit (Gewicht): {actual_deficit:.0f} kcal")
        print(f"Korrekturfaktor: {factor:.2f}")
        # Optionally, apply this factor to future projections
        projector.correction_factor = factor
    else:
        print("\nFür eine sinnvolle Kalorien-Korrektur werden mindestens 5 Gewichtseinträge benötigt.")
        projector.correction_factor = 1.0

def plot_combined_progress(projector):
    """Zeigt oben den neuen Wochenvergleichsplot (Kalorien) und unten den Gewichtsprognose-Plot in einem gemeinsamen Figure."""
    import json
    import os
    import numpy as np
    from datetime import datetime
    import matplotlib.pyplot as plt
    from json.decoder import JSONDecodeError

    # Farben
    farbe_woche1 = '#1f77b4'  # Blau
    farbe_woche2 = '#87ceeb'  # Hellblau
    farbe_real = 'red'
    farbe_punkte = 'orange'

    # Prognosen laden
    prognosis_files = [
        ("prognosis_two_weeks_ago.json", farbe_woche2, 'MA Prognose Woche -2'),
        ("prognosis_last_week.json", farbe_woche1, 'MA Prognose Woche -1')
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    fig.set_size_inches(10, 5.5)
    # --- OBERER PLOT: Kalorienvergleich ---
    for file, color, label in prognosis_files:
        if os.path.exists(file) and os.path.getsize(file) > 0:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    prognosis = json.load(f)
                ma = prognosis.get('moving_average', [])
                all_calories = prognosis.get('all_calories', [])
                if len(ma) > 0 and len(all_calories) >= len(ma):
                    ma_dates = [datetime.fromisoformat(d) for d, _ in all_calories][6:6+len(ma)]
                    ax1.plot(ma_dates, ma, color=color, linestyle='-', linewidth=2, label=label)
            except (JSONDecodeError, KeyError, ValueError):
                continue
    # Echte Tageswerte und reales MA
    if len(projector.training_sessions) > 0:
        start_date = min(s.date for s in projector.training_sessions)
        end_date = max(s.date for s in projector.training_sessions)
        days = (end_date - start_date).days + 1
        real_calories = projector.get_daily_calories(start_date, days)
        real_dates = [d for d, _ in real_calories]
        real_values = [v for _, v in real_calories]
        ax1.scatter(real_dates, real_values, color=farbe_punkte, s=60, label='Echte Tageswerte', zorder=5)
        if len(real_values) >= 7:
            real_ma = np.convolve(real_values, np.ones(7)/7, mode='valid')
            real_ma_dates = real_dates[6:6+len(real_ma)]
            ax1.plot(real_ma_dates, real_ma, color=farbe_real, linestyle='--', linewidth=2, label='Moving Average (7 Tage, real)')
    ax1.set_title('Tagesverbrauch: Reale Werte & Wochenprognosen', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Tagesverbrauch (kcal)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # --- UNTERER PLOT: Gewichtsprognose ---
    if projector.prognosis_history:
        latest_prognosis = projector.prognosis_history[-1]
        dates = [date for date, _ in latest_prognosis["all_calories"]]
        weights = [weight for _, weight in latest_prognosis["weight_projection"]]
        ax2.plot(dates, weights, color='blue', linewidth=2, label='Prognostiziertes Gewicht (Kalorienmodell)')
        ax2.axhline(y=projector.initial_weight, color='gray', linestyle=':', alpha=0.7, label=f'Startgewicht ({projector.initial_weight} kg)')
        # Echte Gewichte
        real_weights = load_weight_log()
        if real_weights:
            real_weight_dates = [d for d, _ in real_weights]
            real_weight_values = [w for _, w in real_weights]
            ax2.scatter(real_weight_dates, real_weight_values, color='red', s=80, zorder=6, label='Echtes Gewicht')
            # Regression durch echte Gewichte
            if len(real_weights) >= 2:
                x = np.array([(d - real_weight_dates[0]).days for d in dates])
                x_real = np.array([(d - real_weight_dates[0]).days for d in real_weight_dates])
                y_real = np.array(real_weight_values)
                slope, intercept = np.polyfit(x_real, y_real, 1)
                y_pred = slope * x + intercept
                ax2.plot(dates, y_pred, color='green', linestyle='--', linewidth=2, label='Prognostiziertes Gewicht (Trend aus echten Werten)')
        ax2.set_title('Gewichtsverlauf: Prognose', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Datum')
        ax2.set_ylabel('Gewicht (kg)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    interactive_session()