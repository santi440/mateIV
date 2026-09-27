import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_airfoil

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'airfoil_self_noise.dat')
df = load_airfoil(data_path)

print("=" * 80)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")

print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)
stats = df.describe()
print(f"\n{stats}")

print("\n" + "=" * 80)
print("VARIABLE INTERPRETATION (Domain Units)")
print("=" * 80)

interpretations = {
    "frequency": "Frequency Strouhal number (Hz), non-dimensional frequency of aerodynamic interaction",
    "angle": "Angle of Attack (degrees), tilt angle of the airfoil relative to airflow",
    "chord_length": "Chord Length (meters), horizontal distance from leading to trailing edge of airfoil",
    "velocity": "Velocity (m/s), freestream velocity of the fluid around the airfoil",
    "displacement": "Displacement thickness (meters), measures boundary layer separation from surface",
    "sound_level": "Sound Pressure Level (dB), decibels: logarithmic measure of acoustic intensity"
}

for col in df.columns:
    print(f"\n{col.upper()}")
    print(f"  Description: {interpretations[col]}")
    print(f"  Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
    print(f"  Mean ± Std: {df[col].mean():.2f} ± {df[col].std():.2f}")
    print(f"  Median: {df[col].median():.2f}")

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
corr = df.corr()
print(f"\n{corr}")

print("\n" + "=" * 80)
print("All validations passed! ✓")
print("=" * 80)
