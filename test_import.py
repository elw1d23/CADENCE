import yaml
from detector.main_detector import PorpoiseDetector

print("Loading config...")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
print("Initialising detector ....")
detector = PorpoiseDetector("config.yaml")

print("All imports and initialising successful")