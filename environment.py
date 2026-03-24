# environment.py

from config import CONFIG

# Dynamically import the correct environment setting based on the configuration.
setting = CONFIG.get("environment_setting", 1)

if setting == 1:
    print("--- Loading Environment: Setting 1 (No Bike Lanes) ---")
    from environment_setting_1 import *
else:
    print("--- Loading Environment: Setting 2 (With Bike Lanes) ---")
    from environment_setting_2 import *