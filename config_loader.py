import yaml
import os

# Individua il percorso del file di configurazione
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Impossibile trovare il file di configurazione in: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

# Esportazione dei blocchi logici come costanti globali
PATHS = _config.get("paths", {})
THRESHOLDS = _config.get("thresholds", {})
EMPATH_CATEGORIES = _config.get("empath_categories", {})