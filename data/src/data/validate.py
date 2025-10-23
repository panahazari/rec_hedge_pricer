
import yaml
from .loader import load_data

def validate(config_path: str = "configs/project.yaml"):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    hist, fwd = load_data(cfg)
    print("Historical rows:", len(hist), "| columns:", list(hist.columns))
    print("Forwards rows:", len(fwd), "| columns:", list(fwd.columns))
    assert {'asset','market','date','he','gen_mwh','rt_hub','rt_node','da_hub','da_node'}.issubset(hist.columns)
    assert {'month','market','peak','offpeak'}.issubset(fwd.columns)
    print("Validation OK.")
