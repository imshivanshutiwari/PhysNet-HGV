import yaml
import os


def load_config(config_name):
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", config_name)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_all_configs():
    return {
        "hgv": load_config("hgv_config.yaml"),
        "pinn": load_config("pinn_config.yaml"),
        "ukf": load_config("ukf_config.yaml"),
        "diffusion": load_config("diffusion_config.yaml"),
        "sensor": load_config("sensor_config.yaml"),
    }
