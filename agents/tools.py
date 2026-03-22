from langchain_core.tools import tool


@tool
def check_radar(env_state: dict) -> float:
    """Returns the availability score (0.0 to 1.0) of the radar sensor."""
    return env_state.get("radar_score", 0.8)


@tool
def check_ir(env_state: dict) -> float:
    """Returns the availability score (0.0 to 1.0) of the IR sensor."""
    return env_state.get("ir_score", 0.5)


@tool
def check_optical(env_state: dict) -> float:
    """Returns the availability score (0.0 to 1.0) of the optical sensor."""
    return env_state.get("optical_score", 0.3)


@tool
def fuse_sensor_data(radar_data: list, ir_data: list, optical_data: list) -> list:
    """Fuses available sensor data into a single state vector."""
    valid_data = []
    if (
        radar_data
        and len(radar_data) == 6
        and not any(isinstance(x, float) and str(x) == "nan" for x in radar_data)
    ):
        valid_data.append(radar_data)
    if (
        ir_data
        and len(ir_data) == 6
        and not any(isinstance(x, float) and str(x) == "nan" for x in ir_data)
    ):
        valid_data.append(ir_data)
    if (
        optical_data
        and len(optical_data) == 6
        and not any(isinstance(x, float) and str(x) == "nan" for x in optical_data)
    ):
        valid_data.append(optical_data)

    if not valid_data:
        return [0.0] * 6

    fused = [sum(x) / len(valid_data) for x in zip(*valid_data)]
    return fused
