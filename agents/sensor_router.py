from typing import Tuple
from langgraph.graph import StateGraph, END


class SensorOrchestrationAgent:
    def __init__(self):
        self.workflow = StateGraph(dict)

        self.workflow.add_node("assess_situation", self.assess_situation)
        self.workflow.add_node("route_radar", self.route_radar)
        self.workflow.add_node("route_ir", self.route_ir)
        self.workflow.add_node("route_optical", self.route_optical)
        self.workflow.add_node("fuse_measurements", self.fuse_measurements)
        self.workflow.add_node("handle_full_blackout", self.handle_full_blackout)

        self.workflow.set_entry_point("assess_situation")

        self.workflow.add_conditional_edges(
            "assess_situation",
            self.route_sensor_logic,
            {
                "route_radar": "route_radar",
                "route_ir": "route_ir",
                "route_optical": "route_optical",
                "handle_full_blackout": "handle_full_blackout",
            },
        )

        self.workflow.add_edge("route_radar", "fuse_measurements")
        self.workflow.add_edge("route_ir", "fuse_measurements")
        self.workflow.add_edge("route_optical", "fuse_measurements")

        self.workflow.add_edge("fuse_measurements", END)
        self.workflow.add_edge("handle_full_blackout", END)

        self.app = self.workflow.compile()

    def assess_situation(self, state: dict) -> dict:
        env_state = state.get("env_state", {})

        radar_avail = env_state.get("radar_avail", True)
        ir_avail = env_state.get("ir_avail", True)
        opt_avail = env_state.get("optical_avail", True)

        if radar_avail:
            route = "route_radar"
        elif ir_avail:
            route = "route_ir"
        elif opt_avail:
            route = "route_optical"
        else:
            route = "handle_full_blackout"

        state["next_route"] = route
        return state

    def route_sensor_logic(self, state: dict) -> str:
        return state.get("next_route", "handle_full_blackout")

    def route_radar(self, state: dict) -> dict:
        state["selected_sensor"] = "radar"
        state["weights"] = [0.8, 0.1, 0.1]
        return state

    def route_ir(self, state: dict) -> dict:
        state["selected_sensor"] = "ir"
        state["weights"] = [0.0, 0.7, 0.3]
        return state

    def route_optical(self, state: dict) -> dict:
        state["selected_sensor"] = "optical"
        state["weights"] = [0.0, 0.2, 0.8]
        return state

    def fuse_measurements(self, state: dict) -> dict:
        env_state = state.get("env_state", {})

        # Perform cross-modal fusion (simulated here, but implements logic)
        radar_m = env_state.get("radar_m", [float("nan")] * 6)
        ir_m = env_state.get("ir_m", [float("nan")] * 6)
        opt_m = env_state.get("optical_m", [float("nan")] * 6)

        # Valid measurements
        valid = []
        if not any(isinstance(x, float) and str(x) == "nan" for x in radar_m):
            valid.append(radar_m)
        if not any(isinstance(x, float) and str(x) == "nan" for x in ir_m):
            valid.append(ir_m)
        if not any(isinstance(x, float) and str(x) == "nan" for x in opt_m):
            valid.append(opt_m)

        if not valid:
            fused_measurement = env_state.get("measurement", [0.0] * 6)
        else:
            fused_measurement = [sum(x) / len(valid) for x in zip(*valid)]

        state["fused_measurement"] = fused_measurement
        return state

    def handle_full_blackout(self, state: dict) -> dict:
        state["selected_sensor"] = "none"
        state["weights"] = [0.0, 0.0, 0.0]
        state["fused_measurement"] = [float("nan")] * 6
        return state

    def run(self, env_state: dict) -> Tuple[list, list]:
        initial_state = {"env_state": env_state}
        result = self.app.invoke(initial_state)

        return result.get("fused_measurement", [float("nan")] * 6), result.get(
            "weights", [0.0, 0.0, 0.0]
        )
