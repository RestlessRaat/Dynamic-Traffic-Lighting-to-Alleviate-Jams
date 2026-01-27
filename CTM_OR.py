# -*- coding: utf-8 -*-
"""
CTM OR model implemented with PySCIPOpt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from pyscipopt import Model, quicksum, Eventhdlr, SCIP_EVENTTYPE


# Ensure reproducible initialization
np.random.seed(42)


class ImprovementStopper(Eventhdlr):
    """Stop the solve when relative improvement between incumbents is small."""

    def __init__(self, improve_rel: float, sense: str, event_type):
        self.improve_rel = improve_rel
        self.sense = sense
        self.event_type = event_type
        self.best_obj = None

    def eventinit(self):
        if self.event_type is None:
            return
        self.model.catchEvent(self.event_type, self)

    def eventexit(self):
        if self.event_type is None:
            return
        self.model.dropEvent(self.event_type, self)

    def eventexec(self, event):
        sol = self.model.getBestSol()
        if sol is None:
            return
        obj = self.model.getSolObjVal(sol)
        if self.best_obj is None:
            self.best_obj = obj
            return

        if self.sense == "minimize":
            improvement = (self.best_obj - obj) / max(abs(self.best_obj), 1e-9)
            if obj < self.best_obj:
                self.best_obj = obj
        else:
            improvement = (obj - self.best_obj) / max(abs(self.best_obj), 1e-9)
            if obj > self.best_obj:
                self.best_obj = obj

        if improvement < self.improve_rel:
            self.model.interruptSolve()


class CTM_OR_Model:
    """Intersection CTM model formulated as an optimization problem."""

    DIRECTIONS = ["N", "E", "S", "W"]
    DIRECTION_INDEX = {"N": 0, "E": 1, "S": 2, "W": 3}
    MOVEMENTS = ["S", "L", "R"]
    MOVEMENT_INDEX = {"S": 0, "L": 1, "R": 2}
    SIGNAL_GROUPS = {"EW": ("E", "W"), "NS": ("N", "S")}
    STARTUP_PROFILE = (0.35, 0.70, 0.90, 1.0)

    def __init__(
        self,
        num_cells=40,
        v=1.0,
        w=0.5,
        rho_jam=1.0,
        num_steps=360,
        cell_length_m=100.0 / 3.0,
        cell_width_m=15.0,
        intersection_size_m=30.0,
        time_step_s=2.0,
        v_mps=50.0 / 3.0,
        w_mps=None,
        use_physical_units=True,
    ):
        self.num_cells = num_cells
        self.rho_jam = rho_jam
        self.num_steps = num_steps

        self.cell_length_m = cell_length_m
        self.cell_width_m = cell_width_m
        self.intersection_size_m = intersection_size_m
        self.time_step_s = time_step_s

        if use_physical_units:
            if v_mps is None:
                v_mps = v * self.cell_length_m / self.time_step_s
            if w_mps is None:
                w_mps = w * self.cell_length_m / self.time_step_s
            self.v = v_mps * self.time_step_s / self.cell_length_m
            self.w = w_mps * self.time_step_s / self.cell_length_m
            self.v_mps = v_mps
            self.w_mps = w_mps
        else:
            self.v = v
            self.w = w
            self.v_mps = self.v * self.cell_length_m / self.time_step_s
            self.w_mps = self.w * self.cell_length_m / self.time_step_s

        self.v_crossing = self.v
        self.Q_max = self.v * self.w * rho_jam / (self.v + self.w)
        self.Q_max_crossing = self.Q_max
        self.crossing_rho_jam = self.rho_jam

        self.turning_ratios = np.array(
            [
                [0.6, 0.15, 0.25],
                [0.6, 0.15, 0.25],
                [0.6, 0.15, 0.25],
                [0.6, 0.15, 0.25],
            ]
        )

        self.boundary_flows = {
            "N": 0.26,
            "E": 0.28,
            "S": 0.22,
            "W": 0.30,
        }

        self.turn_map = {
            "N": {"S": "S", "L": "W", "R": "E"},
            "E": {"S": "W", "L": "N", "R": "S"},
            "S": {"S": "N", "L": "E", "R": "W"},
            "W": {"S": "E", "L": "S", "R": "N"},
        }

        self.model = None
        self.decision_vars = {}
        self.objective_type = None
        self.objective_sense = None

        self.optimization_results = None
        self.rho_optimized = None
        self.flow_optimized = None
        self.initial_rho = None
        self.signal_phase = None

        self._constraint_uid = 0

        print("Initialized CTM OR model")
        print(
            "Params: v={:.3f} cell/step ({:.3f} m/s), w={:.3f} cell/step ({:.3f} m/s), "
            "rho_jam={:.3f}, Q_max={:.4f}".format(
                self.v, self.v_mps, self.w, self.w_mps, rho_jam, self.Q_max
            )
        )
        print(
            "Crossing: v_crossing={:.3f}, capacity_factor={:.2f}, Q_max_crossing={:.4f}".format(
                self.v_crossing, 1.0, self.Q_max_crossing
            )
        )
        print(
            "Cell: length={:.3f} m, width={:.3f} m, dt={:.3f} s, intersection={:.3f} m".format(
                self.cell_length_m,
                self.cell_width_m,
                self.time_step_s,
                self.intersection_size_m,
            )
        )
        print(f"Cells per direction: {num_cells}, time steps: {num_steps}")

    def define_decision_variables(self):
        print("Defining decision variables...")

        if self.model is None:
            self.model = Model("CTM_OR")

        rho = {}
        for dir_idx in range(len(self.DIRECTIONS)):
            for chain_idx in range(2):
                for cell_idx in range(self.num_cells):
                    for t in range(self.num_steps + 1):
                        name = f"rho_{dir_idx}_{chain_idx}_{cell_idx}_{t}"
                        rho[(dir_idx, chain_idx, cell_idx, t)] = self.model.addVar(
                            name=name, vtype="C", lb=0.0, ub=self.rho_jam
                        )

        flow = {}
        for from_dir in self.DIRECTIONS:
            for to_dir in self.DIRECTIONS:
                for movement in self.MOVEMENTS:
                    for t in range(self.num_steps):
                        name = f"flow_{from_dir}_{to_dir}_{movement}_{t}"
                        flow[(from_dir, to_dir, movement, t)] = self.model.addVar(
                            name=name, vtype="C", lb=0.0, ub=self.Q_max_crossing
                        )

        road_flow = {}
        for dir_idx in range(len(self.DIRECTIONS)):
            for chain_idx in range(2):
                for cell_idx in range(self.num_cells - 1):
                    for t in range(self.num_steps):
                        name = f"road_flow_{dir_idx}_{chain_idx}_{cell_idx}_{t}"
                        road_flow[(dir_idx, chain_idx, cell_idx, t)] = self.model.addVar(
                            name=name, vtype="C", lb=0.0, ub=self.Q_max
                        )

        boundary_inflow = {}
        boundary_queue = {}
        for dir_idx in range(len(self.DIRECTIONS)):
            for t in range(self.num_steps):
                name = f"boundary_inflow_{dir_idx}_{t}"
                boundary_inflow[(dir_idx, t)] = self.model.addVar(
                    name=name, vtype="C", lb=0.0, ub=self.Q_max
                )
            for t in range(self.num_steps + 1):
                name = f"boundary_queue_{dir_idx}_{t}"
                boundary_queue[(dir_idx, t)] = self.model.addVar(
                    name=name, vtype="C", lb=0.0, ub=1.0e6
                )

        crossing_inflow = {}
        for from_dir in self.DIRECTIONS:
            for t in range(self.num_steps):
                name = f"crossing_inflow_{from_dir}_{t}"
                crossing_inflow[(from_dir, t)] = self.model.addVar(
                    name=name, vtype="C", lb=0.0, ub=self.Q_max_crossing
                )

        crossing_vars = self._define_crossing_variables()

        self.decision_vars = {
            "rho": rho,
            "flow": flow,
            "road_flow": road_flow,
            "boundary_inflow": boundary_inflow,
            "boundary_queue": boundary_queue,
            "crossing_inflow": crossing_inflow,
            **crossing_vars,
        }
    def _define_crossing_variables(self):
        crossing_queue = {}
        for from_dir in self.DIRECTIONS:
            for t in range(self.num_steps + 1):
                crossing_queue[(from_dir, t)] = self.model.addVar(
                    name=f"crossing_queue_{from_dir}_{t}",
                    vtype="C",
                    lb=0.0,
                    ub=self.crossing_rho_jam,
                )

        current_phase = {}
        for t in range(self.num_steps + 1):
            current_phase[t] = self.model.addVar(
                name=f"current_phase_{t}", vtype="B"
            )

        phase_switch = {}
        for t in range(self.num_steps + 1):
            phase_switch[t] = self.model.addVar(
                name=f"phase_switch_{t}", vtype="B"
            )

        green_start = {}
        startup_step = {}
        startup_factor = {}
        for group in self.SIGNAL_GROUPS:
            for t in range(self.num_steps + 1):
                green_start[(group, t)] = self.model.addVar(
                    name=f"green_start_{group}_{t}", vtype="B"
                )
                startup_factor[(group, t)] = self.model.addVar(
                    name=f"startup_factor_{group}_{t}", vtype="C", lb=0.0, ub=1.0
                )
                for step in range(1, 5):
                    startup_step[(group, t, step)] = self.model.addVar(
                        name=f"startup_step_{group}_{step}_{t}", vtype="B"
                    )

        return {
            "crossing_queue": crossing_queue,
            "current_phase": current_phase,
            "phase_switch": phase_switch,
            "green_start": green_start,
            "startup_step": startup_step,
            "startup_factor": startup_factor,
        }

    def sending_function(self, rho_i):
        return min(self.v * rho_i, self.Q_max)

    def receiving_function(self, rho_i):
        return min(self.w * (self.rho_jam - rho_i), self.Q_max)

    def _startup_profile(self):
        return self.STARTUP_PROFILE

    def _force_flow_min(
        self, flow_var, send_cap, recv_cap, max_cap, name_prefix, max_cap_value=None
    ):
        if max_cap_value is None:
            max_cap_value = self.Q_max
        big_m = max(self.v, self.w) * self.rho_jam + max_cap_value
        uid = self._constraint_uid
        self._constraint_uid += 1
        prefix = f"{name_prefix}_{uid}"

        z_send = self.model.addVar(name=f"{prefix}_z_send", vtype="B")
        z_recv = self.model.addVar(name=f"{prefix}_z_recv", vtype="B")
        z_max = self.model.addVar(name=f"{prefix}_z_max", vtype="B")

        self.model.addCons(
            flow_var >= send_cap - big_m * (1 - z_send),
            name=f"{prefix}_min_send",
        )
        self.model.addCons(
            flow_var >= recv_cap - big_m * (1 - z_recv),
            name=f"{prefix}_min_recv",
        )
        self.model.addCons(
            flow_var >= max_cap - big_m * (1 - z_max),
            name=f"{prefix}_min_cap",
        )
        self.model.addCons(
            z_send + z_recv + z_max == 1,
            name=f"{prefix}_min_select",
        )

    def build_objective(self, objective_type="min_delay"):
        print(f"Building objective: {objective_type}")

        if self.model is None:
            raise ValueError("Call define_decision_variables() before build_objective().")

        if objective_type == "min_delay":
            total_delay = quicksum(
                self.decision_vars["rho"][(dir_idx, chain_idx, cell_idx, t)]
                for dir_idx in range(len(self.DIRECTIONS))
                for chain_idx in range(2)
                for cell_idx in range(self.num_cells)
                for t in range(self.num_steps + 1)
            )
            total_delay += quicksum(
                self.decision_vars["crossing_queue"][(from_dir, t)]
                for from_dir in self.DIRECTIONS
                for t in range(self.num_steps + 1)
            )
            total_delay += quicksum(
                self.decision_vars["boundary_queue"][(dir_idx, t)]
                for dir_idx in range(len(self.DIRECTIONS))
                for t in range(self.num_steps + 1)
            )
            self.model.setObjective(total_delay, sense="minimize")
            self.objective_sense = "minimize"
        elif objective_type == "max_throughput":
            total_throughput = quicksum(
                self.decision_vars["flow"][(from_dir, to_dir, movement, t)]
                for from_dir in self.DIRECTIONS
                for to_dir in self.DIRECTIONS
                for movement in self.MOVEMENTS
                for t in range(self.num_steps)
            )
            self.model.setObjective(total_throughput, sense="maximize")
            self.objective_sense = "maximize"
        elif objective_type == "min_density":
            total_density = quicksum(
                self.decision_vars["rho"][(dir_idx, chain_idx, cell_idx, t)]
                for dir_idx in range(len(self.DIRECTIONS))
                for chain_idx in range(2)
                for cell_idx in range(self.num_cells)
                for t in range(self.num_steps + 1)
            )
            avg_density = total_density / (4 * 2 * self.num_cells * (self.num_steps + 1))
            self.model.setObjective(avg_density, sense="minimize")
            self.objective_sense = "minimize"
        else:
            raise ValueError(f"Unsupported objective type: {objective_type}")

        self.objective_type = objective_type

    def add_ctm_constraints(self):
        print("Adding CTM constraints...")

        constraint_counter = 0
        constraint_counter = self._add_road_constraints(constraint_counter)
        constraint_counter = self._add_crossing_constraints(constraint_counter)
        self._add_signal_constraints(constraint_counter)
    def _add_road_constraints(self, constraint_counter):
        rho = self.decision_vars["rho"]
        flow = self.decision_vars["flow"]
        road_flow = self.decision_vars["road_flow"]
        boundary_inflow = self.decision_vars["boundary_inflow"]
        boundary_queue = self.decision_vars["boundary_queue"]
        crossing_inflow = self.decision_vars["crossing_inflow"]
        crossing_queue = self.decision_vars["crossing_queue"]
        startup_factor = self.decision_vars["startup_factor"]

        for t in range(self.num_steps):
            for dir_idx in range(len(self.DIRECTIONS)):
                from_dir = self.DIRECTIONS[dir_idx]
                demand = self.boundary_flows[from_dir]
                f_in = boundary_inflow[(dir_idx, t)]
                q_in = boundary_queue[(dir_idx, t)]
                q_next = boundary_queue[(dir_idx, t + 1)]

                flow_0_1 = road_flow[(dir_idx, 0, 0, t)]
                self.model.addCons(
                    flow_0_1 <= self.v * rho[(dir_idx, 0, 0, t)],
                    name=f"up_flow_ub1_{constraint_counter}",
                )
                self.model.addCons(
                    flow_0_1 <= self.w * (self.rho_jam - rho[(dir_idx, 0, 1, t)]),
                    name=f"up_flow_ub2_{constraint_counter}",
                )
                self.model.addCons(
                    flow_0_1 <= self.Q_max, name=f"up_flow_ub3_{constraint_counter}"
                )
                self.model.addCons(
                    f_in <= demand + q_in,
                    name=f"boundary_in_demand_{constraint_counter}",
                )
                self.model.addCons(
                    f_in <= self.w * (self.rho_jam - rho[(dir_idx, 0, 0, t)]),
                    name=f"boundary_in_recv_{constraint_counter}",
                )
                self.model.addCons(
                    f_in <= self.Q_max,
                    name=f"boundary_in_cap_{constraint_counter}",
                )
                self.model.addCons(
                    q_next == q_in + demand - f_in,
                    name=f"boundary_queue_evolve_{constraint_counter}",
                )
                self.model.addCons(
                    rho[(dir_idx, 0, 0, t + 1)]
                    == rho[(dir_idx, 0, 0, t)] + f_in - flow_0_1,
                    name=f"upstream_boundary_{constraint_counter}",
                )
                constraint_counter += 1

                for cell_idx in range(1, self.num_cells - 1):
                    inflow = road_flow[(dir_idx, 0, cell_idx - 1, t)]
                    outflow = road_flow[(dir_idx, 0, cell_idx, t)]

                    self.model.addCons(
                        outflow <= self.v * rho[(dir_idx, 0, cell_idx, t)],
                        name=f"up_mid_ub1_{constraint_counter}",
                    )
                    self.model.addCons(
                        outflow
                        <= self.w * (self.rho_jam - rho[(dir_idx, 0, cell_idx + 1, t)]),
                        name=f"up_mid_ub2_{constraint_counter}",
                    )
                    self.model.addCons(
                        outflow <= self.Q_max, name=f"up_mid_ub3_{constraint_counter}"
                    )
                    self.model.addCons(
                        rho[(dir_idx, 0, cell_idx, t + 1)]
                        == rho[(dir_idx, 0, cell_idx, t)] + inflow - outflow,
                        name=f"upstream_evolve_{constraint_counter}",
                    )
                    constraint_counter += 1

                last_up_cell_idx = self.num_cells - 1
                inflow = road_flow[(dir_idx, 0, last_up_cell_idx - 1, t)]
                cross_in = crossing_inflow[(from_dir, t)]
                group = "EW" if from_dir in ("E", "W") else "NS"
                ramp_cap = self.Q_max * startup_factor[(group, t)]
                self.model.addCons(
                    cross_in <= self.v * rho[(dir_idx, 0, last_up_cell_idx, t)],
                    name=f"cross_in_send_{constraint_counter}",
                )
                self.model.addCons(
                    cross_in
                    <= self.w * (self.crossing_rho_jam - crossing_queue[(from_dir, t)]),
                    name=f"cross_in_recv_{constraint_counter}",
                )
                self.model.addCons(
                    cross_in <= ramp_cap,
                    name=f"cross_in_cap_{constraint_counter}",
                )
                self.model.addCons(
                    rho[(dir_idx, 0, last_up_cell_idx, t + 1)]
                    == rho[(dir_idx, 0, last_up_cell_idx, t)]
                    + inflow
                    - cross_in,
                    name=f"upstream_last_{constraint_counter}",
                )
                constraint_counter += 1
                dest_dir = self.DIRECTIONS[dir_idx]
                total_inflow = quicksum(
                    flow[(from_d, dest_dir, movement, t)]
                    for from_d in self.DIRECTIONS
                    for movement in self.MOVEMENTS
                    if self.turn_map[from_d][movement] == dest_dir
                )
                self.model.addCons(
                    total_inflow <= self.w * (self.rho_jam - rho[(dir_idx, 1, 0, t)]),
                    name=f"down_recv_cap_{constraint_counter}",
                )
                self.model.addCons(
                    total_inflow <= self.Q_max,
                    name=f"down_recv_qmax_{constraint_counter}",
                )

                flow_0_1 = road_flow[(dir_idx, 1, 0, t)]
                self.model.addCons(
                    flow_0_1 <= self.v * rho[(dir_idx, 1, 0, t)],
                    name=f"down_flow_ub1_{constraint_counter}",
                )
                self.model.addCons(
                    flow_0_1 <= self.w * (self.rho_jam - rho[(dir_idx, 1, 1, t)]),
                    name=f"down_flow_ub2_{constraint_counter}",
                )
                self.model.addCons(
                    flow_0_1 <= self.Q_max, name=f"down_flow_ub3_{constraint_counter}"
                )
                self.model.addCons(
                    rho[(dir_idx, 1, 0, t + 1)]
                    == rho[(dir_idx, 1, 0, t)] + total_inflow - flow_0_1,
                    name=f"downstream_first_{constraint_counter}",
                )
                constraint_counter += 1

                for cell_idx in range(1, self.num_cells - 1):
                    inflow = road_flow[(dir_idx, 1, cell_idx - 1, t)]
                    outflow = road_flow[(dir_idx, 1, cell_idx, t)]

                    self.model.addCons(
                        outflow <= self.v * rho[(dir_idx, 1, cell_idx, t)],
                        name=f"down_mid_ub1_{constraint_counter}",
                    )
                    self.model.addCons(
                        outflow
                        <= self.w * (self.rho_jam - rho[(dir_idx, 1, cell_idx + 1, t)]),
                        name=f"down_mid_ub2_{constraint_counter}",
                    )
                    self.model.addCons(
                        outflow <= self.Q_max, name=f"down_mid_ub3_{constraint_counter}"
                    )
                    self.model.addCons(
                        rho[(dir_idx, 1, cell_idx, t + 1)]
                        == rho[(dir_idx, 1, cell_idx, t)] + inflow - outflow,
                        name=f"downstream_evolve_{constraint_counter}",
                    )
                    constraint_counter += 1

                last_cell_idx = self.num_cells - 1
                inflow = road_flow[(dir_idx, 1, last_cell_idx - 1, t)]
                outflow = self.model.addVar(
                    name=f"down_outflow_{dir_idx}_{last_cell_idx}_{t}_{constraint_counter}",
                    vtype="C",
                    lb=0.0,
                    ub=self.Q_max,
                )
                self.model.addCons(
                    outflow <= self.v * rho[(dir_idx, 1, last_cell_idx, t)],
                    name=f"down_last_ub1_{constraint_counter}",
                )
                self.model.addCons(
                    outflow <= self.Q_max,
                    name=f"down_last_ub2_{constraint_counter}",
                )
                self.model.addCons(
                    rho[(dir_idx, 1, last_cell_idx, t + 1)]
                    == rho[(dir_idx, 1, last_cell_idx, t)] + inflow - outflow,
                    name=f"downstream_last_{constraint_counter}",
                )
                constraint_counter += 1

        return constraint_counter
    def _add_crossing_constraints(self, constraint_counter):
        flow = self.decision_vars["flow"]
        crossing_queue = self.decision_vars["crossing_queue"]
        crossing_inflow = self.decision_vars["crossing_inflow"]

        for from_dir in self.DIRECTIONS:
            self.model.addCons(
                crossing_queue[(from_dir, 0)] == 0,
                name=f"crossing_queue_init_{constraint_counter}",
            )
            constraint_counter += 1

        for t in range(self.num_steps):
            for from_dir in self.DIRECTIONS:
                total_inflow_to_crossing = crossing_inflow[(from_dir, t)]
                total_outflow_from_crossing = quicksum(
                    flow[(from_dir, self.turn_map[from_dir][movement], movement, t)]
                    for movement in self.MOVEMENTS
                )

                self.model.addCons(
                    crossing_queue[(from_dir, t + 1)]
                    == crossing_queue[(from_dir, t)]
                    + total_inflow_to_crossing
                    - total_outflow_from_crossing,
                    name=f"crossing_queue_evolve_{constraint_counter}",
                )
                constraint_counter += 1

        return constraint_counter

    def _add_signal_constraints(self, constraint_counter):
        flow = self.decision_vars["flow"]
        current_phase = self.decision_vars["current_phase"]
        phase_switch = self.decision_vars["phase_switch"]
        green_start = self.decision_vars["green_start"]
        startup_step = self.decision_vars["startup_step"]
        startup_factor = self.decision_vars["startup_factor"]
        profile = self._startup_profile()

        self.model.addCons(current_phase[0] == 0, name="initial_phase")
        self.model.addCons(phase_switch[0] == 0, name="initial_phase_switch")
        constraint_counter += 2

        for t in range(1, self.num_steps + 1):
            self.model.addCons(
                current_phase[t] - current_phase[t - 1] <= phase_switch[t],
                name=f"phase_evolve_ub_{constraint_counter}",
            )
            self.model.addCons(
                current_phase[t - 1] - current_phase[t] <= phase_switch[t],
                name=f"phase_evolve_lb_{constraint_counter}",
            )
            constraint_counter += 2

            self.model.addCons(
                current_phase[t] + current_phase[t - 1] <= 1 + (1 - phase_switch[t]),
                name=f"phase_flip_ub_{constraint_counter}",
            )
            self.model.addCons(
                current_phase[t] + current_phase[t - 1] >= 1 - (1 - phase_switch[t]),
                name=f"phase_flip_lb_{constraint_counter}",
            )
            constraint_counter += 2

            if t >= 2:
                self.model.addCons(
                    phase_switch[t] + phase_switch[t - 1] + phase_switch[t - 2] <= 1,
                    name=f"phase_switch_freq_{constraint_counter}",
                )
                constraint_counter += 1

        for t in range(self.num_steps + 1):
            for group in self.SIGNAL_GROUPS:
                if group == "EW":
                    green_now = 1 - current_phase[t]
                    green_prev = 1 - current_phase[t - 1] if t > 0 else None
                else:
                    green_now = current_phase[t]
                    green_prev = current_phase[t - 1] if t > 0 else None

                start_var = green_start[(group, t)]
                if t == 0:
                    self.model.addCons(
                        start_var == green_now,
                        name=f"green_start_init_{group}_{constraint_counter}",
                    )
                    constraint_counter += 1
                else:
                    self.model.addCons(
                        start_var >= green_now - green_prev,
                        name=f"green_start_lb_{group}_{constraint_counter}",
                    )
                    self.model.addCons(
                        start_var <= green_now,
                        name=f"green_start_ub_{group}_{constraint_counter}",
                    )
                    self.model.addCons(
                        start_var <= 1 - green_prev,
                        name=f"green_start_prev_{group}_{constraint_counter}",
                    )
                    constraint_counter += 3

                step1 = startup_step[(group, t, 1)]
                step2 = startup_step[(group, t, 2)]
                step3 = startup_step[(group, t, 3)]
                step4 = startup_step[(group, t, 4)]

                self.model.addCons(
                    step1 == start_var,
                    name=f"startup_step1_{group}_{constraint_counter}",
                )
                constraint_counter += 1

                if t >= 1:
                    prev_start = green_start[(group, t - 1)]
                    self.model.addCons(
                        step2 <= prev_start,
                        name=f"startup_step2_ub1_{group}_{constraint_counter}",
                    )
                    self.model.addCons(
                        step2 <= green_now,
                        name=f"startup_step2_ub2_{group}_{constraint_counter}",
                    )
                    self.model.addCons(
                        step2 >= prev_start + green_now - 1,
                        name=f"startup_step2_lb_{group}_{constraint_counter}",
                    )
                    constraint_counter += 3
                else:
                    self.model.addCons(
                        step2 == 0,
                        name=f"startup_step2_zero_{group}_{constraint_counter}",
                    )
                    constraint_counter += 1

                if t >= 2:
                    prev2_start = green_start[(group, t - 2)]
                    self.model.addCons(
                        step3 <= prev2_start,
                        name=f"startup_step3_ub1_{group}_{constraint_counter}",
                    )
                    self.model.addCons(
                        step3 <= green_now,
                        name=f"startup_step3_ub2_{group}_{constraint_counter}",
                    )
                    self.model.addCons(
                        step3 >= prev2_start + green_now - 1,
                        name=f"startup_step3_lb_{group}_{constraint_counter}",
                    )
                    constraint_counter += 3
                else:
                    self.model.addCons(
                        step3 == 0,
                        name=f"startup_step3_zero_{group}_{constraint_counter}",
                    )
                    constraint_counter += 1

                self.model.addCons(
                    step1 + step2 + step3 + step4 == green_now,
                    name=f"startup_step_sum_{group}_{constraint_counter}",
                )
                constraint_counter += 1

                self.model.addCons(
                    startup_factor[(group, t)]
                    == profile[0] * step1
                    + profile[1] * step2
                    + profile[2] * step3
                    + profile[3] * step4,
                    name=f"startup_factor_{group}_{constraint_counter}",
                )
                constraint_counter += 1

        for t in range(self.num_steps):
            for from_dir in self.DIRECTIONS:
                cell_rho = self.decision_vars["crossing_queue"][(from_dir, t)]
                if from_dir in ["E", "W"]:
                    allow = 1 - current_phase[t]
                else:
                    allow = current_phase[t]

                signal_controlled_flow = quicksum(
                    flow[(from_dir, self.turn_map[from_dir][movement], movement, t)]
                    for movement in ["S", "L"]
                )
                right_turn_flow = flow[(from_dir, self.turn_map[from_dir]["R"], "R", t)]
                total_outflow = quicksum(
                    flow[(from_dir, self.turn_map[from_dir][movement], movement, t)]
                    for movement in self.MOVEMENTS
                )

                self.model.addCons(
                    total_outflow <= self.v_crossing * cell_rho,
                    name=f"sending_capacity_v_{constraint_counter}",
                )
                self.model.addCons(
                    total_outflow <= self.Q_max_crossing,
                    name=f"sending_capacity_q_{constraint_counter}",
                )
                constraint_counter += 2

                self.model.addCons(
                    signal_controlled_flow <= self.Q_max_crossing * allow,
                    name=f"signal_controlled_{constraint_counter}",
                )
                constraint_counter += 1

                self.model.addCons(
                    right_turn_flow <= self.Q_max_crossing * 0.5,
                    name=f"right_turn_cap_{constraint_counter}",
                )
                constraint_counter += 1

                for movement in self.MOVEMENTS:
                    to_dir = self.turn_map[from_dir][movement]

                    for other_dir in self.DIRECTIONS:
                        if other_dir != to_dir:
                            self.model.addCons(
                                flow[(from_dir, other_dir, movement, t)] == 0,
                                name=f"invalid_movement_{constraint_counter}",
                            )
                            constraint_counter += 1

                    if movement in ["S", "L"]:
                        self.model.addCons(
                            flow[(from_dir, to_dir, movement, t)]
                            <= self.Q_max_crossing * allow,
                            name=f"movement_allowed_{constraint_counter}",
                        )
                        constraint_counter += 1

        return constraint_counter

    def add_initial_conditions(self, initial_rho=None):
        print("Adding initial conditions...")
        rho = self.decision_vars["rho"]
        boundary_queue = self.decision_vars["boundary_queue"]

        if initial_rho is None:
            def build_initial_sequence(length):
                if length <= 0:
                    return np.zeros(0)
                vals = np.zeros(length, dtype=float)
                vals[0] = np.random.uniform(0.2, 0.3)
                for idx in range(1, length):
                    prev = vals[idx - 1]
                    low = max(0.0, prev * 0.9)
                    high = min(self.rho_jam, prev * 1.1)
                    if low > high:
                        low, high = high, low
                    vals[idx] = np.random.uniform(low, high)
                return vals

            init_rho = np.zeros((len(self.DIRECTIONS), 2, self.num_cells))
            for dir_idx in range(len(self.DIRECTIONS)):
                upstream_initial = build_initial_sequence(self.num_cells)
                downstream_initial = build_initial_sequence(self.num_cells)
                self.model.addCons(
                    boundary_queue[(dir_idx, 0)] == 0,
                    name=f"init_boundary_queue_{dir_idx}",
                )
                for cell_idx in range(self.num_cells):
                    init_rho[dir_idx, 0, cell_idx] = float(upstream_initial[cell_idx])
                    init_rho[dir_idx, 1, cell_idx] = float(downstream_initial[cell_idx])
                    self.model.addCons(
                        rho[(dir_idx, 0, cell_idx, 0)] == float(upstream_initial[cell_idx]),
                        name=f"init_rho_up_{dir_idx}_{cell_idx}",
                    )
                    self.model.addCons(
                        rho[(dir_idx, 1, cell_idx, 0)]
                        == float(downstream_initial[cell_idx]),
                        name=f"init_rho_down_{dir_idx}_{cell_idx}",
                    )
            self.initial_rho = init_rho
        else:
            init_rho = np.array(initial_rho, dtype=float)
            for dir_idx in range(len(self.DIRECTIONS)):
                self.model.addCons(
                    boundary_queue[(dir_idx, 0)] == 0,
                    name=f"init_boundary_queue_{dir_idx}",
                )
                for chain_idx in range(2):
                    for cell_idx in range(self.num_cells):
                        self.model.addCons(
                            rho[(dir_idx, chain_idx, cell_idx, 0)]
                            == float(initial_rho[dir_idx, chain_idx, cell_idx]),
                            name=f"init_rho_{dir_idx}_{chain_idx}_{cell_idx}",
                        )
            self.initial_rho = init_rho

    def solve(
        self,
        gap_rel=None,
        improve_rel=None,
        improve_chunk_seconds=30,
        improve_max_rounds=10,
    ):
        import time

        print("Solving optimization problem...")
        if self.model is None:
            raise ValueError("Model is not initialized. Call build_objective() first.")

        if gap_rel is not None:
            self.model.setParam("limits/gap", gap_rel)

        if improve_chunk_seconds is not None and improve_max_rounds is not None:
            total_time = improve_chunk_seconds * improve_max_rounds
            if total_time > 0:
                self.model.setParam("limits/time", total_time)

        event_type = None
        if improve_rel is not None:
            event_type = getattr(SCIP_EVENTTYPE, "BESTSOLFOUND", None)
            if event_type is None:
                event_type = getattr(SCIP_EVENTTYPE, "SOLFOUND", None)
            if event_type is None:
                print("Warning: SCIP solution event not available; improve_rel ignored.")
            else:
                stopper = ImprovementStopper(improve_rel, self.objective_sense, event_type)
                self.model.includeEventhdlr(
                    stopper, "improve_stop", "stop if improvement is small"
                )
                self._improve_stopper = stopper

        start_time = time.time()
        self.model.optimize()
        solve_time = time.time() - start_time

        status = self.model.getStatus()
        if self.model.getNSols() == 0:
            self.optimization_results = {
                "status": str(status),
                "objective_value": None,
                "solve_time": solve_time,
                "phase_change_count": 0,
            }
            print(f"Solve finished with status: {status}")
            return self.optimization_results

        best_sol = self.model.getBestSol()
        if best_sol is None:
            self.optimization_results = {
                "status": str(status),
                "objective_value": None,
                "solve_time": solve_time,
                "phase_change_count": 0,
            }
            print(f"Solve finished with status: {status}")
            return self.optimization_results

        try:
            objective_value = self.model.getSolObjVal(best_sol)
        except Exception:
            try:
                objective_value = self.model.getObjVal()
            except Exception:
                objective_value = None

        self.optimization_results = {
            "status": str(status),
            "objective_value": objective_value,
            "solve_time": solve_time,
            "phase_change_count": 0,
        }

        self.rho_optimized = np.zeros(
            (len(self.DIRECTIONS), 2, self.num_cells, self.num_steps + 1)
        )
        self.flow_optimized = {}

        rho = self.decision_vars["rho"]
        try:
            for dir_idx in range(len(self.DIRECTIONS)):
                for chain_idx in range(2):
                    for cell_idx in range(self.num_cells):
                        for t in range(self.num_steps + 1):
                            self.rho_optimized[dir_idx, chain_idx, cell_idx, t] = (
                                self.model.getSolVal(
                                    best_sol, rho[(dir_idx, chain_idx, cell_idx, t)]
                                )
                            )

            flow = self.decision_vars["flow"]
            for from_dir in self.DIRECTIONS:
                for to_dir in self.DIRECTIONS:
                    for movement in self.MOVEMENTS:
                        for t in range(self.num_steps):
                            key = (from_dir, to_dir, movement, t)
                            self.flow_optimized[key] = self.model.getSolVal(
                                best_sol, flow[key]
                            )

            phase_change_count = 0
            phase_switch = self.decision_vars["phase_switch"]
            for t in range(1, self.num_steps + 1):
                if self.model.getSolVal(best_sol, phase_switch[t]) > 0.5:
                    phase_change_count += 1
            self.signal_phase = [
                self.model.getSolVal(best_sol, self.decision_vars["current_phase"][t])
                for t in range(self.num_steps + 1)
            ]
        except Exception as exc:
            print(f"Warning: failed to extract solution values ({exc})")
            self.rho_optimized = None
            self.flow_optimized = None
            return self.optimization_results

        self.optimization_results["phase_change_count"] = phase_change_count

        print(f"Solve finished with status: {status}")
        if self.optimization_results["objective_value"] is not None:
            print(f"Objective value: {self.optimization_results['objective_value']:.4f}")
        else:
            print("Objective value: N/A")
        print(f"Solve time: {self.optimization_results['solve_time']:.2f} s")
        print(f"Phase changes: {self.optimization_results['phase_change_count']}")

        return self.optimization_results

    def plot_optimization_results(self, save_path=None):
        if self.rho_optimized is None:
            print("No optimized results. Run solve() first.")
            return

        time_steps = np.arange(self.num_steps + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Key density evolution ({self.objective_type})",
            fontsize=16,
            fontweight="bold",
        )

        direction_names = ["North", "East", "South", "West"]
        for dir_idx in range(4):
            ax = axes[dir_idx // 2, dir_idx % 2]
            rho_up_last = self.rho_optimized[dir_idx, 0, -1, :]
            rho_down_first = self.rho_optimized[dir_idx, 1, 0, :]
            ax.plot(time_steps, rho_up_last, "b-", linewidth=2, label="Upstream last")
            ax.plot(time_steps, rho_down_first, "r-", linewidth=2, label="Downstream first")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Density")
            ax.set_title(direction_names[dir_idx], fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, self.rho_jam * 1.1)

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_density_evolution.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

        if not self.signal_phase:
            print("No signal phase data available for plotting.")
            return

        time_axis = np.arange(len(self.signal_phase))
        ew_signal = [1 - p for p in self.signal_phase]
        ns_signal = [p for p in self.signal_phase]

        fig, ax = plt.subplots(1, 1, figsize=(12, 2.8))
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        grid = np.vstack([ew_signal, ns_signal])
        cmap = ListedColormap(["#d62728", "#2ca02c"])
        ax.pcolormesh(
            np.arange(len(time_axis) + 1),
            np.arange(3),
            grid,
            cmap=cmap,
            vmin=0,
            vmax=1,
            edgecolors="white",
            linewidth=0.2,
        )
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["EW", "NS"])
        tick_step = max(1, len(time_axis) // 10)
        tick_positions = np.arange(0, len(time_axis) + 1, tick_step)
        ax.set_xticks(tick_positions)
        ax.set_xlabel("Time step")
        ax.set_title("Signal state grid (red/green)", fontweight="bold")
        ax.set_xlim(0, len(time_axis))
        ax.set_ylim(0, 2)
        ax.set_aspect("auto")
        ax.legend(
            handles=[
                Patch(color="#d62728", label="Red"),
                Patch(color="#2ca02c", label="Green"),
            ],
            loc="upper right",
            ncol=2,
            frameon=False,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_signal_grid.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def plot_north_upstream_density(self, save_path=None):
        if self.rho_optimized is None:
            print("No optimized results. Run solve() first.")
            return

        north_up_rho = self.rho_optimized[0, 0, -1, :]
        time_steps = np.arange(self.num_steps + 1)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle(
            f"North upstream last cell density ({self.objective_type})",
            fontsize=16,
            fontweight="bold",
        )
        ax.plot(time_steps, north_up_rho, "b-", linewidth=2, alpha=0.8)
        avg_rho = np.mean(north_up_rho)
        ax.axhline(y=avg_rho, color="r", linestyle="--", linewidth=1.5, label=f"Avg {avg_rho:.4f}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Density")
        ax.set_title("North - upstream last cell", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, self.rho_jam * 1.1)
        plt.tight_layout()

        if save_path:
            save_file = f"{save_path}_north_upstream_density.png"
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
            print(f"Saved plot to: {save_file}")
        else:
            plt.show()

    def compare_with_original(self, original_ctm, save_path=None):
        print("Comparing with original CTM model...")

        if self.rho_optimized is None:
            print("No optimized results. Run solve() first.")
            return

        rho_original = np.array(original_ctm.rho_history)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Density comparison", fontsize=16, fontweight="bold")

        direction_names = ["North", "East", "South", "West"]
        for dir_idx in range(4):
            ax = axes[dir_idx // 2, dir_idx % 2]
            rho_up_last_original = rho_original[:, dir_idx, 0, -1]
            rho_up_last_optimized = self.rho_optimized[dir_idx, 0, -1, :]
            rho_down_first_original = rho_original[:, dir_idx, 1, 0]
            rho_down_first_optimized = self.rho_optimized[dir_idx, 1, 0, :]

            time_original = np.arange(len(rho_original))
            time_optimized = np.arange(self.num_steps + 1)

            ax.plot(time_original, rho_up_last_original, "b-", linewidth=2, label="Original upstream")
            ax.plot(time_optimized, rho_up_last_optimized, "b--", linewidth=2, label="Optimized upstream")
            ax.plot(time_original, rho_down_first_original, "r-", linewidth=2, label="Original downstream")
            ax.plot(time_optimized, rho_down_first_optimized, "r--", linewidth=2, label="Optimized downstream")

            ax.set_xlabel("Time step")
            ax.set_ylabel("Density")
            ax.set_title(direction_names[dir_idx], fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, self.rho_jam * 1.1)

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_comparison.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def print_optimization_summary(self):
        if self.optimization_results is None:
            print("No optimized results. Run solve() first.")
            return

        print("=== Optimization Summary ===")
        print(f"Objective: {self.objective_type}")
        print(f"Status: {self.optimization_results['status']}")
        obj_val = self.optimization_results.get("objective_value")
        if obj_val is None:
            print("Objective value: N/A")
        else:
            print(f"Objective value: {obj_val:.4f}")
        print(f"Solve time: {self.optimization_results['solve_time']:.2f} s")
        print(f"Phase changes: {self.optimization_results['phase_change_count']}")

        if self.rho_optimized is None:
            print("Average density: N/A")
        else:
            avg_density = float(np.mean(self.rho_optimized))
            print(f"Average density: {avg_density:.4f}")

        if not self.flow_optimized:
            print("Total throughput: N/A")
        else:
            total_throughput = sum(self.flow_optimized.values())
            print(f"Total throughput: {total_throughput:.4f}")

    def plot_signal_phases(self, save_path=None):
        if not self.signal_phase:
            print("No signal phase data available for plotting.")
            return

        num_plots = len(self.intersections)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3.2 * num_plots))
        if num_plots == 1:
            axes = [axes]

        for idx, inter in enumerate(self.intersections):
            phases = self.signal_phase.get(inter)
            if not phases:
                continue
            time_axis = np.arange(len(phases)) * self.time_step_s
            ew_signal = [1 - p for p in phases]
            ns_signal = [p for p in phases]
            ax = axes[idx]
            ax.step(time_axis, ew_signal, "b-", linewidth=2, where="post", label="EW green")
            ax.step(time_axis, ns_signal, "r-", linewidth=2, where="post", label="NS green")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal state")
            ax.set_title(f"Intersection {inter} signal phases", fontweight="bold")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Red", "Green"])
            ax.set_xlim(0, len(time_axis) - 1)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_signal_phases.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

        if self.signal_phase:
            print("Signal states by time step:")
            for t, phase in enumerate(self.signal_phase):
                state = "EW green" if phase < 0.5 else "NS green"
                print(f"  t={t}: {state}")

@dataclass(frozen=True)
class LinkDef:
    link_id: int
    length: int
    tail: int | None
    head: int | None
    tail_dir: str | None
    head_dir: str | None
    label: str


class CTM_OR_Network_Model:
    """Multi-intersection CTM model formulated as an optimization problem."""

    DIRECTIONS = ["N", "E", "S", "W"]
    DIRECTION_INDEX = {"N": 0, "E": 1, "S": 2, "W": 3}
    MOVEMENTS = ["S", "L", "R"]
    MOVEMENT_INDEX = {"S": 0, "L": 1, "R": 2}
    SIGNAL_GROUPS = {"EW": ("E", "W"), "NS": ("N", "S")}
    STARTUP_PROFILE = (0.35, 0.70, 0.90, 1.0)

    def __init__(
        self,
        num_intersections=2,
        internal_cell_count=5,
        external_cell_count=3,
        connection_pairs=None,
        v=1.0,
        w=0.5,
        rho_jam=1.0,
        num_steps=360,
        cell_length_m=100.0 / 3.0,
        cell_width_m=15.0,
        intersection_size_m=30.0,
        time_step_s=2.0,
        v_mps=50.0 / 3.0,
        w_mps=None,
        use_physical_units=True,
        grid_shape=None,
    ):
        self.num_intersections = num_intersections
        self.intersections = list(range(num_intersections))
        self.internal_cell_count = internal_cell_count
        self.external_cell_count = external_cell_count

        self.grid_shape = self._normalize_grid_shape(grid_shape)
        if self.grid_shape is None and connection_pairs is None and num_intersections == 4:
            self.grid_shape = (2, 2)
        self.intersection_layout = (
            self._grid_intersection_layout(*self.grid_shape) if self.grid_shape else {}
        )
        self.intersection_labels = {inter: f"I{inter}" for inter in self.intersections}
        if self.intersection_layout:
            for inter, (row, col) in self.intersection_layout.items():
                self.intersection_labels[inter] = f"I{inter}_r{row}c{col}"

        if connection_pairs is None:
            if self.grid_shape is not None:
                connection_pairs = self._grid_connection_pairs(*self.grid_shape)
            else:
                connection_pairs = [((0, "E"), (1, "W")), ((1, "W"), (0, "E"))]
        self.connection_pairs = connection_pairs

        self.rho_jam = rho_jam
        self.num_steps = num_steps

        self.cell_length_m = cell_length_m
        self.cell_width_m = cell_width_m
        self.intersection_size_m = intersection_size_m
        self.time_step_s = time_step_s

        if use_physical_units:
            if v_mps is None:
                v_mps = v * self.cell_length_m / self.time_step_s
            if w_mps is None:
                w_mps = w * self.cell_length_m / self.time_step_s
            self.v = v_mps * self.time_step_s / self.cell_length_m
            self.w = w_mps * self.time_step_s / self.cell_length_m
            self.v_mps = v_mps
            self.w_mps = w_mps
        else:
            self.v = v
            self.w = w
            self.v_mps = self.v * self.cell_length_m / self.time_step_s
            self.w_mps = self.w * self.cell_length_m / self.time_step_s

        self.v_crossing = self.v
        self.Q_max = self.v * self.w * rho_jam / (self.v + self.w)
        self.Q_max_crossing = self.Q_max
        self.crossing_rho_jam = self.rho_jam

        self.turning_ratios = np.array(
            [
                [0.6, 0.15, 0.25],
                [0.6, 0.15, 0.25],
                [0.6, 0.15, 0.25],
                [0.6, 0.15, 0.25],
            ]
        )

        self.boundary_flows = {
            "N": 0.26,
            "E": 0.28,
            "S": 0.22,
            "W": 0.30,
        }

        self.turn_map = {
            "N": {"S": "S", "L": "W", "R": "E"},
            "E": {"S": "W", "L": "N", "R": "S"},
            "S": {"S": "N", "L": "E", "R": "W"},
            "W": {"S": "E", "L": "S", "R": "N"},
        }

        self.links = []
        self.incoming_link = {}
        self.outgoing_link = {}
        self.boundary_sources = []
        self.boundary_sinks = []
        self._build_links()

        self.model = None
        self.decision_vars = {}
        self.objective_type = None
        self.objective_sense = None

        self.optimization_results = None
        self.rho_optimized = None
        self.flow_optimized = None
        self.initial_rho = None
        self.signal_phase = None

        self._constraint_uid = 0

        print("Initialized CTM OR network model")
        print(
            "Params: v={:.3f} cell/step ({:.3f} m/s), w={:.3f} cell/step ({:.3f} m/s), "
            "rho_jam={:.3f}, Q_max={:.4f}".format(
                self.v, self.v_mps, self.w, self.w_mps, rho_jam, self.Q_max
            )
        )
        print(
            "Crossing: v_crossing={:.3f}, capacity_factor={:.2f}, Q_max_crossing={:.4f}".format(
                self.v_crossing, 1.0, self.Q_max_crossing
            )
        )
        print(
            "Cell: length={:.3f} m, width={:.3f} m, dt={:.3f} s, intersection={:.3f} m".format(
                self.cell_length_m,
                self.cell_width_m,
                self.time_step_s,
                self.intersection_size_m,
            )
        )
        print(
            "Intersections: {}, links: {}, internal cells: {}, external cells: {}, time steps: {}".format(
                self.num_intersections,
                len(self.links),
                self.internal_cell_count,
                self.external_cell_count,
                self.num_steps,
            )
        )

    def _add_link(self, length, tail, head, tail_dir, head_dir):
        link_id = len(self.links)
        label = self._build_link_label(tail, head, tail_dir, head_dir)
        self.links.append(LinkDef(link_id, length, tail, head, tail_dir, head_dir, label))
        self.link_labels[link_id] = label
        for cell_idx in range(length):
            self.cell_labels[(link_id, cell_idx)] = f"{label}_c{cell_idx}"
        return link_id

    def _build_links(self):
        self.links = []
        self.incoming_link = {}
        self.outgoing_link = {}
        self.link_labels = {}
        self.cell_labels = {}

        for (tail_inter, tail_dir), (head_inter, head_dir) in self.connection_pairs:
            if tail_inter not in self.intersections or head_inter not in self.intersections:
                raise ValueError("Connection pairs reference unknown intersection.")
            if tail_dir not in self.DIRECTIONS or head_dir not in self.DIRECTIONS:
                raise ValueError("Connection pairs reference unknown direction.")
            if (tail_inter, tail_dir) in self.outgoing_link:
                raise ValueError("Duplicate outgoing connection for intersection/direction.")
            if (head_inter, head_dir) in self.incoming_link:
                raise ValueError("Duplicate incoming connection for intersection/direction.")
            link_id = self._add_link(
                self.internal_cell_count,
                tail_inter,
                head_inter,
                tail_dir,
                head_dir,
            )
            self.outgoing_link[(tail_inter, tail_dir)] = link_id
            self.incoming_link[(head_inter, head_dir)] = link_id

        for inter in self.intersections:
            for dir_name in self.DIRECTIONS:
                if (inter, dir_name) not in self.incoming_link:
                    link_id = self._add_link(
                        self.external_cell_count,
                        None,
                        inter,
                        None,
                        dir_name,
                    )
                    self.incoming_link[(inter, dir_name)] = link_id
                if (inter, dir_name) not in self.outgoing_link:
                    link_id = self._add_link(
                        self.external_cell_count,
                        inter,
                        None,
                        dir_name,
                        None,
                    )
                    self.outgoing_link[(inter, dir_name)] = link_id

        self.boundary_sources = [link.link_id for link in self.links if link.tail is None]
        self.boundary_sinks = [link.link_id for link in self.links if link.head is None]

    def _boundary_flow(self, intersection, direction):
        if (intersection, direction) in self.boundary_flows:
            return self.boundary_flows[(intersection, direction)]
        return self.boundary_flows[direction]

    def define_decision_variables(self):
        print("Defining decision variables...")

        if self.model is None:
            self.model = Model("CTM_OR_Network")

        rho = {}
        for link in self.links:
            for cell_idx in range(link.length):
                for t in range(self.num_steps + 1):
                    name = f"rho_{link.link_id}_{cell_idx}_{t}"
                    rho[(link.link_id, cell_idx, t)] = self.model.addVar(
                        name=name, vtype="C", lb=0.0, ub=self.rho_jam
                    )

        flow = {}
        for inter in self.intersections:
            for from_dir in self.DIRECTIONS:
                for to_dir in self.DIRECTIONS:
                    for movement in self.MOVEMENTS:
                        for t in range(self.num_steps):
                            name = f"flow_{inter}_{from_dir}_{to_dir}_{movement}_{t}"
                            flow[(inter, from_dir, to_dir, movement, t)] = self.model.addVar(
                                name=name, vtype="C", lb=0.0, ub=self.Q_max_crossing
                            )

        road_flow = {}
        for link in self.links:
            for cell_idx in range(link.length - 1):
                for t in range(self.num_steps):
                    name = f"road_flow_{link.link_id}_{cell_idx}_{t}"
                    road_flow[(link.link_id, cell_idx, t)] = self.model.addVar(
                        name=name, vtype="C", lb=0.0, ub=self.Q_max
                    )

        boundary_inflow = {}
        boundary_queue = {}
        for link_id in self.boundary_sources:
            for t in range(self.num_steps):
                name = f"boundary_inflow_{link_id}_{t}"
                boundary_inflow[(link_id, t)] = self.model.addVar(
                    name=name, vtype="C", lb=0.0, ub=self.Q_max
                )
            for t in range(self.num_steps + 1):
                name = f"boundary_queue_{link_id}_{t}"
                boundary_queue[(link_id, t)] = self.model.addVar(
                    name=name, vtype="C", lb=0.0, ub=1.0e6
                )

        crossing_inflow = {}
        for inter in self.intersections:
            for from_dir in self.DIRECTIONS:
                for t in range(self.num_steps):
                    name = f"crossing_inflow_{inter}_{from_dir}_{t}"
                    crossing_inflow[(inter, from_dir, t)] = self.model.addVar(
                        name=name, vtype="C", lb=0.0, ub=self.Q_max_crossing
                    )

        crossing_vars = self._define_crossing_variables()

        self.decision_vars = {
            "rho": rho,
            "flow": flow,
            "road_flow": road_flow,
            "boundary_inflow": boundary_inflow,
            "boundary_queue": boundary_queue,
            "crossing_inflow": crossing_inflow,
            **crossing_vars,
        }

    def _define_crossing_variables(self):
        crossing_queue = {}
        for inter in self.intersections:
            for from_dir in self.DIRECTIONS:
                for t in range(self.num_steps + 1):
                    crossing_queue[(inter, from_dir, t)] = self.model.addVar(
                        name=f"crossing_queue_{inter}_{from_dir}_{t}",
                        vtype="C",
                        lb=0.0,
                        ub=self.crossing_rho_jam,
                    )

        current_phase = {}
        for inter in self.intersections:
            for t in range(self.num_steps + 1):
                current_phase[(inter, t)] = self.model.addVar(
                    name=f"current_phase_{inter}_{t}", vtype="B"
                )

        phase_switch = {}
        for inter in self.intersections:
            for t in range(self.num_steps + 1):
                phase_switch[(inter, t)] = self.model.addVar(
                    name=f"phase_switch_{inter}_{t}", vtype="B"
                )

        green_start = {}
        startup_step = {}
        startup_factor = {}
        for inter in self.intersections:
            for group in self.SIGNAL_GROUPS:
                for t in range(self.num_steps + 1):
                    green_start[(inter, group, t)] = self.model.addVar(
                        name=f"green_start_{inter}_{group}_{t}", vtype="B"
                    )
                    startup_factor[(inter, group, t)] = self.model.addVar(
                        name=f"startup_factor_{inter}_{group}_{t}", vtype="C", lb=0.0, ub=1.0
                    )
                    for step in range(1, 5):
                        startup_step[(inter, group, t, step)] = self.model.addVar(
                            name=f"startup_step_{inter}_{group}_{step}_{t}", vtype="B"
                        )

        return {
            "crossing_queue": crossing_queue,
            "current_phase": current_phase,
            "phase_switch": phase_switch,
            "green_start": green_start,
            "startup_step": startup_step,
            "startup_factor": startup_factor,
        }

    def sending_function(self, rho_i):
        return min(self.v * rho_i, self.Q_max)

    def receiving_function(self, rho_i):
        return min(self.w * (self.rho_jam - rho_i), self.Q_max)

    def _startup_profile(self):
        return self.STARTUP_PROFILE

    def _force_flow_min(
        self, flow_var, send_cap, recv_cap, max_cap, name_prefix, max_cap_value=None
    ):
        if max_cap_value is None:
            max_cap_value = self.Q_max
        big_m = max(self.v, self.w) * self.rho_jam + max_cap_value
        uid = self._constraint_uid
        self._constraint_uid += 1
        prefix = f"{name_prefix}_{uid}"

        z_send = self.model.addVar(name=f"{prefix}_z_send", vtype="B")
        z_recv = self.model.addVar(name=f"{prefix}_z_recv", vtype="B")
        z_max = self.model.addVar(name=f"{prefix}_z_max", vtype="B")

        self.model.addCons(
            flow_var >= send_cap - big_m * (1 - z_send),
            name=f"{prefix}_min_send",
        )
        self.model.addCons(
            flow_var >= recv_cap - big_m * (1 - z_recv),
            name=f"{prefix}_min_recv",
        )
        self.model.addCons(
            flow_var >= max_cap - big_m * (1 - z_max),
            name=f"{prefix}_min_cap",
        )
        self.model.addCons(
            z_send + z_recv + z_max == 1,
            name=f"{prefix}_min_select",
        )

    def _normalize_grid_shape(self, grid_shape):
        if grid_shape is None:
            return None
        if len(grid_shape) != 2:
            raise ValueError("grid_shape must be (rows, cols).")
        rows, cols = int(grid_shape[0]), int(grid_shape[1])
        if rows <= 0 or cols <= 0:
            raise ValueError("grid_shape rows/cols must be positive.")
        if rows * cols != self.num_intersections:
            raise ValueError("grid_shape does not match num_intersections.")
        return rows, cols

    def _grid_intersection_layout(self, rows, cols):
        layout = {}
        for row in range(rows):
            for col in range(cols):
                inter = row * cols + col
                layout[inter] = (row, col)
        return layout

    def _grid_connection_pairs(self, rows, cols):
        pairs = []
        for row in range(rows):
            for col in range(cols):
                inter = row * cols + col
                if col + 1 < cols:
                    right = inter + 1
                    pairs.append(((inter, "E"), (right, "W")))
                    pairs.append(((right, "W"), (inter, "E")))
                if row + 1 < rows:
                    down = inter + cols
                    pairs.append(((inter, "S"), (down, "N")))
                    pairs.append(((down, "N"), (inter, "S")))
        return pairs

    def _intersection_tag(self, inter):
        return self.intersection_labels.get(inter, f"I{inter}")

    def _build_link_label(self, tail, head, tail_dir, head_dir):
        if tail is None and head is not None:
            return f"B_in_{self._intersection_tag(head)}_{head_dir}"
        if head is None and tail is not None:
            return f"B_out_{self._intersection_tag(tail)}_{tail_dir}"
        if tail is None and head is None:
            return "B_unknown"
        return f"{self._intersection_tag(tail)}_{tail_dir}_to_{self._intersection_tag(head)}_{head_dir}"

    def _cell_id_label(self, link_id, cell_idx):
        cell_id_map = getattr(self, "_cell_id_map", None)
        if cell_id_map is None:
            cell_id_map = {}
            counter = 1
            for link in self.links:
                for idx in range(link.length):
                    cell_id_map[(link.link_id, idx)] = str(counter)
                    counter += 1
            self._cell_id_map = cell_id_map
        return cell_id_map[(link_id, cell_idx)]

    def _direction_vector(self, dir_name):
        direction_map = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
        if dir_name not in direction_map:
            raise ValueError(f"Unknown direction: {dir_name}")
        return direction_map[dir_name]

    def _intersection_positions(self):
        if self.intersection_layout:
            spacing = self.internal_cell_count + 1
            return {
                inter: (col * spacing, row * spacing)
                for inter, (row, col) in self.intersection_layout.items()
            }

        positions = {}
        spacing = self.internal_cell_count + 1
        if self.intersections:
            positions[self.intersections[0]] = (0, 0)

        for _ in range(len(self.intersections) * 2):
            updated = False
            for (tail_inter, tail_dir), (head_inter, _) in self.connection_pairs:
                if tail_inter in positions and head_inter not in positions:
                    dx, dy = self._direction_vector(tail_dir)
                    base_x, base_y = positions[tail_inter]
                    positions[head_inter] = (base_x + dx * spacing, base_y + dy * spacing)
                    updated = True
                elif head_inter in positions and tail_inter not in positions:
                    dx, dy = self._direction_vector(tail_dir)
                    base_x, base_y = positions[head_inter]
                    positions[tail_inter] = (base_x - dx * spacing, base_y - dy * spacing)
                    updated = True
            if not updated:
                break

        unknown = [inter for inter in self.intersections if inter not in positions]
        for idx, inter in enumerate(unknown):
            positions[inter] = (spacing * (idx + 1), 0)

        return positions

    def _save_initial_density_map(self, initial_rho, save_path=None):
        import os

        if save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            map_path = os.path.join(script_dir, "CTM_OR_network_initial_density_map.png")
        else:
            if save_path.lower().endswith(".png"):
                map_path = save_path
            else:
                map_path = save_path + "_map.png"

        def opposite_direction(dir_name):
            opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}
            if dir_name not in opposite:
                raise ValueError(f"Unknown direction: {dir_name}")
            return opposite[dir_name]

        positions = self._intersection_positions()
        cells = []
        for link in self.links:
            if link.tail is None:
                center = positions[link.head]
                attach_dir = link.head_dir
                travel_dir = opposite_direction(link.head_dir)
            else:
                center = positions[link.tail]
                attach_dir = link.tail_dir
                travel_dir = link.tail_dir
            dx, dy = self._direction_vector(attach_dir)
            start_x = center[0] + dx
            start_y = center[1] + dy
            values = initial_rho[link.link_id]
            for cell_idx in range(link.length):
                cells.append(
                    (
                        start_x + dx * cell_idx,
                        start_y + dy * cell_idx,
                        values[cell_idx],
                        travel_dir,
                    )
                )

        if not cells:
            return

        all_x = [x for x, _, _, _ in cells] + [pos[0] for pos in positions.values()]
        all_y = [y for _, y, _, _ in cells] + [pos[1] for pos in positions.values()]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        pad = 1
        width = max_x - min_x + 1 + 2 * pad
        height = max_y - min_y + 1 + 2 * pad
        scale = 2
        grid = np.full((height * scale, width * scale), np.nan)

        for x, y, val, travel_dir in cells:
            grid_y = (y - min_y + pad) * scale
            grid_x = (x - min_x + pad) * scale
            if travel_dir in ("E", "W"):
                row_offset = 0 if travel_dir == "E" else 1
                grid[grid_y + row_offset, grid_x : grid_x + scale] = val
            else:
                col_offset = 0 if travel_dir == "N" else 1
                grid[grid_y : grid_y + scale, grid_x + col_offset] = val

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        cmap = plt.cm.viridis_r.copy()
        cmap.set_bad(color="white")
        grid_min = float(np.nanmin(grid))
        grid_max = float(np.nanmax(grid))
        if grid_max <= grid_min:
            vmin, vmax = 0.0, self.rho_jam
        else:
            pad_val = 0.05 * (grid_max - grid_min)
            vmin = max(0.0, grid_min - pad_val)
            vmax = min(self.rho_jam, grid_max + pad_val)
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title("Initial density map (network)")
        ax.set_xlabel("Grid")
        ax.set_ylabel("Grid")

        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        for inter, (x, y) in positions.items():
            grid_x = (x - min_x + pad) * scale + (scale - 1) / 2
            grid_y = (y - min_y + pad) * scale + (scale - 1) / 2
            ax.text(
                grid_x,
                grid_y,
                self._intersection_tag(inter),
                ha="center",
                va="center",
                color="black",
                fontsize=5,
            )

        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        plt.tight_layout()
        plt.savefig(map_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_initial_cell_id_map(self, save_path=None):
        import os

        if save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            map_path = os.path.join(script_dir, "CTM_OR_network_initial_cell_id_map.png")
        else:
            if save_path.lower().endswith(".png"):
                map_path = save_path
            else:
                map_path = save_path + "_cell_id_map.png"

        def opposite_direction(dir_name):
            opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}
            if dir_name not in opposite:
                raise ValueError(f"Unknown direction: {dir_name}")
            return opposite[dir_name]

        positions = self._intersection_positions()
        cells = []
        for link in self.links:
            if link.tail is None:
                center = positions[link.head]
                attach_dir = link.head_dir
                travel_dir = opposite_direction(link.head_dir)
            else:
                center = positions[link.tail]
                attach_dir = link.tail_dir
                travel_dir = link.tail_dir
            dx, dy = self._direction_vector(attach_dir)
            start_x = center[0] + dx
            start_y = center[1] + dy
            for cell_idx in range(link.length):
                cell_id = self._cell_id_label(link.link_id, cell_idx)
                cells.append(
                    (
                        start_x + dx * cell_idx,
                        start_y + dy * cell_idx,
                        travel_dir,
                        cell_id,
                    )
                )

        if not cells:
            return

        all_x = [x for x, _, _, _ in cells] + [pos[0] for pos in positions.values()]
        all_y = [y for _, y, _, _ in cells] + [pos[1] for pos in positions.values()]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        pad = 1
        width = max_x - min_x + 1 + 2 * pad
        height = max_y - min_y + 1 + 2 * pad
        scale = 2
        grid = np.full((height * scale, width * scale), np.nan)
        cell_fill = 0.85

        for x, y, travel_dir, _ in cells:
            grid_y = (y - min_y + pad) * scale
            grid_x = (x - min_x + pad) * scale
            if travel_dir in ("E", "W"):
                row_offset = 0 if travel_dir == "E" else 1
                grid[grid_y + row_offset, grid_x : grid_x + scale] = cell_fill
            else:
                col_offset = 0 if travel_dir == "N" else 1
                grid[grid_y : grid_y + scale, grid_x + col_offset] = cell_fill

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        cmap = plt.cm.Blues.copy()
        cmap.set_bad(color="white")
        ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title("Initial cell id map (network)")
        ax.set_xlabel("Grid")
        ax.set_ylabel("Grid")

        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        for x, y, travel_dir, cell_id in cells:
            grid_y = (y - min_y + pad) * scale
            grid_x = (x - min_x + pad) * scale
            if travel_dir in ("E", "W"):
                row_offset = 0 if travel_dir == "E" else 1
                text_x = grid_x + (scale - 1) / 2
                text_y = grid_y + row_offset
            else:
                col_offset = 0 if travel_dir == "N" else 1
                text_x = grid_x + col_offset
                text_y = grid_y + (scale - 1) / 2
            ax.text(
                text_x,
                text_y,
                cell_id,
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )

        for inter, (x, y) in positions.items():
            grid_x = (x - min_x + pad) * scale + (scale - 1) / 2
            grid_y = (y - min_y + pad) * scale + (scale - 1) / 2
            ax.text(
                grid_x,
                grid_y,
                self._intersection_tag(inter),
                ha="center",
                va="center",
                color="black",
                fontsize=5,
            )

        plt.tight_layout()
        plt.savefig(map_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def build_objective(self, objective_type="min_delay"):
        print(f"Building objective: {objective_type}")

        if self.model is None:
            raise ValueError("Call define_decision_variables() before build_objective().")

        if objective_type == "min_delay":
            total_delay = quicksum(
                self.decision_vars["rho"][(link.link_id, cell_idx, t)]
                for link in self.links
                for cell_idx in range(link.length)
                for t in range(self.num_steps + 1)
            )
            total_delay += quicksum(
                self.decision_vars["crossing_queue"][(inter, from_dir, t)]
                for inter in self.intersections
                for from_dir in self.DIRECTIONS
                for t in range(self.num_steps + 1)
            )
            total_delay += quicksum(
                self.decision_vars["boundary_queue"][(link_id, t)]
                for link_id in self.boundary_sources
                for t in range(self.num_steps + 1)
            )
            self.model.setObjective(total_delay, sense="minimize")
            self.objective_sense = "minimize"
        elif objective_type == "max_throughput":
            total_throughput = quicksum(
                self.decision_vars["flow"][(inter, from_dir, to_dir, movement, t)]
                for inter in self.intersections
                for from_dir in self.DIRECTIONS
                for to_dir in self.DIRECTIONS
                for movement in self.MOVEMENTS
                for t in range(self.num_steps)
            )
            self.model.setObjective(total_throughput, sense="maximize")
            self.objective_sense = "maximize"
        elif objective_type == "min_density":
            total_density = quicksum(
                self.decision_vars["rho"][(link.link_id, cell_idx, t)]
                for link in self.links
                for cell_idx in range(link.length)
                for t in range(self.num_steps + 1)
            )
            total_cells = sum(link.length for link in self.links)
            avg_density = total_density / (total_cells * (self.num_steps + 1))
            self.model.setObjective(avg_density, sense="minimize")
            self.objective_sense = "minimize"
        else:
            raise ValueError(f"Unsupported objective type: {objective_type}")

        self.objective_type = objective_type

    def add_ctm_constraints(self):
        print("Adding CTM constraints...")

        constraint_counter = 0
        constraint_counter = self._add_link_constraints(constraint_counter)
        constraint_counter = self._add_crossing_constraints(constraint_counter)
        self._add_signal_constraints(constraint_counter)

    def _add_link_constraints(self, constraint_counter):
        rho = self.decision_vars["rho"]
        flow = self.decision_vars["flow"]
        road_flow = self.decision_vars["road_flow"]
        boundary_inflow = self.decision_vars["boundary_inflow"]
        boundary_queue = self.decision_vars["boundary_queue"]
        crossing_inflow = self.decision_vars["crossing_inflow"]
        crossing_queue = self.decision_vars["crossing_queue"]
        startup_factor = self.decision_vars["startup_factor"]

        for t in range(self.num_steps):
            for link in self.links:
                link_id = link.link_id
                length = link.length

                if link.tail is None:
                    f_in = boundary_inflow[(link_id, t)]
                    q_in = boundary_queue[(link_id, t)]
                    q_next = boundary_queue[(link_id, t + 1)]
                    demand = self._boundary_flow(link.head, link.head_dir)

                    self.model.addCons(
                        f_in <= demand + q_in,
                        name=f"boundary_in_demand_{constraint_counter}",
                    )
                    self.model.addCons(
                        f_in <= self.w * (self.rho_jam - rho[(link_id, 0, t)]),
                        name=f"boundary_in_recv_{constraint_counter}",
                    )
                    self.model.addCons(
                        f_in <= self.Q_max,
                        name=f"boundary_in_cap_{constraint_counter}",
                    )
                    self.model.addCons(
                        q_next == q_in + demand - f_in,
                        name=f"boundary_queue_evolve_{constraint_counter}",
                    )
                    constraint_counter += 1
                    inflow_0 = f_in
                else:
                    tail = link.tail
                    tail_dir = link.tail_dir
                    total_inflow = quicksum(
                        flow[(tail, from_dir, self.turn_map[from_dir][movement], movement, t)]
                        for from_dir in self.DIRECTIONS
                        for movement in self.MOVEMENTS
                        if self.turn_map[from_dir][movement] == tail_dir
                    )
                    self.model.addCons(
                        total_inflow <= self.w * (self.rho_jam - rho[(link_id, 0, t)]),
                        name=f"link_in_recv_{constraint_counter}",
                    )
                    self.model.addCons(
                        total_inflow <= self.Q_max,
                        name=f"link_in_cap_{constraint_counter}",
                    )
                    constraint_counter += 1
                    inflow_0 = total_inflow

                if length > 1:
                    for cell_idx in range(length - 1):
                        outflow = road_flow[(link_id, cell_idx, t)]
                        send_cap = self.v * rho[(link_id, cell_idx, t)]
                        recv_cap = self.w * (self.rho_jam - rho[(link_id, cell_idx + 1, t)])

                        self.model.addCons(
                            outflow <= send_cap,
                            name=f"link_flow_send_{constraint_counter}",
                        )
                        self.model.addCons(
                            outflow <= recv_cap,
                            name=f"link_flow_recv_{constraint_counter}",
                        )
                        self.model.addCons(
                            outflow <= self.Q_max,
                            name=f"link_flow_cap_{constraint_counter}",
                        )
                        constraint_counter += 1

                if link.head is None:
                    outflow_last = self.model.addVar(
                        name=f"link_outflow_{link_id}_{t}_{constraint_counter}",
                        vtype="C",
                        lb=0.0,
                        ub=self.Q_max,
                    )
                    send_cap = self.v * rho[(link_id, length - 1, t)]
                    self.model.addCons(
                        outflow_last <= send_cap,
                        name=f"link_out_send_{constraint_counter}",
                    )
                    self.model.addCons(
                        outflow_last <= self.Q_max,
                        name=f"link_out_cap_{constraint_counter}",
                    )
                    constraint_counter += 1
                else:
                    head = link.head
                    head_dir = link.head_dir
                    outflow_last = crossing_inflow[(head, head_dir, t)]
                    group = "EW" if head_dir in ("E", "W") else "NS"
                    ramp_cap = self.Q_max * startup_factor[(head, group, t)]

                    self.model.addCons(
                        outflow_last <= self.v * rho[(link_id, length - 1, t)],
                        name=f"cross_in_send_{constraint_counter}",
                    )
                    self.model.addCons(
                        outflow_last
                        <= self.w * (self.crossing_rho_jam - crossing_queue[(head, head_dir, t)]),
                        name=f"cross_in_recv_{constraint_counter}",
                    )
                    self.model.addCons(
                        outflow_last <= ramp_cap,
                        name=f"cross_in_cap_{constraint_counter}",
                    )
                    constraint_counter += 1

                for cell_idx in range(length):
                    inflow = inflow_0 if cell_idx == 0 else road_flow[(link_id, cell_idx - 1, t)]
                    outflow = (
                        outflow_last if cell_idx == length - 1 else road_flow[(link_id, cell_idx, t)]
                    )
                    self.model.addCons(
                        rho[(link_id, cell_idx, t + 1)]
                        == rho[(link_id, cell_idx, t)] + inflow - outflow,
                        name=f"link_evolve_{constraint_counter}",
                    )
                    constraint_counter += 1

        return constraint_counter

    def _add_crossing_constraints(self, constraint_counter):
        flow = self.decision_vars["flow"]
        crossing_queue = self.decision_vars["crossing_queue"]
        crossing_inflow = self.decision_vars["crossing_inflow"]

        for inter in self.intersections:
            for from_dir in self.DIRECTIONS:
                self.model.addCons(
                    crossing_queue[(inter, from_dir, 0)] == 0,
                    name=f"crossing_queue_init_{constraint_counter}",
                )
                constraint_counter += 1

        for t in range(self.num_steps):
            for inter in self.intersections:
                for from_dir in self.DIRECTIONS:
                    total_inflow_to_crossing = crossing_inflow[(inter, from_dir, t)]
                    total_outflow_from_crossing = quicksum(
                        flow[(inter, from_dir, self.turn_map[from_dir][movement], movement, t)]
                        for movement in self.MOVEMENTS
                    )

                    self.model.addCons(
                        crossing_queue[(inter, from_dir, t + 1)]
                        == crossing_queue[(inter, from_dir, t)]
                        + total_inflow_to_crossing
                        - total_outflow_from_crossing,
                        name=f"crossing_queue_evolve_{constraint_counter}",
                    )
                    constraint_counter += 1

        return constraint_counter

    def _add_signal_constraints(self, constraint_counter):
        flow = self.decision_vars["flow"]
        current_phase = self.decision_vars["current_phase"]
        phase_switch = self.decision_vars["phase_switch"]
        green_start = self.decision_vars["green_start"]
        startup_step = self.decision_vars["startup_step"]
        startup_factor = self.decision_vars["startup_factor"]
        profile = self._startup_profile()

        for inter in self.intersections:
            self.model.addCons(
                current_phase[(inter, 0)] == 0,
                name=f"initial_phase_{inter}",
            )
            self.model.addCons(
                phase_switch[(inter, 0)] == 0,
                name=f"initial_phase_switch_{inter}",
            )
            constraint_counter += 2

            for t in range(1, self.num_steps + 1):
                self.model.addCons(
                    current_phase[(inter, t)] - current_phase[(inter, t - 1)]
                    <= phase_switch[(inter, t)],
                    name=f"phase_evolve_ub_{constraint_counter}",
                )
                self.model.addCons(
                    current_phase[(inter, t - 1)] - current_phase[(inter, t)]
                    <= phase_switch[(inter, t)],
                    name=f"phase_evolve_lb_{constraint_counter}",
                )
                constraint_counter += 2

                self.model.addCons(
                    current_phase[(inter, t)] + current_phase[(inter, t - 1)]
                    <= 1 + (1 - phase_switch[(inter, t)]),
                    name=f"phase_flip_ub_{constraint_counter}",
                )
                self.model.addCons(
                    current_phase[(inter, t)] + current_phase[(inter, t - 1)]
                    >= 1 - (1 - phase_switch[(inter, t)]),
                    name=f"phase_flip_lb_{constraint_counter}",
                )
                constraint_counter += 2

                if t >= 2:
                    self.model.addCons(
                        phase_switch[(inter, t)]
                        + phase_switch[(inter, t - 1)]
                        + phase_switch[(inter, t - 2)]
                        <= 1,
                        name=f"phase_switch_freq_{constraint_counter}",
                    )
                    constraint_counter += 1

            for t in range(self.num_steps + 1):
                for group in self.SIGNAL_GROUPS:
                    if group == "EW":
                        green_now = 1 - current_phase[(inter, t)]
                        green_prev = 1 - current_phase[(inter, t - 1)] if t > 0 else None
                    else:
                        green_now = current_phase[(inter, t)]
                        green_prev = current_phase[(inter, t - 1)] if t > 0 else None

                    start_var = green_start[(inter, group, t)]
                    if t == 0:
                        self.model.addCons(
                            start_var == green_now,
                            name=f"green_start_init_{inter}_{group}_{constraint_counter}",
                        )
                        constraint_counter += 1
                    else:
                        self.model.addCons(
                            start_var >= green_now - green_prev,
                            name=f"green_start_lb_{inter}_{group}_{constraint_counter}",
                        )
                        self.model.addCons(
                            start_var <= green_now,
                            name=f"green_start_ub_{inter}_{group}_{constraint_counter}",
                        )
                        self.model.addCons(
                            start_var <= 1 - green_prev,
                            name=f"green_start_prev_{inter}_{group}_{constraint_counter}",
                        )
                        constraint_counter += 3

                    step1 = startup_step[(inter, group, t, 1)]
                    step2 = startup_step[(inter, group, t, 2)]
                    step3 = startup_step[(inter, group, t, 3)]
                    step4 = startup_step[(inter, group, t, 4)]

                    self.model.addCons(
                        step1 == start_var,
                        name=f"startup_step1_{inter}_{group}_{constraint_counter}",
                    )
                    constraint_counter += 1

                    if t >= 1:
                        prev_start = green_start[(inter, group, t - 1)]
                        self.model.addCons(
                            step2 <= prev_start,
                            name=f"startup_step2_ub1_{inter}_{group}_{constraint_counter}",
                        )
                        self.model.addCons(
                            step2 <= green_now,
                            name=f"startup_step2_ub2_{inter}_{group}_{constraint_counter}",
                        )
                        self.model.addCons(
                            step2 >= prev_start + green_now - 1,
                            name=f"startup_step2_lb_{inter}_{group}_{constraint_counter}",
                        )
                        constraint_counter += 3
                    else:
                        self.model.addCons(
                            step2 == 0,
                            name=f"startup_step2_zero_{inter}_{group}_{constraint_counter}",
                        )
                        constraint_counter += 1

                    if t >= 2:
                        prev2_start = green_start[(inter, group, t - 2)]
                        self.model.addCons(
                            step3 <= prev2_start,
                            name=f"startup_step3_ub1_{inter}_{group}_{constraint_counter}",
                        )
                        self.model.addCons(
                            step3 <= green_now,
                            name=f"startup_step3_ub2_{inter}_{group}_{constraint_counter}",
                        )
                        self.model.addCons(
                            step3 >= prev2_start + green_now - 1,
                            name=f"startup_step3_lb_{inter}_{group}_{constraint_counter}",
                        )
                        constraint_counter += 3
                    else:
                        self.model.addCons(
                            step3 == 0,
                            name=f"startup_step3_zero_{inter}_{group}_{constraint_counter}",
                        )
                        constraint_counter += 1

                    self.model.addCons(
                        step1 + step2 + step3 + step4 == green_now,
                        name=f"startup_step_sum_{inter}_{group}_{constraint_counter}",
                    )
                    constraint_counter += 1

                    self.model.addCons(
                        startup_factor[(inter, group, t)]
                        == profile[0] * step1
                        + profile[1] * step2
                        + profile[2] * step3
                        + profile[3] * step4,
                        name=f"startup_factor_{inter}_{group}_{constraint_counter}",
                    )
                    constraint_counter += 1

            for t in range(self.num_steps):
                for from_dir in self.DIRECTIONS:
                    cell_rho = self.decision_vars["crossing_queue"][(inter, from_dir, t)]
                    if from_dir in ["E", "W"]:
                        allow = 1 - current_phase[(inter, t)]
                    else:
                        allow = current_phase[(inter, t)]

                    signal_controlled_flow = quicksum(
                        flow[(inter, from_dir, self.turn_map[from_dir][movement], movement, t)]
                        for movement in ["S", "L"]
                    )
                    right_turn_flow = flow[
                        (inter, from_dir, self.turn_map[from_dir]["R"], "R", t)
                    ]
                    total_outflow = quicksum(
                        flow[(inter, from_dir, self.turn_map[from_dir][movement], movement, t)]
                        for movement in self.MOVEMENTS
                    )

                    self.model.addCons(
                        total_outflow <= self.v_crossing * cell_rho,
                        name=f"sending_capacity_v_{constraint_counter}",
                    )
                    self.model.addCons(
                        total_outflow <= self.Q_max_crossing,
                        name=f"sending_capacity_q_{constraint_counter}",
                    )
                    constraint_counter += 2

                    self.model.addCons(
                        signal_controlled_flow <= self.Q_max_crossing * allow,
                        name=f"signal_controlled_{constraint_counter}",
                    )
                    constraint_counter += 1

                    self.model.addCons(
                        right_turn_flow <= self.Q_max_crossing * 0.5,
                        name=f"right_turn_cap_{constraint_counter}",
                    )
                    constraint_counter += 1

                    for movement in self.MOVEMENTS:
                        to_dir = self.turn_map[from_dir][movement]

                        for other_dir in self.DIRECTIONS:
                            if other_dir != to_dir:
                                self.model.addCons(
                                    flow[(inter, from_dir, other_dir, movement, t)] == 0,
                                    name=f"invalid_movement_{constraint_counter}",
                                )
                                constraint_counter += 1

                        if movement in ["S", "L"]:
                            self.model.addCons(
                                flow[(inter, from_dir, to_dir, movement, t)]
                                <= self.Q_max_crossing * allow,
                                name=f"movement_allowed_{constraint_counter}",
                            )
                            constraint_counter += 1

        return constraint_counter

    def add_initial_conditions(self, initial_rho=None):
        print("Adding initial conditions...")
        rho = self.decision_vars["rho"]
        boundary_queue = self.decision_vars["boundary_queue"]

        for link_id in self.boundary_sources:
            self.model.addCons(
                boundary_queue[(link_id, 0)] == 0,
                name=f"init_boundary_queue_{link_id}",
            )

        if initial_rho is None:
            def build_initial_sequence(length):
                if length <= 0:
                    return np.zeros(0)
                vals = np.zeros(length, dtype=float)
                vals[0] = np.random.uniform(0.2, 0.3)
                for idx in range(1, length):
                    prev = vals[idx - 1]
                    low = max(0.0, prev * 0.9)
                    high = min(self.rho_jam, prev * 1.1)
                    if low > high:
                        low, high = high, low
                    vals[idx] = np.random.uniform(low, high)
                return vals

            init_rho = {}
            for link in self.links:
                init_vals = build_initial_sequence(link.length)
                init_rho[link.link_id] = init_vals
                for cell_idx in range(link.length):
                    self.model.addCons(
                        rho[(link.link_id, cell_idx, 0)] == float(init_vals[cell_idx]),
                        name=f"init_rho_{link.link_id}_{cell_idx}",
                    )
            self.initial_rho = init_rho
        else:
            for link in self.links:
                if link.link_id not in initial_rho:
                    raise ValueError("initial_rho missing link {}".format(link.link_id))
                init_vals = initial_rho[link.link_id]
                if len(init_vals) != link.length:
                    raise ValueError("initial_rho length mismatch for link {}".format(link.link_id))
                for cell_idx in range(link.length):
                    self.model.addCons(
                        rho[(link.link_id, cell_idx, 0)] == float(init_vals[cell_idx]),
                        name=f"init_rho_{link.link_id}_{cell_idx}",
                    )
            self.initial_rho = dict(initial_rho)

        if self.initial_rho is not None:
            self._save_initial_density_map(self.initial_rho)
            self._save_initial_cell_id_map()

    def solve(
        self,
        gap_rel=None,
        improve_rel=None,
        improve_chunk_seconds=30,
        improve_max_rounds=10,
    ):
        import time

        print("Solving optimization problem...")
        if self.model is None:
            raise ValueError("Model is not initialized. Call build_objective() first.")

        if gap_rel is not None:
            self.model.setParam("limits/gap", gap_rel)

        if improve_chunk_seconds is not None and improve_max_rounds is not None:
            total_time = improve_chunk_seconds * improve_max_rounds
            if total_time > 0:
                self.model.setParam("limits/time", total_time)

        event_type = None
        if improve_rel is not None:
            event_type = getattr(SCIP_EVENTTYPE, "BESTSOLFOUND", None)
            if event_type is None:
                event_type = getattr(SCIP_EVENTTYPE, "SOLFOUND", None)
            if event_type is None:
                print("Warning: SCIP solution event not available; improve_rel ignored.")
            else:
                stopper = ImprovementStopper(improve_rel, self.objective_sense, event_type)
                self.model.includeEventhdlr(
                    stopper, "improve_stop", "stop if improvement is small"
                )
                self._improve_stopper = stopper

        start_time = time.time()
        self.model.optimize()
        solve_time = time.time() - start_time

        status = self.model.getStatus()
        if self.model.getNSols() == 0:
            self.optimization_results = {
                "status": str(status),
                "objective_value": None,
                "solve_time": solve_time,
                "phase_change_count": 0,
            }
            print(f"Solve finished with status: {status}")
            return self.optimization_results

        best_sol = self.model.getBestSol()
        if best_sol is None:
            self.optimization_results = {
                "status": str(status),
                "objective_value": None,
                "solve_time": solve_time,
                "phase_change_count": 0,
            }
            print(f"Solve finished with status: {status}")
            return self.optimization_results

        try:
            objective_value = self.model.getSolObjVal(best_sol)
        except Exception:
            try:
                objective_value = self.model.getObjVal()
            except Exception:
                objective_value = None

        self.optimization_results = {
            "status": str(status),
            "objective_value": objective_value,
            "solve_time": solve_time,
            "phase_change_count": 0,
        }

        self.rho_optimized = {}
        rho = self.decision_vars["rho"]
        try:
            for link in self.links:
                link_vals = np.zeros((link.length, self.num_steps + 1))
                for cell_idx in range(link.length):
                    for t in range(self.num_steps + 1):
                        link_vals[cell_idx, t] = self.model.getSolVal(
                            best_sol, rho[(link.link_id, cell_idx, t)]
                        )
                self.rho_optimized[link.link_id] = link_vals

            self.flow_optimized = {}
            flow_vars = self.decision_vars["flow"]
            for inter in self.intersections:
                for from_dir in self.DIRECTIONS:
                    for to_dir in self.DIRECTIONS:
                        for movement in self.MOVEMENTS:
                            for t in range(self.num_steps):
                                key = (inter, from_dir, to_dir, movement, t)
                                self.flow_optimized[key] = self.model.getSolVal(
                                    best_sol, flow_vars[key]
                                )

            phase_change_count = 0
            self.signal_phase = {}
            phase_switch = self.decision_vars["phase_switch"]
            for inter in self.intersections:
                for t in range(1, self.num_steps + 1):
                    if self.model.getSolVal(best_sol, phase_switch[(inter, t)]) > 0.5:
                        phase_change_count += 1
                self.signal_phase[inter] = [
                    self.model.getSolVal(
                        best_sol, self.decision_vars["current_phase"][(inter, t)]
                    )
                    for t in range(self.num_steps + 1)
                ]
        except Exception as exc:
            print(f"Warning: failed to extract solution values ({exc})")
            self.rho_optimized = None
            self.flow_optimized = None
            return self.optimization_results

        self.optimization_results["phase_change_count"] = phase_change_count

        print(f"Solve finished with status: {status}")
        if self.optimization_results["objective_value"] is not None:
            print(f"Objective value: {self.optimization_results['objective_value']:.4f}")
        else:
            print("Objective value: N/A")
        print(f"Solve time: {self.optimization_results['solve_time']:.2f} s")
        print(f"Phase changes: {self.optimization_results['phase_change_count']}")

        return self.optimization_results

    def print_optimization_summary(self):
        if self.optimization_results is None:
            print("No optimized results. Run solve() first.")
            return

        print("=== Optimization Summary ===")
        print(f"Objective: {self.objective_type}")
        print(f"Status: {self.optimization_results['status']}")
        obj_val = self.optimization_results.get("objective_value")
        if obj_val is None:
            print("Objective value: N/A")
        else:
            print(f"Objective value: {obj_val:.4f}")
        print(f"Solve time: {self.optimization_results['solve_time']:.2f} s")
        print(f"Phase changes: {self.optimization_results['phase_change_count']}")

        if self.rho_optimized is None:
            print("Average density: N/A")
        else:
            all_vals = np.concatenate([vals.flatten() for vals in self.rho_optimized.values()])
            avg_density = float(np.mean(all_vals))
            print(f"Average density: {avg_density:.4f}")

        if not self.flow_optimized:
            print("Total throughput: N/A")
        else:
            total_throughput = sum(self.flow_optimized.values())
            print(f"Total throughput: {total_throughput:.4f}")

    def print_initial_densities_for_direction(self, intersection, direction):
        if self.initial_rho is None:
            print("No initial densities available. Run add_initial_conditions() first.")
            return

        key = (intersection, direction)
        if key not in self.incoming_link or key not in self.outgoing_link:
            inter_tag = self._intersection_tag(intersection)
            print(f"No links found for {inter_tag} {direction}.")
            return

        inter_tag = self._intersection_tag(intersection)
        link_ids = [
            ("incoming", self.incoming_link[key]),
            ("outgoing", self.outgoing_link[key]),
        ]
        for label, link_id in link_ids:
            vals = self.initial_rho.get(link_id)
            link = self.links[link_id]
            if vals is None:
                print(
                    f"{inter_tag} {direction} {label} link L{link_id} ({link.label}): N/A"
                )
                continue
            formatted = ", ".join(f"{val:.4f}" for val in vals)
            print(
                f"{inter_tag} {direction} {label} link L{link_id} ({link.label}): {formatted}"
            )

    def plot_signal_phases(self, save_path=None):
        if not self.signal_phase:
            print("No signal phase data available for plotting.")
            return

        num_plots = len(self.intersections)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3.2 * num_plots))
        if num_plots == 1:
            axes = [axes]

        for idx, inter in enumerate(self.intersections):
            phases = self.signal_phase.get(inter)
            if not phases:
                continue
            time_axis = np.arange(len(phases))
            ew_signal = [1 - p for p in phases]
            ns_signal = [p for p in phases]
            ax = axes[idx]
            inter_tag = self._intersection_tag(inter)
            ax.step(time_axis, ew_signal, "b-", linewidth=2, where="post", label="EW green")
            ax.step(time_axis, ns_signal, "r-", linewidth=2, where="post", label="NS green")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Signal state")
            ax.set_title(f"Intersection {inter_tag} signal phases", fontweight="bold")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Red", "Green"])
            ax.set_xlim(0, len(time_axis) - 1)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_signal_phases.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def plot_all_cell_densities(self, save_path=None, max_plots=2):
        if self.rho_optimized is None:
            print("No optimized results. Run solve() first.")
            return

        rows = []
        for link_id in sorted(self.rho_optimized):
            link_vals = self.rho_optimized[link_id]
            for cell_idx in range(link_vals.shape[0]):
                rows.append(link_vals[cell_idx, :])

        if not rows:
            print("No density data available for plotting.")
            return

        data = np.vstack(rows)
        total_rows = data.shape[0]
        row_labels = [str(idx) for idx in range(1, total_rows + 1)]
        plot_count = max(1, min(max_plots, total_rows))
        chunk_size = int(np.ceil(total_rows / plot_count))

        for idx in range(plot_count):
            start = idx * chunk_size
            end = min(total_rows, (idx + 1) * chunk_size)
            if start >= end:
                break
            cell_start = start + 1
            cell_end = end
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            im = ax.imshow(
                data[start:end, :],
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=0.0,
                vmax=self.rho_jam,
            )
            ax.set_xlabel("Time step")
            ax.set_ylabel("Cell id")
            ax.set_title(f"Density evolution (cells {cell_start}-{cell_end})")
            ax.set_yticks(np.arange(end - start))
            ax.set_yticklabels(
                [str(cell_id) for cell_id in range(cell_start, cell_end + 1)],
                fontsize=6,
            )
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
            plt.tight_layout()
            if save_path:
                plt.savefig(
                    f"{save_path}_cell_density_{idx + 1}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            else:
                plt.show()

def main():
    print("=== CTM OR optimization demo (four intersections, 2x2 grid) ===")

    ctm_or = CTM_OR_Network_Model(
        num_intersections=4,
        grid_shape=(2, 2),
        internal_cell_count=5,
        external_cell_count=3,
        v=1.0,
        w=0.5,
        rho_jam=1.0,
        num_steps=100,
    )

    ctm_or.define_decision_variables()
    ctm_or.build_objective(objective_type="min_delay")
    ctm_or.add_ctm_constraints()
    ctm_or.add_initial_conditions()
    ctm_or.print_initial_densities_for_direction(0, "W")

    improve_rel = 0.00001
    improve_chunk_seconds = 300
    improve_max_rounds = 10
    ctm_or.solve(
        improve_rel=improve_rel,
        improve_chunk_seconds=improve_chunk_seconds,
        improve_max_rounds=improve_max_rounds,
    )

    ctm_or.print_optimization_summary()

    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "CTM_OR_results")
    ctm_or.plot_signal_phases(save_path=save_path)
    ctm_or.plot_all_cell_densities(save_path=save_path, max_plots=2)

    print("CTM OR optimization demo finished.")


if __name__ == "__main__":
    main()
