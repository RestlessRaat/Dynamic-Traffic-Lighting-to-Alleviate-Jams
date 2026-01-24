# -*- coding: utf-8 -*-
"""
CTM OR model implemented with PySCIPOpt.
"""

from __future__ import annotations

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

    def __init__(
        self,
        num_cells=40,
        v=1.0,
        w=0.5,
        rho_jam=1.0,
        num_steps=360,
        crossing_capacity_factor=2.0,
    ):
        self.num_cells = num_cells
        self.v = v
        self.w = w
        self.rho_jam = rho_jam
        self.num_steps = num_steps

        self.crossing_capacity_factor = crossing_capacity_factor
        self.v_crossing = v / 2.0
        self.Q_max = v * w * rho_jam / (v + w)
        self.Q_max_crossing = self.Q_max * self.crossing_capacity_factor
        self.crossing_rho_jam = self.rho_jam * self.crossing_capacity_factor

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
            "Params: v={:.3f}, w={:.3f}, rho_jam={:.3f}, Q_max={:.4f}".format(
                v, w, rho_jam, self.Q_max
            )
        )
        print(
            "Crossing: v_crossing={:.3f}, capacity_factor={:.2f}, Q_max_crossing={:.4f}".format(
                self.v_crossing, self.crossing_capacity_factor, self.Q_max_crossing
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

        return {
            "crossing_queue": crossing_queue,
            "current_phase": current_phase,
            "phase_switch": phase_switch,
        }

    def sending_function(self, rho_i):
        return min(self.v * rho_i, self.Q_max)

    def receiving_function(self, rho_i):
        return min(self.w * (self.rho_jam - rho_i), self.Q_max)

    def _force_flow_min(self, flow_var, send_cap, recv_cap, max_cap, name_prefix):
        big_m = max(self.v, self.w) * self.rho_jam + max_cap
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
                self._force_flow_min(
                    flow_0_1,
                    self.v * rho[(dir_idx, 0, 0, t)],
                    self.w * (self.rho_jam - rho[(dir_idx, 0, 1, t)]),
                    self.Q_max,
                    f"up_{dir_idx}_0_1_{t}",
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
                    self._force_flow_min(
                        outflow,
                        self.v * rho[(dir_idx, 0, cell_idx, t)],
                        self.w * (self.rho_jam - rho[(dir_idx, 0, cell_idx + 1, t)]),
                        self.Q_max,
                        f"up_{dir_idx}_{cell_idx}_{cell_idx + 1}_{t}",
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
                    cross_in <= self.Q_max_crossing,
                    name=f"cross_in_cap_{constraint_counter}",
                )
                self._force_flow_min(
                    cross_in,
                    self.v * rho[(dir_idx, 0, last_up_cell_idx, t)],
                    self.w * (self.crossing_rho_jam - crossing_queue[(from_dir, t)]),
                    self.Q_max_crossing,
                    f"cross_in_{from_dir}_{t}",
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
                self._force_flow_min(
                    flow_0_1,
                    self.v * rho[(dir_idx, 1, 0, t)],
                    self.w * (self.rho_jam - rho[(dir_idx, 1, 1, t)]),
                    self.Q_max,
                    f"down_{dir_idx}_0_1_{t}",
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
                    self._force_flow_min(
                        outflow,
                        self.v * rho[(dir_idx, 1, cell_idx, t)],
                        self.w * (self.rho_jam - rho[(dir_idx, 1, cell_idx + 1, t)]),
                        self.Q_max,
                        f"down_{dir_idx}_{cell_idx}_{cell_idx + 1}_{t}",
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
                self._force_flow_min(
                    outflow,
                    self.v * rho[(dir_idx, 1, last_cell_idx, t)],
                    self.Q_max,
                    self.Q_max,
                    f"down_{dir_idx}_{last_cell_idx}_out_{t}",
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
            init_rho = np.zeros((len(self.DIRECTIONS), 2, self.num_cells))
            for dir_idx in range(len(self.DIRECTIONS)):
                upstream_initial = np.random.uniform(0.1, 0.3, self.num_cells)
                downstream_initial = np.random.uniform(0.1, 0.3, self.num_cells)
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

        if self.initial_rho is not None:
            self._save_initial_density_map(self.initial_rho)

    def _save_initial_density_map(self, initial_rho, save_path=None):
        import os

        if save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            map_path = os.path.join(script_dir, "CTM_OR_initial_density_map.png")
        else:
            if save_path.lower().endswith(".png"):
                map_path = save_path
            else:
                map_path = save_path + "_map.png"

        size = 2 * self.num_cells + 1
        center = self.num_cells
        grid = np.full((size, size), np.nan)

        for dir_idx, dir_name in enumerate(self.DIRECTIONS):
            for cell_idx in range(self.num_cells):
                up_val = initial_rho[dir_idx, 0, cell_idx]
                down_val = initial_rho[dir_idx, 1, cell_idx]

                if dir_name == "N":
                    grid[center - (cell_idx + 1), center] = up_val
                    grid[center + (cell_idx + 1), center] = down_val
                elif dir_name == "S":
                    grid[center + (cell_idx + 1), center] = up_val
                    grid[center - (cell_idx + 1), center] = down_val
                elif dir_name == "E":
                    grid[center, center + (cell_idx + 1)] = up_val
                    grid[center, center - (cell_idx + 1)] = down_val
                else:  # W
                    grid[center, center - (cell_idx + 1)] = up_val
                    grid[center, center + (cell_idx + 1)] = down_val

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")
        grid_min = float(np.nanmin(grid))
        grid_max = float(np.nanmax(grid))
        if grid_max <= grid_min:
            vmin, vmax = 0.0, self.rho_jam
        else:
            pad = 0.05 * (grid_max - grid_min)
            vmin = max(0.0, grid_min - pad)
            vmax = min(self.rho_jam, grid_max + pad)
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title("Initial density map")
        ax.set_xlabel("Grid")
        ax.set_ylabel("Grid")

        ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        ax.text(center, 0, "N", ha="center", va="bottom")
        ax.text(center, size - 1, "S", ha="center", va="top")
        ax.text(size - 1, center, "E", ha="right", va="center")
        ax.text(0, center, "W", ha="left", va="center")

        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        plt.tight_layout()
        plt.savefig(map_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

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

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        if not self.signal_phase:
            print("No signal phase data available for plotting.")
            plt.close(fig)
            return

        time_axis = np.arange(len(self.signal_phase))
        ew_signal = [1 - p for p in self.signal_phase]
        ns_signal = [p for p in self.signal_phase]

        ax.step(time_axis, ew_signal, "b-", linewidth=3, where="post", label="EW green")
        ax.step(time_axis, ns_signal, "r-", linewidth=3, where="post", label="NS green")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Signal state")
        ax.set_title(
            "Signal timing (dynamic phases)",
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Red", "Green"])
        ax.set_xlim(0, len(time_axis) - 1)
        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_signal_timing.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

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

        if self.signal_phase:
            print("Signal states by time step:")
            for t, phase in enumerate(self.signal_phase):
                state = "EW green" if phase < 0.5 else "NS green"
                print(f"  t={t}: {state}")


def main():
    print("=== CTM OR optimization demo (PySCIPOpt) ===")

    ctm_or = CTM_OR_Model(
        num_cells=10,
        v=1.0,
        w=0.5,
        rho_jam=1.0,
        num_steps=50,
    )

    ctm_or.define_decision_variables()
    ctm_or.build_objective(objective_type="min_delay")
    ctm_or.add_ctm_constraints()
    ctm_or.add_initial_conditions()

    improve_rel = 0.01
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
    ctm_or.plot_optimization_results(save_path=save_path)

    north_density_save_path = os.path.join(script_dir, "CTM_OR_north")
    ctm_or.plot_north_upstream_density(save_path=north_density_save_path)

    print("CTM OR optimization demo finished.")


if __name__ == "__main__":
    main()
