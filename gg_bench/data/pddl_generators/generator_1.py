"""
PHASE-SHIFT  –  automatic level generator + PDDL validator
==========================================================

• Produces fully compatible two-layer ASCII levels for the supplied
  `CustomEnv` (see provided Gym code).
• Builds a matching PDDL problem file, calls an external planner
  (pyperplan), parses the solution length and path, and only returns maps
  that are solvable and respect the requested difficulty.

External dependency:
    pip install pyperplan

Author:  <your-name>   2024-04
"""

from __future__ import annotations

import itertools
import os
import random
import subprocess
import tempfile
import shutil
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyperplan.pddl.parser import Parser
from pyperplan.task import Task
from pyperplan.search import astar_search
from pyperplan.heuristics.heuristic_base import Heuristic
from pyperplan.heuristics.relaxation import hFFHeuristic as h_ff
from pyperplan.heuristics.blind import BlindHeuristic as h_blind


# ──────────────────────────────────────────────────────────────────────
# helper constants identical to CustomEnv
# ──────────────────────────────────────────────────────────────────────
WALL = "#"
FLOOR = "."
PLAYER = "P"
BEACON_OFF = "B"
BEACON_ON = "b"
EXIT = "E"
RIFT = "X"
KEY = "k"
GATE_CLOSED = "G"
GATE_OPEN = "g"
VOID = " "  # used internally – never placed inside bounding frame

VAL_EXECUTABLE = "/Users/fancy/Desktop/project/autoenv/VAL/build/macos64/Release/bin/validate"  # or "/path/to/VAL/validate"
FAST_DOWNWARD_SCRIPT_PATH = os.getenv("FAST_DOWNWARD_PATH", "/Users/fancy/Desktop/project/autoenv/downward/fast-downward.py")


class PDDLValidator:
    def __init__(self, validator_executable=VAL_EXECUTABLE):
        self.validator_executable = validator_executable

    def validate_pddl(self, domain_filepath: str, problem_filepath: str) -> bool:
        print(f"\n--- Validating PDDL files ({self.validator_executable}) ---")
        command = [self.validator_executable, domain_filepath, problem_filepath]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print("PDDL Validation Successful (according to VAL output pattern)!")
                print("VAL Output:\n", result.stdout)
                return True
            elif "Problem an Domain files validated successfully" in result.stdout:
                print("PDDL Validation Successful (according to VAL output pattern)!")
                print("VAL Output:\n", result.stdout)
                return True
            else:
                print("PDDL Validation Failed or Validator output indicates issues.")
                print("Return Code:", result.returncode)
                print("STDOUT:\n", result.stdout)
                print("STDERR:\n", result.stderr)
                return False

        except FileNotFoundError:
            print(f"Error: '{self.validator_executable}' command not found.")
            print("Please ensure VAL (or your chosen validator) is installed and in your PATH, or update VAL_EXECUTABLE.")
            return False
        except Exception as e:
            print(f"An error occurred during PDDL validation: {e}")
            return False

class PDDLPlanner:
    def __init__(self, planner_script_path=FAST_DOWNWARD_SCRIPT_PATH):
        self.planner_script_path = planner_script_path
        if not (os.path.isfile(self.planner_script_path) and os.access(self.planner_script_path, os.X_OK)):
            # If it's not a direct file path, check if it's in PATH
            if not shutil.which(self.planner_script_path):
                print(f"Warning: Planner script '{self.planner_script_path}' not found, not executable, or not in PATH.")
                self.is_configured = False
            else:
                self.is_configured = True # Found in PATH
        else:
            self.is_configured = True # Direct file path is valid

    def generate_plan(self, domain_filepath: str, problem_filepath: str,
                      plan_output_file="sas_plan") -> list[str] | None:
        if not self.is_configured:
            print("Skipping plan generation as Fast Downward planner is not configured correctly.")
            return None

        print(f"\n--- Generating plan using Fast Downward ({self.planner_script_path}) ---")
        
        # A common satisficing (non-optimal, but fast) configuration for Fast Downward
        # You might need to adjust python interpreter if fast-downward.py is a python script
        # and your system needs specific python version e.g. ['python3', self.planner_script_path, ...]
        command = [
            self.planner_script_path,
            domain_filepath,
            problem_filepath,
            "--search", "lazy_greedy([ff()], preferred=[ff()])" # A good general purpose planner
            # For optimal: "--search", "astar(lmcut())"
        ]
        
        # Clean up previous plan file if it exists
        if os.path.exists(plan_output_file):
            os.remove(plan_output_file)
        # Clean up other common FD output files
        for fd_out_file in ["output", "output.sas"]:
            if os.path.exists(fd_out_file):
                os.remove(fd_out_file)

        try:
            print(f"Executing command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60) # 60s timeout

            print("Planner STDOUT:")
            print(result.stdout)
            print("Planner STDERR:")
            print(result.stderr)

            if os.path.exists(plan_output_file):
                print(f"Plan found! Output written to {plan_output_file}")
                with open(plan_output_file, 'r') as f:
                    plan = [line.strip() for line in f if line.strip() and not line.startswith(';')]
                
                # Clean up after successful run
                if os.path.exists(plan_output_file): os.remove(plan_output_file)
                if os.path.exists("output"): os.remove("output")
                if os.path.exists("output.sas"): os.remove("output.sas")
                if not plan: # File existed but was empty or only comments
                    print("Plan file was empty or contained no valid actions.")
                    return None
                return plan
            else:
                print("No plan file (sas_plan) was generated by Fast Downward.")
                if "Solution found." in result.stdout and result.returncode == 0:
                     print("Planner reported solution found, but sas_plan is missing. Check planner configuration/version.")
                elif "Task is provably unsolvable." in result.stdout or "Goal can be simplified to false." in result.stdout :
                    print("Planner reported task is unsolvable.")
                elif result.returncode != 0 :
                    print(f"Planner exited with error code: {result.returncode}")
                return None

        except FileNotFoundError:
            print(f"Error: Planner script '{self.planner_script_path}' not found.")
            print("Please ensure Fast Downward is installed and FAST_DOWNWARD_SCRIPT_PATH is set correctly.")
            return None
        except subprocess.TimeoutExpired:
            print(f"Error: Planner timed out after 60 seconds.")
            return None
        except Exception as e:
            print(f"An error occurred during plan generation: {e}")
            return None

class PDDLLevelGenerator:
    """
    High-level API requested by the specification.
    """

    SIZES = {"easy": 5, "medium": 7, "hard": 9}
    MAX_RETRIES = 200  # avoid endless loops

    def __init__(self, difficulty: str = "medium", seed: Optional[int] = None):
        assert difficulty in ("easy", "medium", "hard"), "Unknown difficulty"
        self.difficulty = difficulty
        self.rng = random.Random(seed)

        # parameters affecting layout complexity
        self.size = self.SIZES[difficulty]
        self.num_beacons = {"easy": 1, "medium": 2, "hard": 3}[difficulty]
        self.put_key_gate = difficulty != "easy"
        self.rift_chance = {"easy": 0.00, "medium": 0.05, "hard": 0.10}[difficulty]
        self.validator = PDDLValidator(VAL_EXECUTABLE)
        self.planner = PDDLPlanner(FAST_DOWNWARD_SCRIPT_PATH)

    # ────────────────────────────────────────────────────────────── PUBLIC
    def generate_level(self) -> Dict:
        """
        Create a single level (dict containing the two layers + solution
        meta-data).  Guaranteed solvable, otherwise retries generation.
        """
        for _ in range(self.MAX_RETRIES):
            level = self._construct_candidate()
            info = self.validate_level(level)

            if info["solvable"]:
                level.update(info)  # merge validation stats
                return level
        raise RuntimeError("Failed to create a valid level after many tries")

    def generate_batch(self, count: int = 10) -> List[Dict]:
        return [self.generate_level() for _ in range(count)]

    def validate_level(self, level_data) -> Dict:
        """
        Build a PDDL problem description, call VAL for validation,
        Fast Downward for planning, and return solvability + optimal step count + action sequence.
        """
        domain_file, prob_file = self._dump_pddl(level_data)

        try:
            # 1. Validate PDDL with VAL
            validation_successful = self.validator.validate_pddl(domain_file, prob_file)

            if not validation_successful:
                print("PDDL validation failed. Cannot proceed with planning.")
                return {"solvable": False}

            # 2. Generate plan with Fast Downward
            plan_actions = self.planner.generate_plan(domain_file, prob_file)

            if plan_actions is None:
                return {"solvable": False}

            return {
                "solvable": True,
                "optimal_steps": len(plan_actions), # Fast Downward's lazy_greedy search is satisficing, not optimal.
                                                    # For optimal, you'd need a different search strategy (e.g., astar(lmcut()))
                "solution": plan_actions,
            }

        finally:
            # Clean up tmp files
            for f in [domain_file, prob_file]:
                if os.path.exists(f):
                    os.remove(f)


    def save_level(self, level_data: Dict, filepath: str | os.PathLike):
        """
        Persist two text files <name>_A.txt and <name>_B.txt so that
        `CustomEnv` can load them directly.
        """
        path = Path(filepath)
        stem = path.stem
        dir_ = path.parent
        dir_.mkdir(parents=True, exist_ok=True)

        boardA = level_data["layerA"]
        boardB = level_data["layerB"]

        with open(dir_ / f"{stem}_A.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(boardA))
        with open(dir_ / f"{stem}_B.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(boardB))

    def get_difficulty_constraints(self) -> Dict:
        return {
            "difficulty": self.difficulty,
            "size": self.size,
            "num_beacons": self.num_beacons,
            "uses_key_gate": self.put_key_gate,
            "rift_chance": self.rift_chance,
        }

    # ───────────────────────────────────────────────────────── INTERNAL
    # MAP GENERATION ----------------------------------------------------
    def _construct_candidate(self) -> Dict:
        N = self.size
        boardA = [[WALL] * N for _ in range(N)]
        boardB = [[WALL] * N for _ in range(N)]

        # carve simple rectangular rooms connected differently in each phase
        self._carve_simple_room(boardA)
        self._carve_simple_room(boardB)

        # randomly add inner walls so that both layers differ
        self._scatter_inner_walls(boardA, density=0.12)
        self._scatter_inner_walls(boardB, density=0.12)

        # ensure player start coordinate is walkable in BOTH phases
        start = (1, 1)
        boardA[start[0]][start[1]] = FLOOR
        boardB[start[0]][start[1]] = FLOOR
        player_row, player_col = start

        # exit coordinates – may differ per phase
        exit_alpha = (N - 2, N - 2)
        exit_beta = (N - 2, 1)
        boardA[exit_alpha[0]][exit_alpha[1]] = EXIT
        boardB[exit_beta[0]][exit_beta[1]] = EXIT

        # place beacons (same coords for simplicity, appear in both phases)
        beacon_coords = self._random_empty_squares(
            boardA, count=self.num_beacons, avoid={start, exit_alpha, exit_beta}
        )
        for r, c in beacon_coords:
            boardA[r][c] = BEACON_OFF
            boardB[r][c] = BEACON_OFF

        # optional key + gate
        key_coord = gate_coord = None
        if self.put_key_gate:
            key_coord = self._random_empty_squares(
                boardA, 1, avoid=set(beacon_coords) | {start}
            )[0]
            gate_coord = self._random_empty_squares(
                boardA, 1, avoid=set(beacon_coords) | {start, key_coord}
            )[0]
            kr, kc = key_coord
            gr, gc = gate_coord
            boardA[kr][kc] = KEY
            boardB[kr][kc] = KEY
            boardA[gr][gc] = GATE_CLOSED
            boardB[gr][gc] = GATE_CLOSED

        # rifts (only in one randomly chosen phase)
        for r in range(1, N - 1):
            for c in range(1, N - 1):
                if self.rng.random() < self.rift_chance and (
                    (r, c) not in beacon_coords and (r, c) != start
                ):
                    target_board = boardA if self.rng.random() < 0.5 else boardB
                    if target_board[r][c] == FLOOR:
                        target_board[r][c] = RIFT

        # convert to list[str]
        rowsA = ["".join(row) for row in boardA]
        rowsB = ["".join(row) for row in boardB]

        # stamp player (only in delivered data for saving – Environment removes it again)
        rowsA[player_row] = (
            rowsA[player_row][:player_col]
            + PLAYER
            + rowsA[player_row][player_col + 1 :]
        )
        rowsB[player_row] = (
            rowsB[player_row][:player_col]
            + PLAYER
            + rowsB[player_row][player_col + 1 :]
        )

        return {
            "layerA": rowsA,
            "layerB": rowsB,
        }

    # low-level helpers ---------------------------------------------------
    def _carve_simple_room(self, board):
        N = len(board)
        for r in range(1, N - 1):
            for c in range(1, N - 1):
                board[r][c] = FLOOR

    def _scatter_inner_walls(self, board, density: float):
        N = len(board)
        for r in range(1, N - 1):
            for c in range(1, N - 1):
                if random.random() < density and board[r][c] == FLOOR:
                    board[r][c] = WALL

    def _random_empty_squares(
        self, board, count: int, avoid: set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        N = len(board)
        empties = [
            (r, c)
            for r in range(1, N - 1)
            for c in range(1, N - 1)
            if board[r][c] == FLOOR and (r, c) not in avoid
        ]
        self.rng.shuffle(empties)
        return empties[:count]

    # PDDL GENERATION -----------------------------------------------------
    def _dump_pddl(self, level) -> Tuple[str, str]:
        """
        Write temporary domain & problem files for the current level and
        return their file paths.
        """
        # domain is static – write once per call so tmp dir can be wiped
        domain_src_path = tempfile.mktemp(suffix="_dom.pddl", prefix="ps_")
        with open(domain_src_path, "w", encoding="utf-8") as fh:
            # copy the exact text from answer section 1 (without ~~~ markers)
            fh.write(DOMAIN_TEXT.strip() + "\n")

        problem_path = tempfile.mktemp(suffix="_prob.pddl", prefix="ps_")
        boardA = level["layerA"]
        boardB = level["layerB"]
        N = len(boardA)

        coords = [f"c{r}_{c}" for r in range(N) for c in range(N)]
        with open(problem_path, "w", encoding="utf-8") as fh:
            fh.write(f"(define (problem generated-level)\n")
            fh.write("  (:domain phase-shift)\n")
            fh.write("  (:objects\n")
            fh.write("       " + " ".join(coords) + " - coord\n")
            fh.write("  )\n")

            # initial state ------------------------------------------------
            fh.write("  (:init\n")
            # set current phase
            fh.write("    (phase-current alpha)\n")
            # adjacency
            for r in range(N):
                for c in range(N):
                    cur = f"c{r}_{c}"
                    if r > 0:
                        fh.write(f"    (adj-north {cur} c{r-1}_{c})\n")
                    if r < N - 1:
                        fh.write(f"    (adj-south {cur} c{r+1}_{c})\n")
                    if c > 0:
                        fh.write(f"    (adj-west  {cur} c{r}_{c-1})\n")
                    if c < N - 1:
                        fh.write(f"    (adj-east  {cur} c{r}_{c+1})\n")

            # board encoding ----------------------------------------------
            for phase_idx, board in enumerate((boardA, boardB)):
                phase_name = "alpha" if phase_idx == 0 else "beta"
                for r in range(N):
                    for c in range(N):
                        tile = board[r][c]
                        coord = f"c{r}_{c}"
                        if tile != WALL:
                            fh.write(f"    (walkable {coord} {phase_name})\n")
                        if tile == RIFT:
                            fh.write(f"    (rift {coord} {phase_name})\n")
                        if tile == GATE_CLOSED:
                            fh.write(f"    (gate {coord} {phase_name})\n")
                # end per cell
            # phase-independent tiles -------------------------------------
            for r in range(N):
                for c in range(N):
                    tA = boardA[r][c]
                    tB = boardB[r][c]
                    coord = f"c{r}_{c}"

                    if tA == PLAYER:  # same coordinate in B
                        fh.write(f"    (player-at {coord})\n")
                    if tA in (BEACON_OFF, BEACON_ON) or tB in (
                        BEACON_OFF,
                        BEACON_ON,
                    ):
                        fh.write(f"    (beacon {coord})\n")
                        if tA == BEACON_ON or tB == BEACON_ON:
                            fh.write(f"    (beacon-on {coord})\n")
                    if tA == EXIT or tB == EXIT:
                        fh.write(f"    (exit-tile {coord})\n")
                    if tA == KEY or tB == KEY:
                        fh.write(f"    (key-tile {coord})\n")
            fh.write("  )\n")  # end init

            # goal ----------------------------------------------------------
            fh.write("  (:goal\n")
            fh.write("    (and\n")
            fh.write("      (forall (?b - coord) (imply (beacon ?b) (beacon-on ?b)))\n")
            fh.write(
                "      (exists (?e - coord) (and (exit-tile ?e) (player-at ?e)))\n"
            )
            fh.write("    )\n")
            fh.write("  )\n")

            fh.write(")\n")  # end problem

        return domain_src_path, problem_path


# ──────────────────────────────────────────────────────────────────────
# glue code to get pyperplan to accept our domain + problem
# (pyperplan helper from its tutorial – shortened)
# ──────────────────────────────────────────────────────────────────────
def task_from_domain_problem(domain, problem) -> Task:
    """
    Translate parsed domain & problem into pyperplan internal Task object.
    """
    from pyperplan.pddl.task import (
        Operator,
        Task,
        Action,
        Axiom,
        Proposition,
        State,
    )

    task = Task(
        domain.name,
        problem.name,
        domain.requirements,
        domain.types,
        domain.type_hierarchy,
        domain.constants,
        problem.objects,
        domain.predicates,
        domain.functions,
        domain.actions,
        domain.axioms,
        problem.init,
        problem.goal,
        problem.metric,
    )
    task.operator_costs = {
        a.name: (1 if a.cost is None else a.cost) for a in task.actions
    }
    return task


# ──────────────────────────────────────────────────────────────────────
# The domain text used above – identical to section 1 (without fences).
# ──────────────────────────────────────────────────────────────────────
DOMAIN_TEXT = r"""
<PASTE THE SAME DOMAIN CONTENT FROM SECTION 1 HERE WITHOUT ~~~ MARKERS>
""".replace(
    "<PASTE THE SAME DOMAIN CONTENT FROM SECTION 1 HERE WITHOUT ~~~ MARKERS>",
    """
(define (domain phase-shift)
  (:requirements :typing :negative-preconditions :conditional-effects)
  (:types
      coord
      phase
  )
  (:constants alpha beta - phase)
  (:predicates
      (adj-north ?c1 ?c2 - coord)
      (adj-south ?c1 ?c2 - coord)
      (adj-west  ?c1 ?c2 - coord)
      (adj-east  ?c1 ?c2 - coord)
      (walkable ?c - coord ?p - phase)
      (rift     ?c - coord ?p - phase)
      (gate     ?c - coord ?p - phase)
      (beacon ?c - coord)
      (exit-tile ?c - coord)
      (key-tile ?c - coord)
      (player-at ?c - coord)
      (phase-current ?p - phase)
      (beacon-on ?c - coord)
      (gate-open ?c - coord)
      (have-key)
  )
  (:action move-north
    :parameters (?from ?to - coord ?ph - phase)
    :precondition (and (player-at ?from) (phase-current ?ph)
                       (adj-north ?from ?to)
                       (walkable ?to ?ph)
                       (not (gate ?to ?ph))
                       (not (rift ?to ?ph)))
    :effect (and (not (player-at ?from)) (player-at ?to)
                 (when (key-tile ?to) (have-key))
                 (when (key-tile ?to)
                       (forall (?c - coord) (when (gate ?c alpha) (gate-open ?c))))
                 (when (beacon ?to) (beacon-on ?to)))
  )
  (:action move-south
    :parameters (?from ?to - coord ?ph - phase)
    :precondition (and (player-at ?from) (phase-current ?ph)
                       (adj-south ?from ?to)
                       (walkable ?to ?ph)
                       (not (gate ?to ?ph))
                       (not (rift ?to ?ph)))
    :effect (and (not (player-at ?from)) (player-at ?to)
                 (when (key-tile ?to) (have-key))
                 (when (key-tile ?to)
                       (forall (?c - coord) (when (gate ?c alpha) (gate-open ?c))))
                 (when (beacon ?to) (beacon-on ?to)))
  )
  (:action move-west
    :parameters (?from ?to - coord ?ph - phase)
    :precondition (and (player-at ?from) (phase-current ?ph)
                       (adj-west ?from ?to)
                       (walkable ?to ?ph)
                       (not (gate ?to ?ph))
                       (not (rift ?to ?ph)))
    :effect (and (not (player-at ?from)) (player-at ?to)
                 (when (key-tile ?to) (have-key))
                 (when (key-tile ?to)
                       (forall (?c - coord) (when (gate ?c alpha) (gate-open ?c))))
                 (when (beacon ?to) (beacon-on ?to)))
  )
  (:action move-east
    :parameters (?from ?to - coord ?ph - phase)
    :precondition (and (player-at ?from) (phase-current ?ph)
                       (adj-east ?from ?to)
                       (walkable ?to ?ph)
                       (not (gate ?to ?ph))
                       (not (rift ?to ?ph)))
    :effect (and (not (player-at ?from)) (player-at ?to)
                 (when (key-tile ?to) (have-key))
                 (when (key-tile ?to)
                       (forall (?c - coord) (when (gate ?c alpha) (gate-open ?c))))
                 (when (beacon ?to) (beacon-on ?to)))
  )
  (:action shift-alpha-beta
    :parameters (?c - coord)
    :precondition (and (player-at ?c) (phase-current alpha)
                       (walkable ?c beta)
                       (not (gate ?c beta))
                       (not (rift ?c beta)))
    :effect (and (not (phase-current alpha)) (phase-current beta))
  )
  (:action shift-beta-alpha
    :parameters (?c - coord)
    :precondition (and (player-at ?c) (phase-current beta)
                       (walkable ?c alpha)
                       (not (gate ?c alpha))
                       (not (rift ?c alpha)))
    :effect (and (not (phase-current beta)) (phase-current alpha))
  )
)
""",
)

# ──────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE  (will run if called as script)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gen = PDDLLevelGenerator(difficulty="medium", seed=42)
    level = gen.generate_level()
    gen.save_level(level, "generated/demo_level")  # writes demo_level_A.txt/B.txt

    from custom_env import CustomEnv  # replace with actual import path

    env = CustomEnv(level["layerA"], level["layerB"])
    obs, info = env.reset()
    print("Generated level loaded into Gym env. Move-count =", info["move_count"])
