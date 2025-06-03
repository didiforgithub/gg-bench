import copy
from typing import List, Tuple, Optional, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomEnv(gym.Env):
    """
    ======================================================================
    PHASE SHIFT – gym-compatible single-agent environment
    ----------------------------------------------------------------------
    Action mapping
        0 : move north   (row –1)
        1 : move south   (row +1)
        2 : move west    (col –1)
        3 : move east    (col +1)
        4 : phase-shift  (Alpha ⇄ Beta)
    ----------------------------------------------------------------------
    Reward scheme
        +100  – puzzle solved
        -50   – rift death or move-limit exceeded
        -1    – every executed step while game is running
    ----------------------------------------------------------------------
    Observation
        A 2-D integer numpy array with shape (rows, cols) that depicts the
        CURRENT phase.  The player’s square is encoded with a dedicated
        value so the agent sees the avatar’s location directly.
            0 : void / blank          6 : key (not yet collected)
            1 : floor                 7 : closed gate
            2 : wall                  8 : open gate
            3 : beacon (inactive)     9 : rift (lethal)
            4 : beacon (active)      10 : player (overlay)
            5 : exit
    ======================================================================
    """

    metadata = {"render.modes": ["ansi"]}

    # ────────────────────────────────────────────────────────────── helpers
    _CHAR_TO_CODE = {
        " ": 0,
        ".": 1,
        "#": 2,
        "B": 3,
        "b": 4,
        "E": 5,
        "k": 6,
        "G": 7,
        "g": 8,
        "X": 9,
    }
    _CODE_TO_CHAR = {v: k for k, v in _CHAR_TO_CODE.items()}
    _PLAYER_CODE = 10

    _DIRS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # n, s, w, e

    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        levelA: Optional[List[str]] = None,
        levelB: Optional[List[str]] = None,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # If no external level supplied use a small bundled 5 × 5 demo level
        # ------------------------------------------------------------------
        if levelA is None or levelB is None:
            levelA = [
                "#####",
                "#P.B#",
                "#.#.#",
                "#..E#",
                "#####",
            ]
            levelB = [
                "#####",
                "#P..#",
                "#.#B#",
                "#..E#",
                "#####",
            ]

        self.boardA_raw = [list(row) for row in levelA]
        self.boardB_raw = [list(row) for row in levelB]

        self.rows = len(self.boardA_raw)
        self.cols = len(self.boardA_raw[0])

        assert self.rows == len(self.boardB_raw) and self.cols == len(
            self.boardB_raw[0]
        ), "Phase boards must be of identical size."

        # Maximum move counts derived from official spec
        if self.rows == 5:
            self.move_limit = 50
        elif self.rows == 7:
            self.move_limit = 75
        elif self.rows == 9:
            self.move_limit = 90
        else:  # arbitrary size – proportional heuristic
            self.move_limit = int(self.rows * self.cols * 2)

        # Gym spaces -------------------------------------------------------
        self.action_space = spaces.Discrete(5)

        # Each cell encoded as int 0-10 (inclusive)
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(self.rows, self.cols),
            dtype=np.int8,
        )

        # ------------------------------------------------------------------
        self._seed()  # registers the RNG
        self.reset()

    # ──────────────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Deep copies so originals stay untouched for consecutive episodes
        self.boardA = copy.deepcopy(self.boardA_raw)
        self.boardB = copy.deepcopy(self.boardB_raw)

        self.beacons: Dict[Tuple[int, int], bool] = {}
        self.key_collected = False

        # Locate player & beacons, scrub 'P' from raw boards
        for r in range(self.rows):
            for c in range(self.cols):
                for board in (self.boardA, self.boardB):
                    if board[r][c] == "P":
                        self.player_pos = (r, c)
                        board[r][c] = "."
                if self.boardA[r][c] in ("B", "b") or self.boardB[r][c] in (
                    "B",
                    "b",
                ):
                    self.beacons[(r, c)] = (
                        self.boardA[r][c] == "b" or self.boardB[r][c] == "b"
                    )

        self.phase = 0  # 0 = Alpha, 1 = Beta
        self.move_counter = 0
        self.terminated = False
        self.truncated = False

        return self._get_obs(), self._info()

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.terminated or self.truncated:
            return self._get_obs(), 0.0, self.terminated, self.truncated, self._info()

        reward = -1.0  # default step penalty

        if action in self._DIRS:
            dr, dc = self._DIRS[action]
            reward += self._attempt_move(dr, dc)
        elif action == 4:
            reward += self._attempt_phase_shift()
        else:
            # unknown action -> treated as wasted move
            pass

        self.move_counter += 1

        # Check for move-limit overrun
        if not self.terminated and self.move_counter > self.move_limit:
            self.truncated = True
            reward = -50.0

        obs = self._get_obs()
        info = self._info()
        return obs, reward, self.terminated, self.truncated, info

    # ------------------------------------------------------------------
    def render(self) -> str:
        board = self.boardA if self.phase == 0 else self.boardB
        out = []
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                if (r, c) == self.player_pos:
                    row_chars.append("P")
                else:
                    row_chars.append(board[r][c])
            out.append("".join(row_chars))
        return "\n".join(out)

    # ------------------------------------------------------------------
    def valid_moves(self) -> List[int]:
        valids: List[int] = []
        r, c = self.player_pos

        # four cardinal moves
        for act, (dr, dc) in self._DIRS.items():
            if self._is_walkable(r + dr, c + dc):
                valids.append(act)

        # phase shift
        if self._can_shift():
            valids.append(4)

        return valids

    # ------------------------------------------------------------------
    def is_solved(self) -> bool:
        if not all(self.beacons.values()):
            return False
        board = self.boardA if self.phase == 0 else self.boardB
        return board[self.player_pos[0]][self.player_pos[1]] == "E"

    # ──────────────────────────────────────────────────────────────────────
    # Internal mechanics
    # ──────────────────────────────────────────────────────────────────────
    def _attempt_move(self, dr: int, dc: int) -> float:
        """Returns additional reward/penalty beyond the -1 step cost."""
        r, c = self.player_pos
        nr, nc = r + dr, c + dc

        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            # outside map – wasted move
            return 0.0

        tile = self._tile(nr, nc)

        # Walls, closed gates, voids = illegal
        if tile in ("#", "G", " "):
            return 0.0

        # Rift = instant failure
        if tile == "X":
            self.player_pos = (nr, nc)
            self.truncated = True
            return -50.0

        # Legal move – update position
        self.player_pos = (nr, nc)

        # Key handling
        if tile == "k" and not self.key_collected:
            self.key_collected = True
            self._open_all_gates()

        # Beacon activation
        if tile in ("B", "b"):
            self.beacons[(nr, nc)] = True
            self._set_tile(nr, nc, "b")  # both phases

        # Check win
        if self.is_solved():
            self.terminated = True
            return 101.0  # neutralises step-penalty: -1 + 101 = +100

        return 0.0

    # ------------------------------------------------------------------
    def _attempt_phase_shift(self) -> float:
        if not self._can_shift():
            return 0.0  # illegal shift – wasted move

        self.phase ^= 1  # toggle 0 ↔ 1

        # Check rift under feet after shift
        if self._tile(*self.player_pos) == "X":
            self.truncated = True
            return -50.0 + 0.0  # -1 step already counted, so add -50

        if self.is_solved():
            self.terminated = True
            return 101.0  # compensate for step penalty

        return 0.0

    # ------------------------------------------------------------------
    def _can_shift(self) -> bool:
        board_target = self.boardB if self.phase == 0 else self.boardA
        tile = board_target[self.player_pos[0]][self.player_pos[1]]
        return tile not in (" ", "#", "G")

    # ------------------------------------------------------------------
    def _is_walkable(self, r: int, c: int) -> bool:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        tile = self._tile(r, c)
        return tile in (".", "B", "b", "E", "k", "g")

    # ------------------------------------------------------------------
    def _tile(self, r: int, c: int) -> str:
        board = self.boardA if self.phase == 0 else self.boardB
        return board[r][c]

    # ------------------------------------------------------------------
    def _set_tile(self, r: int, c: int, ch: str):
        self.boardA[r][c] = ch
        self.boardB[r][c] = ch

    # ------------------------------------------------------------------
    def _open_all_gates(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.boardA[r][c] == "G":
                    self.boardA[r][c] = "g"
                if self.boardB[r][c] == "G":
                    self.boardB[r][c] = "g"

    # ------------------------------------------------------------------
    def _encode_board(self) -> np.ndarray:
        board = self.boardA if self.phase == 0 else self.boardB
        arr = np.zeros((self.rows, self.cols), dtype=np.int8)

        for r in range(self.rows):
            for c in range(self.cols):
                arr[r, c] = self._CHAR_TO_CODE[board[r][c]]
        pr, pc = self.player_pos
        arr[pr, pc] = self._PLAYER_CODE
        return arr

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        return self._encode_board()

    # ------------------------------------------------------------------
    def _info(self) -> Dict:
        return {
            "phase": "Alpha" if self.phase == 0 else "Beta",
            "move_count": self.move_counter,
            "move_limit": self.move_limit,
            "beacons_activated": sum(self.beacons.values()),
            "beacons_total": len(self.beacons),
            "key_collected": self.key_collected,
        }

    # ------------------------------------------------------------------
    def _seed(self, seed: Optional[int] = None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
