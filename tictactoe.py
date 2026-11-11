"""Noughts and crosses with pluggable agent policies.

Board cells are indexed 0-8, row by row:

     0 | 1 | 2
    ---+---+---
     3 | 4 | 5
    ---+---+---
     6 | 7 | 8

Three policies are provided, and any of them can be used for either side:

* RandomPolicy  - picks uniformly among the legal moves.
* FixedPolicy   - plays a predetermined, open-loop sequence of cells. It never
                  looks at the board; if its next planned cell is already taken,
                  or it has run out of planned moves, it cannot play and forfeits.
* OptimalPolicy - full minimax search (memoised). Never loses; ties between
                  equally good moves are broken randomly or deterministically.

Example (command line):

    python tictactoe.py --player optimal --opponent random --games 1000
    python tictactoe.py --player fixed:4,0,8 --opponent optimal --show
    python tictactoe.py --player random --opponent fixed:0,1,2 --first opponent
"""

from __future__ import annotations

import argparse
import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Optional, Sequence

EMPTY = " "
X = "X"
O = "O"

Board = tuple[str, ...]

WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)


def empty_board() -> Board:
    return (EMPTY,) * 9


def other(mark: str) -> str:
    return O if mark == X else X


def legal_moves(board: Board) -> list[int]:
    return [i for i, c in enumerate(board) if c == EMPTY]


def winner(board: Board) -> Optional[str]:
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_full(board: Board) -> bool:
    return EMPTY not in board


def apply_move(board: Board, move: int, mark: str) -> Board:
    if board[move] != EMPTY:
        raise ValueError(f"Cell {move} is already occupied")
    return board[:move] + (mark,) + board[move + 1:]


def render(board: Board) -> str:
    rows = []
    for r in range(3):
        cells = [board[3 * r + c] if board[3 * r + c] != EMPTY else str(3 * r + c)
                 for c in range(3)]
        rows.append(" " + " | ".join(cells))
    return "\n---+---+---\n".join(rows)


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------

class Policy(ABC):
    """An agent policy: maps a board (and the mark it plays) to a move."""

    name = "policy"

    def reset(self) -> None:
        """Called at the start of every game."""

    @abstractmethod
    def select_move(self, board: Board, mark: str) -> Optional[int]:
        """Return the cell index to play, or None if the policy cannot play
        (which forfeits the game)."""

    def __repr__(self) -> str:
        return self.name


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_move(self, board: Board, mark: str) -> Optional[int]:
        moves = legal_moves(board)
        return self.rng.choice(moves) if moves else None


class FixedPolicy(Policy):
    """Open-loop policy: plays `moves` in order, ignoring the board.

    If the next planned cell is occupied, or the plan is exhausted, it returns
    None and so loses the game.
    """

    def __init__(self, moves: Sequence[int]):
        for m in moves:
            if not 0 <= m <= 8:
                raise ValueError(f"Fixed move {m} is not a cell in 0-8")
        self.moves = list(moves)
        self._step = 0
        self.name = "fixed:" + ",".join(map(str, self.moves))

    def reset(self) -> None:
        self._step = 0

    def select_move(self, board: Board, mark: str) -> Optional[int]:
        if self._step >= len(self.moves):
            return None
        move = self.moves[self._step]
        self._step += 1
        return move if board[move] == EMPTY else None


@lru_cache(maxsize=None)
def minimax_value(board: Board, to_move: str) -> int:
    """Game value from the perspective of `to_move` under perfect play:
    +1 win, 0 draw, -1 loss. Faster wins / slower losses are not distinguished
    here; see `OptimalPolicy` for that."""
    w = winner(board)
    if w is not None:
        return 1 if w == to_move else -1
    if is_full(board):
        return 0
    return max(-minimax_value(apply_move(board, m, to_move), other(to_move))
               for m in legal_moves(board))


@lru_cache(maxsize=None)
def _scored_value(board: Board, to_move: str) -> int:
    """Like minimax_value but scaled by remaining moves, so the search prefers
    winning sooner and losing later."""
    w = winner(board)
    empties = board.count(EMPTY)
    if w is not None:
        return (empties + 1) if w == to_move else -(empties + 1)
    if empties == 0:
        return 0
    return max(-_scored_value(apply_move(board, m, to_move), other(to_move))
               for m in legal_moves(board))


def optimal_moves(board: Board, mark: str) -> list[int]:
    """All moves that achieve the best (depth-aware) minimax score."""
    scores = {m: -_scored_value(apply_move(board, m, mark), other(mark))
              for m in legal_moves(board)}
    if not scores:
        return []
    best = max(scores.values())
    return [m for m, s in scores.items() if s == best]


class OptimalPolicy(Policy):
    """Perfect play via minimax. If `seed` is None ties are broken by lowest
    cell index (fully deterministic); otherwise they are broken randomly."""

    name = "optimal"

    def __init__(self, seed: Optional[int] = None, randomise_ties: bool = False):
        self.randomise_ties = randomise_ties or seed is not None
        self.rng = random.Random(seed)

    def select_move(self, board: Board, mark: str) -> Optional[int]:
        best = optimal_moves(board, mark)
        if not best:
            return None
        return self.rng.choice(best) if self.randomise_ties else best[0]


def make_policy(spec: str, seed: Optional[int] = None) -> Policy:
    """Build a policy from a string spec:
    'random', 'optimal', or 'fixed:<cell>,<cell>,...' (e.g. 'fixed:4,0,8')."""
    kind, _, arg = spec.partition(":")
    kind = kind.strip().lower()
    if kind == "random":
        return RandomPolicy(seed)
    if kind == "optimal":
        return OptimalPolicy(seed)
    if kind == "fixed":
        if not arg:
            raise ValueError("fixed policy needs moves, e.g. 'fixed:4,0,8'")
        return FixedPolicy([int(x) for x in arg.split(",")])
    raise ValueError(f"Unknown policy '{spec}' (use random, optimal, fixed:...)")


# --------------------------------------------------------------------------
# Game loop
# --------------------------------------------------------------------------

class Outcome(Enum):
    PLAYER_WIN = "player wins"
    OPPONENT_WIN = "opponent wins"
    DRAW = "draw"


@dataclass
class GameResult:
    outcome: Outcome
    moves: list[tuple[str, int]] = field(default_factory=list)  # (mark, cell)
    final_board: Board = field(default_factory=empty_board)
    forfeit: bool = False  # True if the loser lost by being unable to move


def play_game(player: Policy, opponent: Policy, player_first: bool = True,
              verbose: bool = False) -> GameResult:
    """Play one game. The side that moves first plays X."""
    player.reset()
    opponent.reset()
    player_mark = X if player_first else O
    agents = {player_mark: player, other(player_mark): opponent}

    def outcome_for(mark: str) -> Outcome:
        return Outcome.PLAYER_WIN if mark == player_mark else Outcome.OPPONENT_WIN

    board = empty_board()
    to_move = X
    history: list[tuple[str, int]] = []

    while True:
        agent = agents[to_move]
        move = agent.select_move(board, to_move)
        if move is None or not 0 <= move <= 8 or board[move] != EMPTY:
            if verbose:
                print(f"{to_move} ({agent}) cannot play - forfeits.\n")
            return GameResult(outcome_for(other(to_move)), history, board, forfeit=True)

        board = apply_move(board, move, to_move)
        history.append((to_move, move))
        if verbose:
            print(f"{to_move} ({agent}) plays {move}\n{render(board)}\n")

        if winner(board) == to_move:
            return GameResult(outcome_for(to_move), history, board)
        if is_full(board):
            return GameResult(Outcome.DRAW, history, board)
        to_move = other(to_move)


def play_many(player: Policy, opponent: Policy, games: int,
              player_first: bool = True) -> Counter:
    return Counter(play_game(player, opponent, player_first).outcome
                   for _ in range(games))


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Noughts and crosses with agent policies.")
    p.add_argument("--player", default="optimal",
                   help="random | optimal | fixed:c1,c2,... (default: optimal)")
    p.add_argument("--opponent", default="random",
                   help="random | optimal | fixed:c1,c2,... (default: random)")
    p.add_argument("--first", choices=["player", "opponent"], default="player",
                   help="who moves first and plays X (default: player)")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--show", action="store_true",
                   help="print every move (default when --games is 1)")
    args = p.parse_args(argv)

    seed2 = None if args.seed is None else args.seed + 1
    player = make_policy(args.player, args.seed)
    opponent = make_policy(args.opponent, seed2)
    player_first = args.first == "player"

    if args.games == 1 or args.show:
        for _ in range(args.games):
            print(render(empty_board()) + "\n")
            result = play_game(player, opponent, player_first, verbose=True)
            print(f"Result: {result.outcome.value}"
                  + (" (by forfeit)" if result.forfeit else "") + "\n")
        if args.games == 1:
            return

    tally = play_many(player, opponent, args.games, player_first)
    print(f"{args.games} games: player={player} vs opponent={opponent}, "
          f"{args.first} first")
    for o in Outcome:
        n = tally[o]
        print(f"  {o.value:14s} {n:6d}  ({100 * n / args.games:5.1f}%)")


if __name__ == "__main__":
    main()
