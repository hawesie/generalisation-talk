import unittest

from tictactoe import (
    EMPTY, O, X, FixedPolicy, OptimalPolicy, Outcome, RandomPolicy,
    empty_board, make_policy, minimax_value, play_game, play_many,
)


def board_from(s: str):
    return tuple(EMPTY if c == "." else c for c in s)


class TestOptimal(unittest.TestCase):
    def test_empty_board_is_a_draw(self):
        self.assertEqual(minimax_value(empty_board(), X), 0)

    def test_takes_immediate_win(self):
        b = board_from("XX.OO....")
        self.assertEqual(OptimalPolicy().select_move(b, X), 2)

    def test_blocks_immediate_loss(self):
        b = board_from("OO..X....")
        self.assertEqual(OptimalPolicy().select_move(b, X), 2)

    def test_self_play_draws(self):
        for seed in range(20):
            r = play_game(OptimalPolicy(seed), OptimalPolicy(seed + 100))
            self.assertEqual(r.outcome, Outcome.DRAW)

    def test_never_loses_to_random(self):
        for first in (True, False):
            tally = play_many(OptimalPolicy(1), RandomPolicy(2), 300, first)
            self.assertEqual(tally[Outcome.OPPONENT_WIN], 0)
            # and as opponent
            tally = play_many(RandomPolicy(3), OptimalPolicy(4), 300, first)
            self.assertEqual(tally[Outcome.PLAYER_WIN], 0)


class TestFixed(unittest.TestCase):
    def test_follows_plan_and_wins(self):
        # Player: 0,1,2 (top row). Opponent: 3,4,... never blocks.
        r = play_game(FixedPolicy([0, 1, 2]), FixedPolicy([3, 4, 5]))
        self.assertEqual(r.outcome, Outcome.PLAYER_WIN)
        self.assertEqual(r.moves, [(X, 0), (O, 3), (X, 1), (O, 4), (X, 2)])
        self.assertFalse(r.forfeit)

    def test_forfeits_when_cell_taken(self):
        r = play_game(FixedPolicy([4, 0]), FixedPolicy([4]))
        self.assertEqual(r.outcome, Outcome.PLAYER_WIN)
        self.assertTrue(r.forfeit)

    def test_forfeits_when_plan_exhausted(self):
        r = play_game(FixedPolicy([0]), FixedPolicy([4, 8]))
        self.assertEqual(r.outcome, Outcome.OPPONENT_WIN)
        self.assertTrue(r.forfeit)

    def test_reset_between_games(self):
        p = FixedPolicy([0, 1, 2])
        for _ in range(3):
            r = play_game(p, FixedPolicy([3, 4, 5]))
            self.assertEqual(r.outcome, Outcome.PLAYER_WIN)

    def test_loses_to_optimal(self):
        r = play_game(FixedPolicy([0, 1, 2, 3, 5]), OptimalPolicy())
        self.assertEqual(r.outcome, Outcome.OPPONENT_WIN)

    def test_invalid_cell(self):
        with self.assertRaises(ValueError):
            FixedPolicy([9])


class TestRandomAndFactory(unittest.TestCase):
    def test_random_always_legal(self):
        tally = play_many(RandomPolicy(0), RandomPolicy(1), 500)
        self.assertEqual(sum(tally.values()), 500)

    def test_make_policy(self):
        self.assertIsInstance(make_policy("random"), RandomPolicy)
        self.assertIsInstance(make_policy("optimal"), OptimalPolicy)
        f = make_policy("fixed:4,0,8")
        self.assertIsInstance(f, FixedPolicy)
        self.assertEqual(f.moves, [4, 0, 8])
        with self.assertRaises(ValueError):
            make_policy("greedy")

    def test_player_second_plays_O(self):
        r = play_game(FixedPolicy([0]), FixedPolicy([4]), player_first=False)
        self.assertEqual(r.moves[0], (X, 4))


if __name__ == "__main__":
    unittest.main()
