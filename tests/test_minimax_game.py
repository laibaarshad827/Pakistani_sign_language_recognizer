
import unittest

from minimax_game import check_winner, get_best_move


class TestMinimaxGame(unittest.TestCase):

    def test_detect_x_winner(self):
        board = ["X", "X", "X",
                 "O", "", "O",
                 "", "", ""]
        self.assertEqual(check_winner(board), "X")

    def test_detect_draw(self):
        board = ["X", "O", "X",
                 "X", "O", "O",
                 "O", "X", "X"]
        self.assertEqual(check_winner(board), "draw")

    def test_computer_takes_winning_move(self):
        board = ["X", "X", "",
                 "O", "O", "",
                 "", "", ""]
        self.assertEqual(get_best_move(board), 2)

    def test_computer_blocks_opponent(self):
        board = ["O", "O", "",
                 "X", "", "",
                 "", "", "X"]
        self.assertEqual(get_best_move(board), 2)

    def test_computer_returns_valid_move(self):
        board = ["", "", "",
                 "", "", "",
                 "", "", ""]
        move = get_best_move(board)
        self.assertIn(move, range(9))
        self.assertEqual(board[move], "")


if __name__ == "__main__":
    unittest.main()
