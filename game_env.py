import numpy as np


# 2048 game environment with a 4x4 board, log2 state encoding, and invalid move detection
class Game2048:

    # initialize board and score, then place two starting tiles
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.reset()

    # clear board and score, place two tiles, return initial state
    def reset(self):
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self._add_random_tile()
        self._add_random_tile()
        return self.get_state()

    # place a tile of value 2 with 90 percent probability or 4 otherwise in a random empty cell
    def _add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            r, c = empty_cells[np.random.randint(len(empty_cells))]
            self.board[r, c] = 2 if np.random.random() < 0.9 else 4

    # pack all nonzero values to the left and pad the rest with zeros
    def _compress(self, row):
        non_zero = row[row != 0]
        return np.pad(non_zero, (0, 4 - len(non_zero)), 'constant')

    # merge each adjacent equal pair once from left to right, zeroing the right element
    def _merge(self, row):
        score = 0
        for i in range(3):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                score += row[i]
                row[i + 1] = 0
        return row, score

    def _move_left(self, board):
        new_board = np.zeros_like(board)
        total_score = 0
        for i in range(4):
            row = self._compress(board[i])
            row, score = self._merge(row)
            new_board[i] = self._compress(row)
            total_score += score
        return new_board, total_score

    # apply the chosen action, detect invalid moves, spawn a tile, and return transition info
    def step(self, action):
        old_board = self.board.copy()

        # rotate or flip the board so the target direction maps to a leftward slide
        if action == 0:
            transformed = self.board.T.copy()
            new_transformed, score = self._move_left(transformed)
            new_board = new_transformed.T
        elif action == 1:
            transformed = np.flipud(self.board).T.copy()
            new_transformed, score = self._move_left(transformed)
            new_board = np.flipud(new_transformed.T)
        elif action == 2:
            new_board, score = self._move_left(self.board.copy())
        else:
            transformed = np.fliplr(self.board).copy()
            new_transformed, score = self._move_left(transformed)
            new_board = np.fliplr(new_transformed)

        invalid_move = np.array_equal(old_board, new_board)

        if not invalid_move:
            self.board = new_board
            self.score += score
            self._add_random_tile()

        done = self._is_game_over()
        info = {'invalid_move': invalid_move, 'score': self.score}
        return self.get_state(), score, done, info

    # check if the board has no empty cells and no adjacent equal tiles in any direction
    def _is_game_over(self):
        if 0 in self.board:
            return False
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False
                if self.board[j, i] == self.board[j + 1, i]:
                    return False
        return True

    # return the board as a flat 16d array with log2 encoding, zeros stay zero
    def get_state(self):
        flat = self.board.flatten().astype(np.float32)
        state = np.zeros(16, dtype=np.float32)
        mask = flat > 0
        state[mask] = np.log2(flat[mask])
        return state
