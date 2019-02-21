import gym

from gym import error, spaces, utils
from tabulate import tabulate

import copy

class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.BOARD_SIZE = 3
        self.EMPTY = ' '
        self.STONE_TYPE_COUNT = 2
        self.STONES = ['X', 'O']

        self.action_space = spaces.MultiDiscrete([self.BOARD_SIZE, self.BOARD_SIZE, self.STONE_TYPE_COUNT])
        self.observation_space = spaces.MultiDiscrete([self.BOARD_SIZE, self.BOARD_SIZE])

        self.reset()

    def step(self, action):
        if self.done:
            raise error.ResetNeeded("")

        r, c, stone = action
        if self.board[r][c] != self.EMPTY:
            raise error.InvalidAction("Stone '{}' already exists in row: {}, col: {}".format(self.board[r][c], r, c))

        if stone >= self.STONE_TYPE_COUNT:
            raise error.InvalidAction("Unknown stone type '{}'".format(stone))

        if stone == self.last_stone:
            raise error.InvalidAction("Need to change stone.")

        self.board[r][c] = self.STONES[stone]
        self.last_stone = self.STONES[stone]
        self.remaining_place -= 1

        reward, self.done = self._check_status()

        return copy.deepcopy(self.board), reward, self.done, {}

    def _check_status(self):
        rows = self.board

        cols = []
        for c in range(self.BOARD_SIZE):
            col = []
            for r in range(self.BOARD_SIZE):
                col.append(self.board[r][c])
            cols.append(col)

        diag_one = []
        diag_two = []
        for i in range(self.BOARD_SIZE):
            diag_one.append(self.board[i][i])
            diag_two.append(self.board[i][self.BOARD_SIZE - i - 1])

        candidates = rows + cols + [diag_one, diag_two]
        if [self.STONES[0]] * self.BOARD_SIZE in candidates:
            return 1, True
        elif [self.STONES[1]] * self.BOARD_SIZE in candidates:
            return -1, True
        else:
            return 0, self.remaining_place == 0

    def reset(self):
        self.board = [[self.EMPTY] * self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]
        self.remaining_place = self.BOARD_SIZE * self.BOARD_SIZE
        self.done = False
        self.last_stone = self.STONES[1]
        return copy.deepcopy(self.board)

    def render(self, mode='human', close=False):
        print(tabulate(self.board, tablefmt='grid'))

    # Useful util methods for tic-tac-toe
    def legal_moves(self):
        moves = []
        if self.done:
            return moves

        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r][c] == self.EMPTY:
                    moves.append((r, c))
        return moves
