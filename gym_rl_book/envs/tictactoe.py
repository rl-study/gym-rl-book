import gym

from gym import error, spaces, utils
from tabulate import tabulate


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.BOARD_SIZE = 3
        self.EMPTY = ' '
        self.FIRST_STONE = 'X'
        self.SECOND_STONE = 'O'
        self.STONE_TYPE_COUNT = 2

        self.action_space = spaces.MultiDiscrete([self.BOARD_SIZE, self.BOARD_SIZE, self.STONE_TYPE_COUNT])
        self.observation_space = spaces.MultiDiscrete([self.BOARD_SIZE, self.BOARD_SIZE])

        self.reset()

    def step(self, action):
        if self.done:
            raise error.ResetNeeded("")

        r, c, stone = action
        if self.board[r][c] != self.EMPTY:
            raise error.InvalidAction("Stone '{}' already exists in row: {}, col: {}".format(self.board[r][c], r, c))

        if stone not in [self.FIRST_STONE, self.SECOND_STONE]:
            raise error.InvalidAction("Unknown stone type '{}'", stone)

        if stone == self.last_stone:
            raise error.InvalidAction("Need to change stone.")

        self.board[r][c] = stone
        self.last_stone = stone
        self.remaining_place -= 1

        reward, self.done = self._check_status()

        return self.board, reward, self.done, {}

    def _check_status(self):
        rows = self.board
        cols = list(zip(*self.board))
        diag_one = []
        diag_two = []
        for i in range(self.BOARD_SIZE):
            diag_one.append(self.board[i][i])
            diag_two.append(self.board[i][self.BOARD_SIZE - i - 1])

        candidates = rows + cols + [diag_one, diag_two]
        if [self.FIRST_STONE] * self.BOARD_SIZE in candidates:
            return 1, True
        elif [self.SECOND_STONE] * self.BOARD_SIZE in candidates:
            return -1, True
        else:
            return 0, self.remaining_place == 0

    def reset(self):
        self.board = [[self.EMPTY] * self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]
        self.remaining_place = self.BOARD_SIZE * self.BOARD_SIZE
        self.done = False
        self.last_stone = self.SECOND_STONE

    def render(self, mode='human', close=False):
        print(tabulate(self.board, tablefmt='grid'))
