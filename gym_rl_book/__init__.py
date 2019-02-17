from gym.envs.registration import register

register(
    id='TicTacToe-v0',
    entry_point='gym_rl_book.envs:TicTacToeEnv',
)

register(
    id='MultiArmBandit-v0',
    entry_point='gym_rl_book.envs:MultiArmBanditEnv'
)