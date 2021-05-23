from gym.envs.registration import register

from .lstm_grid import LSTM_Grid

environments = [['LSTM_Grid', 'v0']]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='grid:{}'.format(environment[0]),
        nondeterministic=True
    )
