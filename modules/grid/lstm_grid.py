import gym
import numpy as np
from gym import spaces
import copy

np.random.seed(0)
_gym_disable_underscore_compat = True
class LSTM_Grid(gym.core.Env):
    """
    This is a simple env where the agent must learn to maximize price while minimizing the difference between the demand and supply
    """
    metadata = {'render.modes': ['console']}
    # Define constants for clearer code
    SUPPLY_LOWER_BOUND=0
    SUPPLY_UPPER_BOUND=10
    DEMAND_LOWER_BOUND=0
    DEMAND_UPPER_BOUND=10
    PRICE_LOWER_BOUND=0
    PRICE_UPPER_BOUND=30
    LEFT = 0
    RIGHT = 1
    TOTAL_ACTIONS = 10

    def __init__(self):
        super(LSTM_Grid, self).__init__()

        # self.current_observation[2] = np.random.randint(0, high=self.PRICE_UPPER_BOUND)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: increase and decrease in price
        self.action_space = spaces.Discrete(self.TOTAL_ACTIONS)

        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space

        # observation space: [supply, demand, price]
        self.observation_space = spaces.Box(
                        low=np.array([self.SUPPLY_LOWER_BOUND, self.DEMAND_LOWER_BOUND, self.PRICE_LOWER_BOUND]),
                        high=np.array([self.SUPPLY_UPPER_BOUND, self.DEMAND_UPPER_BOUND, self.PRICE_UPPER_BOUND]),
                        dtype=np.float64)

        # Initialize [supply, demand, price] randomly
        self.current_observation = self.observation_space.sample()

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize [supply, demand, price] randomly
        self.current_observation = self.observation_space.sample()
        self.history = []

        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array(self.current_observation).astype(np.float32)

    def step(self, action):
        old_observation = copy.deepcopy(self.current_observation)
        self.history.append(old_observation)
        old_supply = old_observation[0]
        old_demand = old_observation[1]
        old_price = old_observation[2]

        # change price
        def change_price(original_price, price_lb, price_ub, action, total_num_actions):
            max_change = (price_ub - price_lb) / 2
            # what we want is that the most extreme actions, should only have
            # max_change effect
            # (correcting_factor * total_num_actions / 2)**3 == max_change
            correcting_factor = 2 * (max_change ** (1/3)) / total_num_actions
            corrected_action = action - (total_num_actions / 2)
            new_price = original_price + (correcting_factor * corrected_action) ** 3
            return new_price

        corrected_action = action - (self.TOTAL_ACTIONS / 2)
        # self.current_observation[2] += corrected_action / (self.TOTAL_ACTIONS / 2)
        self.current_observation[2] = change_price(
                old_price, self.PRICE_LOWER_BOUND, self.PRICE_UPPER_BOUND, action, self.TOTAL_ACTIONS)
        # print(f"Action := {action - (self.TOTAL_ACTIONS / 2)}")
        # Limit value betwen the upper and lower bounds
        self.current_observation[2] = np.clip(self.current_observation[2], self.PRICE_LOWER_BOUND, self.PRICE_UPPER_BOUND)

        # change supply
        # supply_{n} = supply_{n - 1} + \gamma(demand_{n - 1} - supply_{n - 1})
        gamma = 0.000001
        supply = self.current_observation[0]
        supply = old_supply + gamma * (old_demand - old_supply)
        self.current_observation[0] = supply
        self.current_observation[0] = np.clip(
                self.current_observation[0],
                self.SUPPLY_LOWER_BOUND,
                self.SUPPLY_UPPER_BOUND)

        # change demand
        # demand_{n} = demand_{n - 1} - \alpha (price_{n} - price_{n - 1})
        alpha = 0.2
        N = 10
        average_of_last_n_demand = np.average([e[1] for e in self.history[-N:]])
        demand = average_of_last_n_demand - alpha * (self.current_observation[2] - old_price)
        self.current_observation[1] = demand
        self.current_observation[1] = np.clip(
                self.current_observation[1],
                self.DEMAND_LOWER_BOUND,
                self.DEMAND_UPPER_BOUND)

        # calculate reward
        # reward_{n} = price_{n-1} * min(demand_{n-1}, supply_{n-1}) - \beta(demand_{n-1} - supply_{n-1})^{2}
        # \beta = 0.2
        # the higher the value of \beta, the more importance we give to
        # ensuring that demand and supply are equal.
        beta = 0.5
        mu = 0.01
        reward = mu * (old_price * min(old_demand, old_supply)) / (beta * ((old_demand - old_supply) ** 2) + 1)

        # set done
        done = False

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array(self.current_observation).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        # if mode != 'console':
        #     raise NotImplementedError()
        print(f"Rendering | {self.current_observation}")
        pass

    def close(self):
        pass
