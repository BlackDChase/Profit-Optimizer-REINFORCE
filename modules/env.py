import gym
import pandas as pd
import numpy as np
from collections import deque
import random
import log
import multiprocessing

__author__ = 'Biribiri,BlackDChase'
__version__ = '1.3.1'

# A class for encapsulating the dataset
# This class maintains the original dataset along with its related parameters and functions
class DatasetHelper:
    def __init__(self, dataset_path, max_input_len):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path)
        self.max_input_len = max_input_len
    
    def reset(self):
        """
        This function returns a random starting state from a random timestep based on the max length allowed 
        and returns it as a numpy array
        """
        random_index = random.randint(0, len(self.df) - self.max_input_len + 1)
        self.first_input = self.df.iloc[random_index:random_index + self.max_input_len, :].values
        return self.first_input

class LSTMEnv(gym.Env):
    def __init__(self,
                 model,
                 dataset_path="../datasets/normalized_weird_13_columns_with_supply.csv",
                 min_max_values_csv_file="../datasets/min_max_values_13_columns_with_supply.csv",
                 max_input_len=25,
                 actionSpace=[-15,-10,0,10,15],
                 debug=False):
        """
        model = trained LSTM model from lstm.py

        By default, all the data (including the dataset helper's dataset) are
        all normalized.
        Only when sending data to the agent, is the data de-normalized before
        sending it. This includes second-order data sent to the agent such as
        reward.
        """
        self.model = model
        self.dataset_helper = DatasetHelper(dataset_path,max_input_len)

        # Initialize self.observation_space
        # required for self.current_observation
        self.observation_space = gym.spaces.Box(low=-np.inf,high=np.inf,shape=(self.model.input_dim,))

        # create model input deque
        self.model_input = deque([], maxlen=max_input_len)
        self.max_input_len = max_input_len
        self.actionSpace=actionSpace

        # list of min / max values for each of the 13 columns used for wrapping inputs and unwrapping outputs
        self.min_max_values = pd.read_csv(min_max_values_csv_file)

        # In case debug is enabled, by default debugging is disabled
        self.debug=debug

    def reset(self):
        """
        Return a value within self.observation_space
        Why clear input?
        """
        self.model_input.clear()
        dataset_helper_input = self.dataset_helper.reset()
        [self.model_input.append(element) for element in dataset_helper_input]

        # convert to numpy version
        np_model_input = np.array(self.model_input)

        """
        Saving starting states for testing
        """
        self.startState = []
        [self.startState.append(element) for element in dataset_helper_input]

        curr = multiprocessing.current_process()
        if self.debug:
            log.debug(f"Reset call for {curr.name}")
        current_observation = self.model.forward(np_model_input, numpy=True)

        """
        Starting with a preset positive price which is 0.5 (because normalized).
        Saving oldPrice for future refernce as our currnet Agent commands a percentage change on the oldPrice,
        rather than dishing out a new price.
        We can also have this start price as a random value between 0 and 1.
        oldPrice is going to be a normalized value as it always is calculated in respect to
        current_observation (which is normalized).
        """
        price_index = 0
        current_observation[price_index] = float(np.random.rand(1)*(0.6)+0.2)
        self.current_observation = current_observation
    
        # Denormalizing current observation for storing in logs along with normalized observations
        self.denormalized_current_observation = self.denormalize(self.current_observation)
        self.oldPrice = self.current_observation[price_index]

        if self.debug:
            log.debug(f"Reset complete for {curr.name}")
            log.debug(f">current_observation = {self.current_observation}")
            log.debug(f">denormalized_current_observation = {self.denormalized_current_observation}")
            log.debug(f">oldPrice = {self.oldPrice}")
        return self.current_observation

    def possibleState(self,time=100):
        """
        Output  : Return the output of the enviornment for `time` number of steps without the feedback of A3C agent.
        """
        states = []
        model_input = deque([], maxlen=self.max_input_len)
        [model_input.append(element) for element in self.startState]

        for i in range(time+1):
            np_model_input = np.array(model_input)
            current_observation = self.model.forward(np_model_input, numpy=True)
            model_input.append(current_observation)
            current_observation = self.denormalize(current_observation)
            log.info(f"Possible set {i} = {current_observation}")
            states.append(current_observation)
            denormalized_current_observation = self.denormalize(current_observation)
            # Logging normalState (after denormalization)
            log.info(f"Normal State = {denormalized_current_observation}")

        return states

    def step(self, action, LOG=False):
        """
        Calculate new current observation
        Calculate reward
        Return relevant data
        """
        if action < 0 or action >= len(self.actionSpace):
            log.info(f"Illegal action = {action}")
            log.debug(f"Action Space = {self.actionSpace}")
            import sys
            sys.exit()

        # get the next observation
        numpy_model_input = np.array(self.model_input)
        self.current_observation = self.model.forward(numpy_model_input, numpy=True)
        # Set done as False as there's no reason to end an episode
        done = False

        # Implement effects of action
        new_price = self.get_new_price(action)
        self.oldPrice=new_price
        price_index = 0
        self.current_observation[price_index] = new_price
        self.denormalized_current_observation = self.denormalize(self.current_observation)

        # get reward
        # Ensure that numpy array shape is (1,), not () otherwise conversion to torch.Tensor will get messed up
        # Use denormalized new price to get denormalized reward
        denormalized_reward = np.array([self.get_reward(denormalize=True,LOG=LOG)])

        # We update the price in the current observation
        # This ensures that the model takes into account the action we just
        # took when giving us the next timestep
        # append the current observation to the model input
        self.model_input.append(self.current_observation)
        if self.debug:
            log.debug(f">current_observation = {self.current_observation}")
            log.debug(f">denormalized_current_observation = {self.denormalized_current_observation}")
        return self.current_observation, denormalized_reward, done, {}

    def get_new_price(self, action):
        """
        Modify the price by a percentage according to the action taken.
        """
        old_price = self.oldPrice
        # Increase or decrease the old price by a percentage, as defined by actions
        new_price = old_price * (1 + (self.actionSpace[action])/ 100)
        return new_price

    def get_reward(self, denormalize=False,LOG=False):

        """
        Calculate reward based on the new_price

        Demand is always positive in the dataset
        Prices can be negative in the dataset

        We cannot use min(demand, supply) * price because it does not take into
        account times when supply is greater than demand, meaning the
        electricity would either be sent at a loss, or would be bought from
        these smaller producers.
        """

        # Indices of attributes based on the Ontario dataset we are using 
        price_index = 0
        ontario_demand_index = 1
        supply_index = 2

        """
        The networks are trained based on normalized values of the env states.
        But the reward calculation has to be done using denormalized (original) values of supply,demand and price
        
        Since the reward calculations using normalized values appeared to be less rewarding 
        than expected in certain cases or was going negative where it shouldnt be so we decided to use denormalized 
        values of supply,demand and price to compute rewards.

        So we are primarily using denormalized values for reward calculation for now.
        This reward calculation can be done using normalized values if denormalize is set to False while calling this function.   
        """

        # Parameters are denormalized for reward calculation 
        if denormalize:
            self.denormalized_current_observation = self.denormalize(self.current_observation)

            log.debug(f"self.denormalized_current_observation.shape = {self.denormalized_current_observation.shape}")
            demand = self.denormalized_current_observation[ontario_demand_index]
            supply = self.denormalized_current_observation[supply_index]
            new_price = self.denormalized_current_observation[price_index]
        # OtherWise normalized values are used 
        else:
            log.debug(f"self.current_observation.shape = {self.current_observation.shape}")
            demand = self.current_observation[ontario_demand_index]
            supply = self.current_observation[supply_index]
            new_price = self.current_observation[price_index]

        """
        Acc to dataset minAllowed is 0, maxAllowed is arround 2.3k.
        Considering three cases:
        Price ∉ (minAllowed,maxAllowed)                 : Heavily Punished by High Correction Value
        Demand - Supply or Price Negative               : Loss, thus punished, (Heavily if is out of domain)
        Demand - Supply, Price Positive and in domain   : Profit, Rewarded
        Decreaseing the overall Reward value: Correction/=(10**8)
        """
        
        # TODO Make Reward Better
        maxAllowed = self.min_max_values["max"][price_index]
        minAllowed = self.min_max_values["min"][price_index]

        correction = 1
        if ((demand-supply) <0) or (new_price<minAllowed) or (new_price>maxAllowed):
            correction=0-abs(correction)

        if denormalize:
            correction/=(10**17)
        reward = (abs(demand - supply)**3) * (abs(new_price)**2) * correction
        
        # Included profits generated in state set (for plotting graph)
        profit = (demand - supply)*new_price
        if LOG:
            log.info(f"State set = {new_price}, {correction}, {demand}, {supply}, {profit}")
        return reward

    def denormalize(self, arr):
        """
        Take any numpy array of 13 elements and de-normalize it, that is, undo
        the normalization done to the data, and then return it.
        """
        array=np.random.rand(*arr.shape)
        for feature in range(arr.shape[0]):
            minv = self.min_max_values["min"][feature]
            maxv = self.min_max_values["max"][feature]
            value = arr[feature]
            array[feature] = (value * (maxv - minv)) + minv
        return array

    def normalize(self, arr):
        """
        Take any numpy array of 13 elements and normalize it according to
        pre-defined values.
        """
        array=np.random.rand(*arr.shape)
        for feature in range(arr.shape[0]):
            minv = self.min_max_values["min"][feature]
            maxv = self.min_max_values["max"][feature]
            value = arr[feature]
            array[feature] = (value - minv)/(maxv - minv)
        return array
