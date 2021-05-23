import numpy as np
import log 
import torch
from torch import nn, Tensor
#import gym
from matplotlib import pyplot as plt
from tqdm import tqdm
from env import LSTMEnv as ENV
from lstm import LSTM

def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y


#import grid
# env = gym.make("LSTM_Grid-v0")
# env = gym.make("CartPole-v0")

# Certain LSTM env params
output_size = 13
input_dim = output_size
hidden_dim = 40
layer_dim = 2
Debug = False
actions = 5
actionSpaceIndices = list(range(0,actions))

n=int(actions//2)
if actions%2==0:
    actionSpace = [i for i in range(-n*5,n*5+1,5)]
    actionSpace.pop(len(actionSpace)//2)
else:
    actionSpace = [i/10 for i in range(-n*55,n*55+1,55)]

# print(f'ActionSpace = {actionSpace}')
#actionSpace=[-15,-10,0,+10,+15]
#actionSpace=[0,1,2,3,4,5,6,7,8,9]

# Dataset path
envDATA="../datasets/normalized_weird_13_columns_with_supply.csv"

# LSTM model
LSTM_instance = LSTM(output_size, input_dim, hidden_dim, layer_dim,debug=Debug)
LSTM_instance.loadM("ENV_MODEL/lstm_modelV3.pt")

# LSTM env 
env = ENV(
    model=LSTM_instance,
    dataset_path=envDATA,
    actionSpace=actionSpace,
    debug=Debug,
)

# [0,1,2,3,4]
# l1 = 3
# l2 = 150
# l3 = 10

l1 = input_dim
l2 = 150
l3 = actions

model = torch.nn.Sequential(
    # torch.nn.Linear(3, l1),
    torch.nn.Linear(l1, l2),
    # torch.nn.LeakyReLU(),
    torch.nn.Tanh(),
    torch.nn.Linear(l2, l3),
    torch.nn.Softmax(dim=0)
)

learning_rate = 0.009
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
state1 = env.reset()
pred = model(torch.from_numpy(state1).float())
action = np.random.choice(np.array(actionSpaceIndices), p=pred.data.numpy().flatten())
state2, reward, done, info = env.step(action,train=True)

def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma,torch.arange(lenr).float()) * rewards
    disc_return /= disc_return.max()
    return disc_return

def loss_fn(preds, r):
    return -1 * torch.sum(r * torch.log(preds))


MAX_DUR = 20
#MAX_EPISODES = 10
MAX_EPISODES = 4000
gamma = 0.3
score = []
expectation = 0.0

# TRAIN-----------------------------------------------------------------------
print('TRAINING started')
for episode in tqdm(range(MAX_EPISODES)):
    curr_state = env.reset()
    done = False
    transitions = []
    
    reward_batch = []
    for t in range(MAX_DUR):
        # act_prob = torch.tensor([np.nan])
        # while any(torch.isnan(act_prob)):
        act_prob = model(torch.from_numpy(curr_state).float().flatten())
        # print(f"act_prob = {act_prob}")
        action = np.random.choice(np.array(actionSpaceIndices), p=act_prob.data.numpy().flatten())
        prev_state = curr_state
        curr_state, reward, done, info = env.step(action,train=True)
        transitions.append((prev_state, action, reward))
        reward_batch.append(reward)


    ep_len = len(transitions)
    score.append(sum(reward_batch))
    reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))
    disc_returns = discount_rewards(reward_batch)
    # print(transitions)
    state_batch = torch.Tensor([s for (s,a,r) in transitions])
    action_batch = torch.Tensor([a for (s,a,r) in transitions])
    pred_batch = model(state_batch)
    prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze()
    loss = loss_fn(prob_batch, disc_returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

score = np.array(score)
#avg_score = running_mean(score, 2)
avg_score = running_mean(score, 50)
print("Training ends")



plt.figure(figsize=(10,7))
plt.ylabel(f"Reward Accrued in {MAX_DUR} timesteps",fontsize=22)
plt.xlabel("Training Episodes",fontsize=22)
plt.plot(avg_score, color='green')
plt.savefig("output.png")
plt.clf()

score = []
#games = 10
games = 200
done = False
state1 = env.reset()

# TESTING------------------------------------------------------------
print('Testing started')
for i in tqdm(range(games)):
    t=0
    reward_batch = []
    while not done:
        pred = model(torch.from_numpy(state1).float())
        action = np.random.choice(np.array(actionSpaceIndices), p=pred.data.numpy())
        state2, reward, done, info = env.step(action)
        state1 = state2 
        t += 1
        reward_batch.append(reward)
        if t > MAX_DUR:
            break;
    state1 = env.reset()
    done = False
    score.append(sum(reward_batch)/len(reward_batch))
    log.info(f"rewards = {Tensor(reward_batch)}")

score = np.array(score)
print('Testing ends')

plt.scatter(np.arange(score.shape[0]),score)
plt.savefig("output2.png")
