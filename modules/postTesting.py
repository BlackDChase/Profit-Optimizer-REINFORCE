"""
Post processing to produce graphs from logs
- [X] A3C Profit
- [X] Normal Profit
- [X] A3C State
- [X] Normal State
- [X] Diff
#"""
__author__ = 'BlackDChase'
__version__ = '1.3.1'

# Imports

import os
import matplotlib.pyplot as plt
import sys
import numpy as np

def modelLoss(fileN):
    arr = []
    with open(fileN) as loss:
        for line in loss:
            l = line.strip()
            arr.append(float(l))
    return arr

def uniqueColor():
    """There're better ways to generate unique colors, but this isn't awful."""
    return plt.cm.gist_ncar(np.random.random())


def getMostRecent(folder):
    path=os.path.dirname(os.path.realpath(""))
    pwd = path.split("/")[0]
    if pwd=='Saved_model':
        all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        latest_subdir = max(all_subdirs, key=os.path.getmtime)
    raise NotImplemented


def readState(fileN):
    states = []
    with open(fileN) as state:
        for i in state:
            x = i.strip().split(",")
            while "" in x:
                x.remove('')
            states.append(list(map(float,x)))
    states = np.array(states,dtype=np.float32).transpose()
    return states

def readProfit(fileN):
    profit = []
    with open(fileN) as state:
        for i in state:
            profit.append(float(i.strip()))
    return np.array(profit)


def getAvg(array):
    arr = []
    for i in array:
        arr.append(float(i))
    return (sum(arr)/len(arr),len(arr))

def rewardAvg(fileN):
    arr = []
    with open(fileN) as reward:
        for line in reward:
            avgReward = line.strip().replace(' ','').split(",")
            while '' in avgReward:
                avgReward.remove('')
            if len(avgReward)>0:
                arr.append(getAvg(avgReward))
    return arr

def rewardAvgLen(data):
    avgReward = []
    rewardLen = []
    for avg,length in data:
        avgReward.append(avg)
        rewardLen.append(length)
    return avgReward,rewardLen

def stateExtract(fileN,order=None):
    with open(fileN) as state:
        price=[]
        corre=[]
        demand=[]
        supply=[]
        profit=[]
        temp=order
        x,y,z,w,p=0,0,0,0,0
        for i in state:
            sp = i.strip().split(",")
            x+=float(sp[0])
            y+=float(sp[1])
            z+=float(sp[2])
            w+=float(sp[3])
            p+=float(sp[4])
            if temp==0:
                temp=order
            if temp==order:
                if order!=None:
                    x/=order
                    y/=order
                    z/=order
                    w/=order
                    p/=order
                price.append(x)
                corre.append(y)
                demand.append(z)
                supply.append(w)
                profit.append(p)
                x,y,z,w,p=0,0,0,0,0
            if order!=None:
                temp-=1
    return price,corre,demand,supply,profit

def computeAvg(data):
    avg = 0
    for element in data:
        avg += element
    return avg/len(data)

def computeAvgChunks(data,chunkSize):
    avgChunks = [computeAvg(data[i * chunkSize:(i + 1) * chunkSize]) for i in range((len(data) + chunkSize - 1) // chunkSize )] 
    return avgChunks

if __name__ == '__main__':
    #print(sys.argv)
    #print(os.path.dirname(os.path.realpath("")))
    #folderName = getMostRecent(sys.argv[1])
    folderName = ""

    # TODO, Update this
    #a3cState = readState(folderName+"A3CState.tsv")
    #a3cProfit = readProfit(folderName+"A3CProfit.tsv")
    #meanProfit = a3cProfit.mean()
    #meanProfit = np.ones(shape=a3cProfit.shape)*meanProfit
    #policyLoss = modelLoss(folderName+"policyLossLog.tsv")
    #criticLoss = modelLoss(folderName+"criticLossLog.tsv")

    # Advantage logged in both online/offline inside their respective nStepAdvantage()
    #avgAdvantage,episodeLength = rewardAvgLen(rewardAvg(folderName+"advantageLog.tsv"))
    # Reward logged in online in size(trajectoryR)/ offline one at a time
    avgReward, episodeLength = rewardAvgLen(rewardAvg(folderName+"rewardLog.tsv"))
    # Extracting state attributes from state set logged in both online and offline internally within env after env.step() is called 
    priceAvg,correAvg,demandAvg,supplyAvg,profitAvg = stateExtract(folderName+"stateLog.tsv",len(episodeLength)//4)
    price,corre,demand,supply,profit = stateExtract(folderName+"stateLog.tsv")
    demSupAvg = [-supplyAvg[i]+demandAvg[i] for i in range(len(demandAvg))]
    demSup = [-supply[i]+demand[i] for i in range(len(demand))]

    fig,ax = plt.subplots(dpi=400)
    fig.suptitle('Demand-Supply vs Profit', fontsize=14)
    ax.set_xlabel(f"Time steps")
    ax.set_ylabel('Exchange (Demand-Supply)')
    color='b'
    ax.plot(demSup,color=color,label='Demand-Supply')
    # ax.tick_params(axis='y',labelcolor=color)
    ax2 = ax.twinx()
    # Based on data profit (original)
    color='g'
    ax2.plot(profit,color=color,label='Model Profit')
    ax2.set_ylabel('Profit')
    ax2.tick_params(axis='y',labelcolor=color)
    fig.tight_layout()
    plt.legend()
    plt.savefig(folderName+"Demand-Supply vs Profit.svg")
    plt.close()

    # offline=True
    # try:
    #     #normalState = readState(folderName+"NormalState.tsv")
    #     normalProfit = readProfit(folderName+"NormalProfit.tsv")
    #     meanNormalProfit = normalProfit.mean()
    #     mean = np.ones(shape=a3cProfit.shape)*meanNormalProfit
    #     diff = readProfit(folderName+"ProfitDiff.tsv")
    # except:
    #     offline=False
    #     pass

    # # Ploting Profit
    # fig,ax1 = plt.subplots(dpi=400)
    # color='r'
    # ax1.plot(a3cProfit,color=color)
    # ax1.tick_params(axis='y',labelcolor=color)
    # ax1.set_ylabel('A3C Profit',color=color)
    # ax1.set_xlabel(f"Time step")
    # ax2 = ax1.twinx()
    # color='green'
    # if offline:
    #     ax2.plot(normalProfit,color=color)
    #     ax2.plot(mean,color='y')
    #     ax2.tick_params(axis='y',labelcolor=color)
    #     ax2.set_ylabel('Profit w/o A3C',color=color)
    # else:
    #     mean = np.ones(shape=a3cProfit.shape)*106272
    #     mini = np.ones(shape=a3cProfit.shape)*0.19
    #     maxi = np.ones(shape=a3cProfit.shape)*5860463
    #     ax2.plot(mean,color=color,label="Dataset Mean")
    #     ax2.plot(mini,color=color,label="Dataset Min")
    #     ax2.plot(maxi,color=color,label="Dataset Max")
    #     ax2.tick_params(axis='y',labelcolor=color)
    #     ax2.set_ylabel('Original Dataset',color=color)
    # ax1.plot(meanProfit,color='k',label='A3C mean')
    # fig.tight_layout()
    # plt.savefig(folderName+"Profit.svg")
    # plt.close()

    # # Model Profit and Data Profits (STD not included)
    # fig,ax = plt.subplots(dpi=400)
    # fig.suptitle('Profits', fontsize=14)
    # ax.set_xlabel(f"Time steps")
    # ax.set_ylabel('Profit')
    # color='b'
    # ax.plot(a3cProfit,color=color,label='Model Profit')
    # ax2 = ax.twinx()
    # # Based on data profit (original)
    # color='g'
    # bareProfitMean=np.ones(len(profitAvg))*106272
    # ax2.plot(bareProfitMean,color=color,label='Mean Profit (Dataset)')
    # color='r'
    # bareProfitMax=np.ones(len(profitAvg))*5860463
    # ax2.plot(bareProfitMax,color=color,label='Max Profit (Dataset)')
    # color='m'
    # bareProfitMin=np.ones(len(profitAvg))*0.1
    # ax2.plot(bareProfitMin,color=color,label='Min Profit (Dataset)')
    # # Based on model profit
    # color='c'
    # modelProfitMean=np.ones(len(profitAvg))*a3cProfit.mean()
    # ax2.plot(modelProfitMean,color=color,label='Mean Profit (Model)')
    # color='y'
    # modelProfitMax=np.ones(len(profitAvg))*a3cProfit.max()
    # ax2.plot(modelProfitMax,color=color,label='Max Profit (Model)')
    # color='k'
    # modelProfitMin=np.ones(len(profitAvg))*a3cProfit.min()
    # ax2.plot(modelProfitMin,color=color,label='Min Profit (Model)')
    # ax2.tick_params(axis='y',labelcolor='r')
    # fig.tight_layout()
    # plt.legend()
    # plt.savefig(folderName+"Model Profit vs Data Profit.svg")
    # plt.close()

    # if offline:
    #     # Plotting Difference in Profit
    #     fig,ax = plt.subplots(dpi=100)
    #     ax.set_xlabel(f"Time step")
    #     ax.plot(diff)
    #     ax.set_ylabel(f"A3C Profit - Normal Profit")
    #     fig.tight_layout()
    #     plt.savefig(folderName+"Differnce in profit.svg")
    #     plt.close()

    # # Ploting episodic Policy Loss
    # plt.figure(dpi=400)
    # plt.xlabel("Episode")
    # plt.ylabel("Policy Loss")
    # plt.plot(policyLoss)
    # plt.savefig(folderName+"policyLoss.svg")
    # plt.close()

    # # Ploting episodic Critic Loss
    # plt.figure(dpi=400)
    # plt.xlabel("Episode")
    # plt.ylabel("criticLoss")
    # plt.plot(criticLoss)
    # plt.savefig(folderName+"criticLoss.svg")
    # plt.close()

    # # Rewards Accumulated (TESTING)
    # plt.figure(dpi=400)
    # plt.xlabel("Episode")
    # plt.ylabel("Avg Reward")
    # plt.plot(avgReward)
    # plt.savefig(folderName+"Avg_rewards.svg")
    # plt.close()

    # # Avg advantage
    # #episodeLength = 2000
    # #avgAdvantage = computeAvgChunks(advantage,episodeLength)
    # plt.figure(dpi=400)
    # plt.xlabel(f"Episode")
    # plt.ylabel("Advantage")
    # plt.plot(avgAdvantage)
    # plt.savefig(folderName+"Avg_Advantage.svg")
    # plt.close()

    # # Avg rewards vs correction (on seperate y axis scaling)
    # #episodeLength = 2000
    # #avgRewards = computeAvgChunks(rewards,episodeLength)
    # #avgCorrection = computeAvgChunks(correction,episodeLength)
    # fig,ax = plt.subplots(dpi=400)
    # fig.suptitle('Avg rewards vs correction', fontsize=14)
    # ax.set_xlabel(f"Average per {episodeLength//4} episodes")
    # ax.set_ylabel('Rewards')
    # color='b'
    # ax.plot(avgReward,color=color,label='Avg rewards')
    # ax2 = ax.twinx()
    # ax2.set_ylabel('Correction')
    # color='r'
    # ax2.plot(correAvg,color=color,label='Avg Correction')
    # ax2.tick_params(axis='y',labelcolor=color)
    # fig.tight_layout()
    # plt.legend()
    # plt.savefig(folderName+"Avg rewards vs correction.svg")
    # plt.close()

    # # Avg model price vs exchange vs profit (on seperate y axis scaling)
    # #episodeLength = 2000
    # #avgPrice = computeAvgChunks(price,episodeLength)
    # #avgExchange = computeAvgChunks(dem_Sup,episodeLength)
    # #avgProfit = computeAvgChunks(profit,episodeLength)
    # fig,ax = plt.subplots(dpi=400)
    # fig.suptitle('Avg model price vs exchange vs profit', fontsize=14)
    # ax.set_xlabel(f"Average per {episodeLength//4} episodes")
    # ax.set_ylabel('Demand-Supply')
    # color='b'
    # ax.plot(demSupAvg,color=color,label='Avg Exchange(demand-supply)')
    # ax2 = ax.twinx()
    # ax2.set_ylabel('Price')
    # color='r'
    # ax2.plot(priceAvg,color=color,label='Avg model price')
    # ax2.tick_params(axis='y',labelcolor=color)
    # color='g'
    # ax3 = ax.twinx()
    # ax3.set_ylabel('Profit')
    # ax3.plot(profitAvg,color=color,label='Avg profit')
    # ax3.tick_params(axis='y',labelcolor=color)
    # fig.tight_layout()
    # plt.legend()
    # plt.savefig(folderName+"Avg model price vs exchange vs profit.svg")
    # plt.close()

    # print(f"Min of Profit Acquired: {a3cProfit.min()}")
    # print(f"Max of Profit Acquired: {a3cProfit.max()}")
    # print(f"Avg of Profit Acquired: {a3cProfit.mean()}")
    # print(f"STD of Profit Acquired: {a3cProfit.std()}")

    """
    # Ploting States
    # Plotting A3C State
    states=len(normalState)
    square=int(np.ceil(states**(1/2)))
    fig,ax = plt.subplot(square,square,dpi=800)
    for i in range(states):
        lstm=2*i
        a3c=lstm+1

        x,y=lstm//square,lstm%square
        ax1=ax[x,y]
        ax1.set_title(f"State {i}")
        ax1.set_xlabel(f"Time step")

        color='r'
        ax1.plot(normalState[i],color=color)
        ax1.tick_params(axis='y',labelcolor=color)
        ax1.set_ylabel('Normal',color=color)

        ax2 = ax1.twinx()
        color='b'
        ax2.plot(normalProfit,color=color)
        ax2.tick_params(axis='y',labelcolor=color)
        ax2.set_ylabel('A3C',color=color)

    fig.tight_layout()
    plt.savefig(folderName+"States.svg")
    plt.close()
    """
