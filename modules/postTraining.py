"""
Post processing to produce graphs from logs
Combinations:
    - [X] Reward
    - [X] Policy Loss
    - [X] Critic Loss
    - [X] Advantage
    - [X] Demand
    - [X] Supply
    - [X] Price
    - [X] Profit
#"""

__author__ = 'BlackDChase'
__version__ = '1.3.1'

# Imports
import os
import matplotlib.pyplot as plt
import sys
import numpy as np

def getAvg(array):
    arr = []
    for i in array:
        arr.append(float(i))
    return (sum(arr)/len(arr),len(arr))


def uniqueColor():
    """There're better ways to generate unique colors, but this isn't awful."""
    return plt.cm.gist_ncar(np.random.random())

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

def modelLoss(fileN):
    arr = []
    with open(fileN) as loss:
        for line in loss:
            l = line.strip()
            arr.append(float(l))
    return arr

def rewardAvgLen(data):
    avgReward = []
    rewardLen = []
    for avg,length in data:
        avgReward.append(avg)
        rewardLen.append(length)
    return avgReward,rewardLen


def getMostRecent(folder):
    path=os.path.dirname(os.path.realpath(""))
    pwd = path.split("/")[0]
    if pwd=='Saved_model':
        all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        latest_subdir = max(all_subdirs, key=os.path.getmtime)
    raise NotImplemented

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
    return np.array(price),np.array(corre),np.array(demand),np.array(supply),np.array(profit)

if __name__ == '__main__':
    #print(sys.argv)
    #print(os.path.dirname(os.path.realpath("")))
    #folderName = getMostRecent(sys.argv[1])
    folderName = ""
    #avgAdvantage,episodeLength = rewardAvgLen(rewardAvg(folderName+"advantageLog.tsv"))
    avgReward, episodeLength = rewardAvgLen(rewardAvg(folderName+"rewardLog.tsv"))
    policyLoss = modelLoss(folderName+"policyLossLog.tsv")
    #criticLoss = modelLoss(folderName+"criticLossLog.tsv")
    priceAvg,correAvg,demandAvg,supplyAvg,profitAvg = stateExtract(folderName+"stateLog.tsv",len(episodeLength)//4)
    price,corre,demand,supply,profit = stateExtract(folderName+"stateLog.tsv")
    demSupAvg = [-supplyAvg[i]+demandAvg[i] for i in range(len(demandAvg))]
    demSup = [-supply[i]+demand[i] for i in range(len(demand))]
    correAvg2 = []
    for i in episodeLength:
        correAvg2.append(sum(corre[:i])/i)
        corre=corre[i:]


    # Ploting Demand, Supply 
    fig,ax1 = plt.subplots(dpi=400)
    color='r'
    ax1.plot(demandAvg,color=color)
    ax1.tick_params(axis='y',labelcolor=color)
    ax1.set_ylabel('Demand',color=color)
    ax1.set_xlabel(f"Average of Per {len(episodeLength)//4} Episode")
    ax2 = ax1.twinx()
    color='b'
    ax2.plot(demSupAvg,color=color)
    ax2.tick_params(axis='y',labelcolor=color)
    ax2.set_ylabel('Demand-Supply',color=color)
    ax3 = ax1.twinx()
    color='g'
    ax3.plot(supplyAvg,color=color)
    ax3.tick_params(axis='y',labelcolor=color)
    ax3.set_ylabel('Supply',color=color)
    fig.tight_layout()
    plt.savefig(folderName+"Supply vs Demand.svg")
    plt.close()

    # Ploting AVG A3C Price vs Exchange (both have different y-axis scaling and plotted on different axes)
    fig,ax1 = plt.subplots(dpi=400)
    color='r'
    ax1.plot(priceAvg,color=color)
    ax1.tick_params(axis='y',labelcolor=color)
    ax1.set_ylabel('Model Price',color=color)
    ax1.set_xlabel(f"Average of Per {len(episodeLength)/4} Episode")
    ax2 = ax1.twinx()
    color='b'
    ax2.plot(demSupAvg,color=color)
    ax2.tick_params(axis='y',labelcolor=color)
    ax2.set_ylabel('Demand-Supply',color=color)
    fig.tight_layout()
    plt.savefig(folderName+"AVG Model Price vs Exchange.svg")
    plt.close()

    # # Plotting Avg A3C price vs (Demand - Supply) exchange vs Profits generated (all have same y-axis scaling and plotted on same axis)
    # fig,ax = plt.subplots(dpi=400)
    # fig.suptitle('A3C price vs Exchange vs Profits', fontsize=14)
    # ax.set_xlabel(f"Average of Per {len(episodeLength)/4} Episode")
    # color='g'
    # ax.plot(profitAvg,color=color,label='Model Profit')
    # color='r'
    # dataAvg=np.ones(len(profitAvg))*106272
    # ax.plot(dataAvg,color=color,label='Model Price')
    # ax.tick_params(axis='y',labelcolor=color)

    # ax1 = ax.twinx()
    # color='b'
    # ax1.plot(priceAvg,color=color)
    # ax1.tick_params(axis='y',labelcolor=color)
    # ax1.set_ylabel('Model Price',color=color)

    # color='y'
    # ax2 = ax.twinx()
    # ax2.plot(demSupAvg,color=color)
    # ax2.tick_params(axis='y',labelcolor=color)
    # ax2.set_ylabel('Demand-Supply',color=color)
    # fig.tight_layout()
    # plt.legend()
    # plt.savefig(folderName+"AVG Model Price vs Exchange vs Profit.svg")
    # plt.close()

    # Plotting Avg A3C price vs (Demand - Supply) exchange vs Profits generated (different y-axis scaling)
    # AXIS OVERLAPPING FIX NEEDED
    fig,ax = plt.subplots(dpi=400)
    fig.suptitle('A3C price vs Exchange vs Profits', fontsize=14)
    ax.set_xlabel(f"Average of Per {len(episodeLength)/4} Episode")
    ax.set_ylabel('Demand-Supply')
    color='b'
    ax.plot(demSupAvg,color=color,label='Demand-Supply')
    # color='r'
    # bareProfitMax=np.ones(len(profitAvg))*5860463
    # ax.plot(bareProfitMax,color=color,label='Max Profit (Dataset)')
    # color='y'
    # bareProfitStd=np.ones(len(profitAvg))*74348
    # ax.plot(bareProfitStd,color=color,label='STD Profit (Dataset)')
    ax.tick_params(axis='y',labelcolor=color)

    color='g'
    ax1 = ax.twinx()
    ax1.plot(priceAvg,color=color,label='Model Price')
    ax1.tick_params(axis='y',labelcolor=color)
    ax1.set_ylabel('Model Price',color=color)

    color='r'
    ax2 = ax.twinx()
    ax2.plot(profitAvg,color=color,label='Model Profit')
    color='m'
    bareProfitMean=np.ones(len(profitAvg))*106272
    ax2.plot(bareProfitMean,color=color,label='Mean Profit (Dataset)')
    ax2.tick_params(axis='y',labelcolor=color)
    ax2.set_ylabel('Profit',color=color)
    fig.tight_layout()
    plt.legend()
    plt.savefig(folderName+"AVG Model Price vs Exchange vs Profit_2.svg")
    plt.close()

    # Profits 
    fig,ax = plt.subplots(dpi=400)
    fig.suptitle('Profits Accumulated', fontsize=14)
    ax.set_xlabel(f"Average of Per {len(episodeLength)/4} Episode")
    ax.set_ylabel('Profit')
    color='b'
    ax.plot(profitAvg,color=color,label='Model Profit')
    ax2 = ax.twinx()
    color='g'
    bareProfitMean=np.ones(len(profitAvg))*106272
    ax2.plot(bareProfitMean,color=color,label='Mean Profit (Dataset)')
    color='g'
    bareProfitMax=np.ones(len(profitAvg))*5860463
    ax2.plot(bareProfitMax,color=color,label='Max Profit (Dataset)')
    color='g'
    bareProfitMin=np.ones(len(profitAvg))*0.19
    ax2.plot(bareProfitMin,color=color,label='Min Profit (Dataset)')
    # bareProfitStd=np.ones(len(profitAvg))*74348
    # ax2.plot(bareProfitStd,color=color,label='STD Profit (Dataset)')
    color='r'
    modelProfitMean=np.ones(len(profitAvg))*profitAvg.mean()
    ax2.plot(modelProfitMean,color=color,label='Mean Profit (Model)')
    color='r'
    modelProfitMax=np.ones(len(profitAvg))*profitAvg.max()
    ax2.plot(modelProfitMax,color=color,label='Max Profit (Model)')
    color='r'
    modelProfitMin=np.ones(len(profitAvg))*profitAvg.min()
    ax2.plot(modelProfitMin,color=color,label='Min Profit (Model)')
    # color='y'
    # bareProfitMedian=np.ones(len(profitAvg))*104574
    # ax2.plot(bareProfitMedian,color=color,label='Median Profit (Dataset)')
    ax2.tick_params(axis='y',labelcolor=color)
    fig.tight_layout()
    plt.legend()
    plt.savefig(folderName+"Profits.svg")
    plt.close()

    # Ploting average advantage
    # plt.figure(dpi=400)
    # plt.xlabel("Episode")
    # plt.ylabel("Average advantage")
    # plt.plot(avgAdvantage)
    # plt.savefig(folderName+"avgAdvantage.svg")
    # plt.close()

    # Ploting episodic Policy Loss
    plt.figure(dpi=400)
    plt.xlabel("Episode")
    plt.ylabel("Policy Loss")
    plt.plot(policyLoss)
    plt.savefig(folderName+"policyLoss.svg")
    plt.close()

    # Ploting episodic Critic Loss
    # plt.figure(dpi=400)
    # plt.xlabel("Episode")
    # plt.ylabel("criticLoss")
    # plt.plot(criticLoss)
    # plt.savefig(folderName+"criticLoss.svg")
    # plt.close()

    # Ploting average reward vs correction
    fig,ax1 = plt.subplots(dpi=400)
    color='r'
    ax1.plot(avgReward,color=color)
    ax1.tick_params(axis='y',labelcolor=color)
    ax1.set_ylabel('Average Reward',color=color)
    ax1.set_xlabel(f"Average of Per {len(episodeLength)/4} Episode")
    ax2 = ax1.twinx()
    color='b'
    ax2.plot(correAvg2,color=color)
    ax2.tick_params(axis='y',labelcolor=color)
    ax2.set_ylabel('Average Correction',color=color)
    fig.tight_layout()
    plt.savefig(folderName+"avgRewardnCorrection.svg")
    plt.close()

    print(f"Min of Profit Acquired: {profitAvg.min()}")
    print(f"Max of Profit Acquired: {profitAvg.max()}")
    print(f"Avg of Profit Acquired: {profitAvg.mean()}")
    print(f"STD of Profit Acquired: {profitAvg.std()}")

    #"""
    # REMOVE ---------------------------------------------------------------------------
    # Ploting A3C Price vs Exchange
    # plt.figure(dpi=400)
    # plt.xlabel(f"Episode")
    # plt.plot(demSup,label="Demand-Supply")
    # plt.plot(price,label="Model Price")
    # plt.legend()
    # plt.savefig(folderName+"Price VS Exchange.svg")
    # plt.close()
    #"""
