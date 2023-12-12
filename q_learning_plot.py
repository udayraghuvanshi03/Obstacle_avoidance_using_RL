import numpy as np
import matplotlib.pyplot as plt

def plot_q(eps_list,rew_list):
    plt.plot(eps_list)
    plt.xlabel('Training frames')
    plt.title('Q-learning')
    plt.ylabel('Episodes')
    plt.show()

    plt.plot(rew_list)
    plt.xlabel('Number of episodes')
    plt.title('Q-learning')
    plt.ylabel('Rewards')
    plt.show()

if __name__=='__main__':
    eps_list= np.load('epi_list.npy')
    rew_list=np.load('rew_list.npy')
    plot_q(eps_list,rew_list)