
import numpy as np
import timeit
from flat_game import q_learning_carmunk
from collections import defaultdict

def argmax(arr):
    max_val = max(arr)
    max_ind = []
    for index, val in enumerate(arr):
        if val == max_val:
            max_ind.append(index)

    return np.random.choice(max_ind)

def create_int_defaultdict():
    return defaultdict(int)

def q_learning(
        num_steps: int,
        gamma: float,
        epsilon: float,
        step_size: float,
):
    game_state=q_learning_carmunk.GameState(random_env=False)

    Q = defaultdict(lambda: np.zeros(3))
    max_car_distance = 0
    car_distance = 0
    t = 0
    data_collect = []
    eps_list = []
    eps=0
    #Initial state
    _, state = game_state.frame_step((2))
    rew_list=[]
    ts=0
    total_rew=0
    while ts<num_steps:
        ts += 1
        car_distance += 1
        if np.random.random() < epsilon:
            action = np.random.randint(0, 3)
        else:
            action = argmax(Q[state])

        reward,next_state = game_state.frame_step(action)
        total_rew+=reward
        max_q = max(Q[next_state])
        Q[state][action] += step_size * (reward + gamma * max_q - Q[state][action])
        state = next_state
        if reward==-500:
            eps += 1
            rew_list.append(total_rew)
            data_collect.append([t, car_distance])
            if car_distance > max_car_distance:
                max_car_distance = car_distance
            car_distance = 0
            total_rew=0

        if (eps+1)%100==0:
            print(f'Steps:{ts}, Episodes:{eps}')
        eps_list.append(eps)

    return Q, eps_list,max_car_distance,rew_list

def play(Q):

    car_distance = 0
    game_state = q_learning_carmunk.GameState(random_env=False)

    _, state = game_state.frame_step((2))

    while True:
        car_distance += 1
        action = (np.argmax(Q))

        _, state = game_state.frame_step(action)

        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)


if __name__=='__main__':
    q_100000,eps_list,max_dist,rew_list=q_learning(num_steps=100,gamma=1.0,epsilon=0.1,step_size=0.5)
    # np.save('epi_list.npy', eps_list)
    # np.save('rew_list.npy',rew_list)
    play(q_100000)

