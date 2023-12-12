"""
Once a model is learned, use this to play it.
"""

import car_env
import numpy as np
from nn import neural_net

NUM_SENSORS = 3


def play(model):
    car_distance = 0
    game_state = carmunk.GameState(random_env=True)

    _, state = game_state.frame_step((2))

    while True:
        car_distance += 1

        action = (np.argmax(model.predict(state, batch_size=1)))
        _, state = game_state.frame_step(action)
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)

if __name__ == "__main__":
    saved_model = 'saved-models/128-128-64-50000-36000.h5'
    model = neural_net(NUM_SENSORS, [128, 128], saved_model)
    play(model)
