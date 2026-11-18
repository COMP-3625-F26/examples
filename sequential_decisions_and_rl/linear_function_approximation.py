import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
import pandas as pd


world = GridWorld(
    size=5,
    state_includes_location=False,
    state_includes_sensory=True,
    wall_probability=0.1,
)
features = world.reset()


# create a matrix of weights for each feature and action
weights = pd.DataFrame(index=['up', 'down', 'left', 'right'], 
                       columns=list(features.keys()) + [ 'bias'], 
                       data=np.random.random(size=(4, len(features)+1))
                       )

# learning parameters
gamma = 0.9
alpha = 0.1
epsilon = 0.25
