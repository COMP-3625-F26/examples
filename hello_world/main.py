from environment import MiningGame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# instantiate the game
game = MiningGame(n_mines=2, render=False)
cumulative_reward = 0

values = np.array([1.0, 1.0])
value_history = np.zeros((500, 2))
alpha = 0.05

# repeatedly ask user to choose a place to mine, and execute their choice
for step in range(500):
    # choice = input("choose a place to mine (0 or 1). q to quit: ")
    # if choice == 'q':
        # break
    
    if np.random.rand() < 0.1:
        choice = np.random.choice([0, 1])
    else:
        choice = np.argmax(values)
    
    reward = game.choose_mine(int(choice))
    cumulative_reward += reward

    values[choice] = values[choice] + alpha * (reward - values[choice])
    value_history[step, :] = values
    print(values)

    print(f"reward: {reward}, cumulative reward: {cumulative_reward}")
    
# print the true reward probabilities for mines 0 and 1
print('The true reward probabilities for mine 0/1 were:', game.reward_probabilities)

plt.plot(value_history)
plt.show()