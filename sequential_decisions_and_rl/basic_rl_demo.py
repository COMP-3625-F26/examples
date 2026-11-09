# from slide #10 in deck #9

from environments import ThreeStateMDP
import pandas as pd
import numpy as np


mdp = ThreeStateMDP()
state, _ = mdp.reset()
actions = mdp.get_actions(state)

Q = pd.DataFrame()
gamma = 0.9
alpha = 0.1


for step in range(3000):
    action = np.random.choice(actions)
    new_state, reward, _, _, _ = mdp.step(action)

    if state not in Q.index:
        Q.loc[state, mdp.get_actions(state)] = 0
    if new_state not in Q.index:
        Q.loc[new_state, mdp.get_actions(new_state)] = 0

    # td_error ← r + 𝛾•maxa’(Q[s’, a’]) - Q[s, a]
    rpe = (reward + gamma * Q.loc[new_state].max()) - Q.loc[state, action]
    Q.loc[state, action] += alpha * rpe

    state = new_state

print(Q)
