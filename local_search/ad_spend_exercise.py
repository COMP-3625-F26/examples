import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from optimization_test_functions import ad_campaign_profit


sample_input = np.array([0.9, 0.9, 1.0, 0.9])
print(ad_campaign_profit(sample_input))

def ob_fun_wrapper(input_value):
    return -ad_campaign_profit(input_value)

result = minimize(ob_fun_wrapper, x0=np.array([0.9, 0.9, 1.0, 0.9]), method='Nelder-Mead',
                  bounds = [(0, 1), (0, 1), (0, 1), (0, 1)])
print(result)