import pandas as pd
import numpy as np
from hyppo.ksample import MMD

chat_id = 683820405 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    alpha = 0.01
    return MMD(compute_kernel = "laplacian").test(x,y)[1] < alpha
