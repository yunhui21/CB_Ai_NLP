# Da
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def extrast(token_count, target, windows_size ):
    start = max(target - windows_size, 0)
    end = min(target + windows_size + 1, token_count)
    return [tokens[i] for i in range(start, end) if i != target]
