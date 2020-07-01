import matplotlib.pyplot as plt
import numpy as np

def f():
    x = np.array([1, 2, 3, 4, 5, 6])
    y = x*x

    fig, ax = plt.subplots()
    ax.plot(x, y)

    return fig