

import matplotlib.pyplot as plt
import time

def main():
    plt.ion()
    figure, axis = plt.subplots()
    x, y = [], []
    for i in range(100):
        x.append(i)
        y.append(i / 2)
        axis.clear()
        axis.scatter(x, y, 0.5)
        figure.canvas.draw()
        plt.pause(0.01)
        time.sleep(0.1)
    plt.show()

main()
