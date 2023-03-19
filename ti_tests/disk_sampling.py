import numpy as np
import matplotlib.pyplot as plt

pi_4 = np.pi / 4
pi_2 = np.pi / 2

def concentric_disk_sample():
    offset = 2 * np.random.rand(2) - 1.
    if (offset == 0).any():
        return np.zeros(2)
    if abs(offset[0]) > abs(offset[1]):
        r = offset[0]
        theta = pi_4 * (offset[1] / offset[0])
    else:
        r = offset[1]
        theta = pi_2 - pi_4 * (offset[0] / offset[1])
    return r * np.float32([np.cos(theta), np.sin(theta)])

if __name__ == '__main__':
    samples = []
    for i in range(1000):
        sample = concentric_disk_sample()
        samples.append(sample)
    samples = np.stack(samples, axis = 0)
    circle = np.linspace(0, 2. * np.pi, 3600)
    circle_line = np.stack((np.cos(circle), np.sin(circle)), axis = 1)
    plt.plot(circle_line[:, 0], circle_line[:, 1], c = 'b')
    plt.scatter(samples[:, 0], samples[:, 1], s = 5, c = 'r')
    plt.grid(axis = 'both')
    plt.show()
