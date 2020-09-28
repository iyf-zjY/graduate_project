from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    # x = np.linspace(-np.pi,np.pi,256,endpoint=True)
    # y = np.sin(x)
    # y_1 = np.sin(x+np.pi/4)
    # plt.plot(x,y)
    # plt.plot(x,y_1)
    # plt.show()
    with open("F:\msra_private\\access.ptxt","rb") as f:
        for line in f.readlines():
            print(str(line))