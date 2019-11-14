import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot(dicmAP, cfg):
    mAPSeries = pd.Series(dicmAP)
    mAPSeries.plot()
    plt.grid(True, linestyle = "--",color = "gray", linewidth = "0.5",axis = 'both')
    plt.legend(['mAP iteration'])
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "mAP.png"))
    #plt.show()

if __name__ == "__main__":
    dic = {1:50,2:30,3:90}
    plot(dic,cfg)
