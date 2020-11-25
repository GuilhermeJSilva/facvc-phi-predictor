import matplotlib.pyplot as plt
import numpy as np
import phipredictor.datagen

def visualizeMatrix(matrix, filename):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap=plt.get_cmap("cividis"))
    fig.colorbar(cax)
    fig.savefig(filename)

gen = phipredictor.datagen.PhiGen()
measurement, poses = gen.generateSample()
visualizeMatrix(np.log(measurement), "test.png")
