import matplotlib.pyplot as plt
import numpy as np
import phipredictor.datagen

def visualizeMatrix(matrix, filename):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap=plt.get_cmap("cividis"))
    fig.colorbar(cax)
    fig.savefig(filename)

if __name__ == "__main__":
    gen = phipredictor.datagen.SampleGen()
    measurement, poses = gen.genSample()
    visualizeMatrix(measurement, "test.png")
