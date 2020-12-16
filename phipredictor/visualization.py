import matplotlib.pyplot as plt
import numpy as np
import phipredictor.simulation
import phipredictor.random_sampler

def visualizeMatrix(matrix, filename):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap=plt.get_cmap("cividis"))
    fig.colorbar(cax)
    fig.savefig(filename)

if __name__ == "__main__":
    sim = phipredictor.simulation.PhaseSimulator()
    gen = phipredictor.random_sampler.RandomSampler(sim)
    measurement, poses = gen.genSamples(1)
    visualizeMatrix(measurement[0], "test.png")
