import matplotlib.pyplot as plt
import phipredictor.datagen

def visualizeMatrix(matrix, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest')
    fig.colorbar(cax)
    fig.savefig(filename)

gen = phipredictor.datagen.PhiGen(mirror_size=(256, 256))
measurement, poses = gen.generateSample()
visualizeMatrix(measurement, "test.png")
