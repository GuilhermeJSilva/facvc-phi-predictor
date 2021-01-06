import matplotlib.pyplot as plt
import numpy as np


def visualizeMatrix(matrix, filename):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap=plt.get_cmap("cividis"))
    fig.colorbar(cax)
    fig.savefig(filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="file to visualize")
    parser.add_argument("out_file", nargs="?", default="output.png")
    args = parser.parse_args()
    data = np.load(args.in_file)
    visualizeMatrix(data, args.out_file) 
