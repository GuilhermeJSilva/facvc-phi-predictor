import pandas
import matplotlib.pyplot as plt


def graphTrainError(filename):
    df = pandas.read_csv(filename)

    plt.plot("Steps", "Value", data=df[df["Validation"] == 0], label="Training Set")
    plt.plot(
        "Steps",
        "Value",
        data=df[(df["Validation"] == 1) & (df["Value"] < 1)],
        label="Validation Set",
    )
    plt.legend()
    plt.show()
    plt.clf()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    graphTrainError(args.filename)