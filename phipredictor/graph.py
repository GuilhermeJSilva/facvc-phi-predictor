import pandas
import matplotlib.pyplot as plt


def graphTrainError(filename: str, out_file: str):
    df = pandas.read_csv(filename)

    plt.plot("Steps", "Value", data=df[df["Validation"] == 0], label="Training Set")
    plt.plot(
        "Steps",
        "Value",
        data=df[(df["Validation"] == 1) & (df["Value"] < 1)],
        label="Validation Set",
    )
    plt.xlabel("Number of Training Steps")
    plt.legend()
    plt.savefig(out_file)
    plt.clf()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("out_file", type=str)
    args = parser.parse_args()
    graphTrainError(**vars(args))