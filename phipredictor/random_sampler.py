import os
import numpy as np
import pandas as pd
from phipredictor.simulation import PhaseSimulator
from typing import Tuple


class RandomSampler:
    def __init__(self, simulator: PhaseSimulator):
        self.simulator = simulator
        self.columns = ["filename"] + [
            part + "_" + str(i)
            for i in range(1, 5)
            for part in ["piston", "tilt", "tip"]
        ]

    def genSample(self, noise: bool):
        random_sample = np.random.normal(size=(4, 3)) / 10
        return self.simulator.simulate(random_sample, noise), random_sample

    def genToFiles(self, folder_path: str, n: int, noise: bool = False):
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(folder_path + "/samples")
        df = pd.DataFrame(columns=self.columns)
        for i in range(n):
            samples, poses = self.genSample(noise)
            filename = str(i) + ".npy"
            np.save(folder_path + "/samples/" + filename, samples)
            l_poses = [filename] + list(poses.flatten())
            df.loc[len(df.index)] = l_poses
        df.to_csv(folder_path + "/data.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="folder destination for the files")
    parser.add_argument(
        "-n", help="number of examples to generate", dest="n", default=500
    )
    parser.add_argument(
        "-p",
        help="apply poisson noise to the generated samples",
        dest="noise",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    simulator = PhaseSimulator()
    sampler = RandomSampler(simulator)
    sampler.genToFiles(args.output_dir, args.n, args.noise)
