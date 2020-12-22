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

    def genToFiles(self, folder_path: str, n: int):
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(folder_path + "/samples")
        df = pd.DataFrame(columns=self.columns)
        for i in range(n):
            random_sample = np.random.normal(size=(4, 3)) / 10
            samples, poses = self.simulator.simulate(random_sample)
            filename = str(i) + ".npy"
            np.save(folder_path + "/samples/" + filename, samples)
            l_poses = [filename] + list(poses.flatten())
            df.loc[len(df.index)] = l_poses
        df.to_csv(folder_path + "/data.csv")


if __name__ == "__main__":
    simulator = PhaseSimulator()
    sampler = RandomSampler(simulator)
    sampler.genToFiles("data/set_2", 500)
