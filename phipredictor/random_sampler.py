import os
import numpy as np
from phipredictor.simulation import PhaseSimulator
from typing import Tuple


class RandomSampler:
    def __init__(self, simulator: PhaseSimulator):
        self.simulator = simulator

    def genSamples(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        samples = []
        mirrors = []

        for _ in range(n):
            random_sample = np.random.normal(size=(4, 3)) / 10
            sample, mirror_poses = self.simulator.simulate(random_sample)
            samples.append(sample)
            mirrors.append(mirror_poses)

        return np.stack(samples), np.stack(mirrors)

    def genToFiles(self, folder_path: str, n_per_file: int, n_files: int):
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(folder_path + "/samples")
        os.makedirs(folder_path + "/poses")
        for i in range(n_files):
            samples, poses = self.genSamples(n_per_file)
            np.save(folder_path + "/samples/part" + str(i) + ".npy", samples)
            np.save(folder_path + "/poses/part" + str(i) + ".npy", poses)


if __name__ == "__main__":
    simulator = PhaseSimulator()
    sampler = RandomSampler(simulator)
    sampler.genToFiles("data/set_1", 3, 3)
