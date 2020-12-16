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


if __name__ == "__main__":
    simulator = PhaseSimulator()
    sampler = RandomSampler(simulator)
    samples, mirror_poses = sampler.genSamples(3)
    print(samples.shape)