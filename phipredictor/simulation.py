import numpy as np
from typing import List, Tuple
import phipredictor.visualization as vis


def toVector(matrix: np.ndarray):
    return np.reshape(matrix, (-1,))


def normalizeTilt(size_segment: int):
    linspace = np.expand_dims(np.linspace(-1, 1, num=size_segment), 0)
    tilt = linspace.T @ np.ones((1, size_segment))
    return tilt / np.sqrt(np.sum(np.square(tilt)) / (size_segment ** 2))


def insertMatrix(target: np.ndarray, x_start: int, y_start: int, origin: np.ndarray):
    size_x, size_y = origin.shape
    target[x_start : x_start + size_x, y_start : y_start + size_y] = origin


class PhaseSimulator:
    def __init__(self, size_segment=128, crop_size=200):
        self.wvl = 850e-9
        self.size_segment = size_segment
        self.size_surface = 4 * self.size_segment
        self.nyquistSampleFactor = 4
        self.crop_size = crop_size

        self.piston = np.ones((self.size_segment, self.size_segment))
        self.tilt = normalizeTilt(self.size_segment)
        self.tip = np.transpose(self.tilt)
        self.modes = np.array(
            [toVector(self.piston), toVector(self.tip), toVector(self.tilt)]
        ).T

        self.mirror_positions = self._genMirrorPositions()
        self.amplitude = self._amplitude()

    def _addSegments(self, sample: np.ndarray, mirror_poses: np.ndarray) -> np.ndarray:
        for i, (x_start, y_start) in enumerate(self.mirror_positions):
            phase_microns = np.matmul(self.modes, mirror_poses[:, i])
            phase_microns = np.reshape(
                phase_microns, (self.size_segment, self.size_segment)
            )
            insertMatrix(sample, x_start, y_start, phase_microns)

    def _genMirrorPositions(self) -> List[Tuple[int, int]]:
        """Calculates the position of the right upper corner
        of each mirror

        Returns:
            List[Tuple[int, int]]: List of right corner positions
        """
        middle_pos = self.size_surface / 2 + 1

        return [
            (
                int(middle_pos - 3 * self.size_segment / 2),
                int(middle_pos - self.size_segment / 2),
            ),
            (
                int(middle_pos - self.size_segment / 2),
                int(middle_pos - 3 * self.size_segment / 2),
            ),
            (
                int(middle_pos + self.size_segment / 2),
                int(middle_pos - self.size_segment / 2),
            ),
            (
                int(middle_pos - self.size_segment / 2),
                int(middle_pos + self.size_segment / 2),
            ),
        ]

    def _amplitude(self) -> np.ndarray:
        amplitude = np.zeros((self.size_surface, self.size_surface))
        for x_start, y_start in self.mirror_positions:
            amplitude[
                x_start : self.size_segment + x_start,
                y_start : self.size_segment + y_start,
            ] = 1

        return amplitude

    def _electricFieldPupil(self, phase: np.ndarray) -> np.ndarray:
        return np.dot(self.amplitude, np.exp(2 * np.pi * 1j / self.wvl * phase * 1e-6))

    def _electricFieldFocal(self, e_field_pupil: np.ndarray) -> np.ndarray:
        sample_size = self.size_surface * 2 * self.nyquistSampleFactor
        e_field_focal = np.fft.fftshift(e_field_pupil)
        e_field_focal = np.fft.fft2(e_field_focal, s=(sample_size, sample_size))
        return np.fft.fftshift(e_field_focal)

    def _cropCenter(self, matrix: np.ndarray, size: int) -> np.ndarray:
        center = int(matrix.shape[0] / 2 + 1)
        half = int(size / 2)
        return matrix[center - half : center + half, center - half : center + half]

    def _removeSymmetry(self, matrix: np.ndarray):
        size_of_removal = 4
        center_x, center_y = self.mirror_positions[2]
        start_x = int(center_x + self.size_segment - size_of_removal)
        start_y = int(center_y + self.size_segment - size_of_removal)
        matrix[
            start_x : start_x + size_of_removal, start_y : start_y + size_of_removal
        ] = 0

    def simulate(
        self,
        mirror_pose: np.ndarray,
        noise: bool,
        symmetry: bool,
        save_step: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert mirror_pose.shape == (3, 4)
        sample = np.zeros((self.size_surface, self.size_surface))
        self._addSegments(sample, mirror_pose)
        if save_step is not None:
            vis.visualizeMatrix(sample, save_step)
        sample = self._electricFieldPupil(sample)
        if not symmetry:
            self._removeSymmetry(sample)
        sample = self._electricFieldFocal(sample)
        sample = np.square(np.abs(sample))
        if noise:
            sample = np.random.poisson(sample)
        sample = self._cropCenter(sample, self.crop_size)

        return sample


if __name__ == "__main__":
    import os
    import pandas as pd

    coef = np.array([[0, 0, 0, 0], [1, 0, -1, 0], [0, -1, 0, 1]]) / 10
    # coef = np.array([[0, 0, 0, 0], [0, 1, 0, -1], [1, 0, -1, 0]]) / 10
    simulator = PhaseSimulator()

    columns = ["filename"] + [
        part + "_" + str(i) for i in range(1, 5) for part in ["piston", "tilt", "tip"]
    ]
    folder_path = "data/same_out"
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(folder_path + "/samples")
    df = pd.DataFrame(columns=columns)
    for j, sym in enumerate([True, False]):
        for i, poses in enumerate([coef, -coef]):
            samples = simulator.simulate(poses, False, sym)
            filename = str(i + j) + ".npy"
            np.save(folder_path + "/samples/" + filename, samples)
            l_poses = [filename] + list(poses.flatten())
            df.loc[len(df.index)] = l_poses
    df.to_csv(folder_path + "/data.csv")