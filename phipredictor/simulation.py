import numpy as np
from typing import List, Tuple


def toVector(matrix: np.ndarray):
    return np.reshape(matrix, (-1,))


def normalizeTilt(size_segment: int):
    linspace = np.expand_dims(np.linspace(-1, 1, num=size_segment), 1)
    tilt = linspace @ np.ones((1, size_segment))
    return tilt / np.sqrt(np.sum(np.square(tilt)) / size_segment**2)


def insertMatrix(target: np.ndarray, x_start: int, y_start: int, origin: np.ndarray):
    size_x, size_y = origin.shape
    target[x_start:x_start + size_x, y_start:y_start + size_y] = origin


class PhaseSimulator:

    def __init__(self, size_segment=128, crop_size=200):
        self.wvl = 850e-9
        self.size_segment = size_segment
        self.size_surface = 4*self.size_segment
        self.nyquistSampleFactor = 4
        self.crop_size = crop_size

        self.piston = np.ones((self.size_segment, self.size_segment))
        self.tilt = normalizeTilt(self.size_segment)
        self.tip = self.tilt.T
        self.modes = np.array(
            [toVector(self.piston), toVector(self.tilt), toVector(self.tip)]).T

        self.mirror_positions = self._genMirrorPositions()
        self.amplitude = self._amplitude()

    def _addSegments(self, sample: np.ndarray, mirror_poses: np.ndarray) -> np.ndarray:
        for i, (x_start, y_start) in enumerate(self.mirror_positions):
            phase_microns = self.modes @ mirror_poses[i].T
            phase_microns = np.reshape(
                phase_microns, (self.size_segment, self.size_segment))
            insertMatrix(sample, x_start, y_start, phase_microns)

    def _genMirrorPositions(self) -> List[Tuple[int, int]]:
        """Calculates the position of the right upper corner
        of each mirror

        Returns:
            List[Tuple[int, int]]: List of right corner positions
        """
        middle_pos = self.size_surface / 2 + 1

        return [
            (int(middle_pos - 3 * self.size_segment / 2),
                int(middle_pos - self.size_segment / 2)),
            (int(middle_pos - self.size_segment / 2),
                int(middle_pos - 3 * self.size_segment / 2)),
            (int(middle_pos + self.size_segment / 2),
                int(middle_pos - self.size_segment / 2)),
            (int(middle_pos - self.size_segment / 2),
                int(middle_pos + self.size_segment / 2))
        ]

    def _amplitude(self) -> np.ndarray:
        amplitude = np.zeros((self.size_surface, self.size_surface))
        for x_start, y_start in self.mirror_positions:
            amplitude[x_start: self.size_segment + x_start,
                      y_start: self.size_segment + y_start] = 1

        return amplitude

    def _electricFieldPupil(self, phase: np.ndarray) -> np.ndarray:
        return np.dot(self.amplitude, np.exp(2*np.pi*1j/self.wvl*phase*1e-6))

    def _electricFieldFocal(self, e_field_pupil: np.ndarray) -> np.ndarray:
        sample_size = self.size_surface*2*self.nyquistSampleFactor
        e_field_focal = np.fft.fftshift(e_field_pupil)
        e_field_focal = np.fft.fft2(
            e_field_focal, s=(sample_size, sample_size))
        return np.fft.fftshift(e_field_focal)

    def _cropCenter(self, matrix: np.ndarray, size: int) -> np.ndarray:
        center = int(matrix.shape[0]/2+1)
        half = int(size / 2)
        return matrix[center - half: center + half, center - half: center + half]

    def simulate(self, mirror_pose: np.ndarray, noise: bool) -> Tuple[np.ndarray, np.ndarray]:
        assert mirror_pose.shape == (4, 3)
        sample = np.zeros((self.size_surface, self.size_surface))
        self._addSegments(sample, mirror_pose)
        sample = self._electricFieldPupil(sample)
        sample = self._electricFieldFocal(sample)
        sample = np.square(np.abs(sample))
        if noise:
            random_poisson = np.random.poisson(sample)
            sample += random_poisson
        sample = self._cropCenter(sample, self.crop_size)

        return sample
