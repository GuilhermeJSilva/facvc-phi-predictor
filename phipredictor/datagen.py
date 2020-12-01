import numpy as np
from typing import List, Tuple


def toVector(matrix: np.ndarray):
    return np.reshape(matrix, (-1,))


def normalizeTilt(sizeSegment: int):
    linspace = np.expand_dims(np.linspace(-1, 1, num=sizeSegment), 1)
    tilt = linspace @ np.ones((1, sizeSegment))
    return tilt / np.sqrt(np.sum(np.square(tilt)) / sizeSegment**2)


def insertMatrix(target: np.ndarray, x_start: int, y_start: int, origin: np.ndarray):
    size_x, size_y = origin.shape
    target[x_start:x_start + size_x, y_start:y_start + size_y] = origin


class SampleGen:

    def __init__(self, sizeSegment=128):
        self.wvl = 850e-9
        self.sizeSegment = sizeSegment
        self.sizeSurface = 4*self.sizeSegment
        self.nyquistSampleFactor = 4

        self.piston = np.ones((self.sizeSegment, self.sizeSegment))
        self.tilt = normalizeTilt(self.sizeSegment)
        self.tip = self.tilt.T
        self.modes = np.array(
            [toVector(self.piston), toVector(self.tilt), toVector(self.tip)]).T

        self.mirror_positions = self._genMirrorPositions()
        self.amplitude = self._amplitude()

    def _addSegments(self, sample: np.ndarray) -> np.ndarray:
        mirror_poses = np.random.normal(size=(4, 3))
        for i, (x_start, y_start) in enumerate(self.mirror_positions):
            phase_microns = self.modes @ mirror_poses[i].T
            phase_microns = np.reshape(
                phase_microns, (self.sizeSegment, self.sizeSegment))
            insertMatrix(sample, x_start, y_start, phase_microns)
        return mirror_poses

    def _genMirrorPositions(self) -> List[Tuple[int, int]]:
        """Calculates the position of the right upper corner
        of each mirror

        Returns:
            List[Tuple[int, int]]: List of right corner positions
        """
        middle_pos = self.sizeSurface / 2 + 1

        return [
            (int(middle_pos - 3 * self.sizeSegment / 2),
                int(middle_pos - self.sizeSegment / 2)),
            (int(middle_pos - self.sizeSegment / 2),
                int(middle_pos - 3 * self.sizeSegment / 2)),
            (int(middle_pos + self.sizeSegment / 2),
                int(middle_pos - self.sizeSegment / 2)),
            (int(middle_pos - self.sizeSegment / 2),
                int(middle_pos + self.sizeSegment / 2))
        ]

    def _amplitude(self) -> np.ndarray:
        amplitude = np.zeros((self.sizeSurface, self.sizeSurface))
        for x_start, y_start in self.mirror_positions:
            amplitude[x_start: self.sizeSegment + x_start,
                      y_start: self.sizeSegment + y_start] = 1

        return amplitude

    def _electricFieldPupil(self, phase: np.ndarray) -> np.ndarray:
        return np.dot(self.amplitude, np.exp(2*np.pi*1j/self.wvl*phase*1e-6))

    def _electricFieldFocal(self, e_field_pupil: np.ndarray) -> np.ndarray:
        sample_size = self.sizeSurface*2*self.nyquistSampleFactor
        e_field_focal = np.fft.fftshift(e_field_pupil)
        e_field_focal = np.fft.fft2(
            e_field_focal, s=(sample_size, sample_size))
        return np.fft.fftshift(e_field_focal)

    def _cropCenter(self, matrix: np.ndarray, size: int) -> np.ndarray:
        center = int(matrix.shape[0]/2+1)
        return matrix[center - size: center+size, center - size: center + size]

    def genSample(self):
        sample = np.zeros((self.sizeSurface, self.sizeSurface))
        mirror_poses = self._addSegments(sample)
        sample = self._electricFieldPupil(sample)
        sample = self._electricFieldFocal(sample)
        sample = np.square(np.abs(sample))
        sample = self._cropCenter(sample, 100)

        return sample, mirror_poses
