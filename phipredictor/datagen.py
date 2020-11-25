import math
import numpy as np
from typing import Tuple, List, Generator


def randomNormal(params: Tuple[float, float]) -> Generator[float, None, None]:
    """Generator which returns random variables in
    a normal distribution

    Args:
        params (Tuple[float, float]): Tuple consisting of
            min and variance

    Yields:
        Generator[float, None, None]: Generator for a random
            variable in a normal distribution
    """
    mean, var = params
    while True:
        yield np.random.normal(mean, var)


class PhiGen(object):
    """
    Generates measurements based on random mirror poses.
    """

    def __init__(self, out_size: int = 1024,
                 mirror_size: Tuple[int, int] = (128, 128),
                 piston: Tuple[float, float] = (0.0, 1.0),
                 tilt: Tuple[float, float] = (0.0, 1.0),
                 tip: Tuple[float, float] = (0.0, 1.0)):
        """

        Args:
            out_size (int, optional): Size of the measurement matrix.
                Defaults to 1024.
            mirror_size (Tuple[int, int], optional): Dimensions of
                each mirror. Defaults to (128, 128).
            piston (Tuple[float, float], optional): Mean and variance
                of the distribution of the piston position. Defaults
                to (0.0, 1.0).
            tilt (Tuple[float, float], optional): Mean and variance
                of the distribution of the piston position. Defaults
                to (0.0, 1.0).
            tip (Tuple[float, float], optional): Mean and variance
                of the distribution of the piston position. Defaults
                to (0.0, 1.0).
        """
        self.out_size = out_size
        self.mirror_size_w, self.mirror_size_h = mirror_size
        self.piston_gen = randomNormal(piston)
        self.tip_gen = randomNormal(tip)
        self.tilt_gen = randomNormal(tilt)
        self.mirror_pos = self._genMirrorPositions()

    def _genMirrorPositions(self) -> List[Tuple[int, int]]:
        """Calculates the position of the right upper corner
        of each mirror

        Returns:
            List[Tuple[int, int]]: List of right corner positions
        """
        middle_pos = self.out_size / 2

        return [(int(middle_pos - self.mirror_size_w / 2),
                 int(middle_pos - 3 * self.mirror_size_h / 2)),
                (int(middle_pos - 3 * self.mirror_size_w / 2),
                 int(middle_pos - self.mirror_size_h / 2)),
                (int(middle_pos + self.mirror_size_w / 2),
                 int(middle_pos - self.mirror_size_h / 2)),
                (int(middle_pos - self.mirror_size_w / 2),
                 int(middle_pos + self.mirror_size_h / 2))]

    def _getRandomPose(self) -> Tuple[float, float, float]:
        """Return a random pose for a mirror

        Returns:
            Tuple[float, float, float]: Piston, tip, tilt
        """
        return next(self.piston_gen), next(self.tip_gen), next(self.tilt_gen)

    def _applyMirror(self,
                     measurement: np.ndarray,
                     starting_pos: Tuple[int, int],
                     piston: float,
                     tip: float,
                     tilt: float) -> None:
        start_x, start_y = starting_pos
        x_multiplier = math.sin(tip) * math.cos(tilt)
        y_multiplier = math.sin(tilt)
        denominator = math.cos(tip) * math.cos(tilt)
        for x in range(self.mirror_size_w):
            x_offset = x + start_x
            for y in range(self.mirror_size_h):
                y_offset = y + start_y
                real_x = x - self.mirror_size_w/2
                real_y = x - self.mirror_size_h/2
                measurement[x_offset, y_offset] = (
                    piston - x_multiplier * real_x + y_multiplier*real_y)/denominator

    def _addMirror(self, measuremt: np.ndarray, starting_pos: Tuple[int, int] = (0, 0)) -> np.array:
        """Adds the phase of the mirror to the measurement

        Args:
            measuremt (np.ndarray): matrix where the phase is going
                to be changed
            starting_pos (Tuple[int, int], optional): position of
                the upper left corner of the mirror. Defaults to (0, 0).

        Returns:
            np.array: array containing the pose of the mirror, which
            is randomly generated
        """
        piston, tip, tilt = self._getRandomPose()
        self._applyMirror(measuremt, starting_pos, piston, tip, tilt)
        return np.array([piston, tip, tilt])

    def generateSample(self) -> Tuple[np.ndarray, np.array]:
        """Generates a single sample from a random mirror pose

        Returns:
            Tuple[np.ndarray, np.array]: A tuple where the first element
                is a matrix of size out_size x out_size and corresponds to
                the simulation of the sensor's measurement and the second
                element is an array with the poses of the 4 mirrors in order,
                i.e. [pose1, tip1, tilt1, pose2, tip2, tilt2,...]
        """
        measurement = np.zeros((self.out_size, self.out_size))

        mirror_poses = np.array([])
        for start_x, start_y in self.mirror_pos:
            mirror_poses = np.append(mirror_poses,
                                     self._addMirror(measurement, starting_pos=(start_x, start_y)))

        return measurement, mirror_poses
