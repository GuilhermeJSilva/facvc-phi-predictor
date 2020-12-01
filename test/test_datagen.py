import pytest
import numpy as np
import phipredictor.datagen


def testDatagenSize():
    """
    Should return a tuple with a matrix of the correct size
    and a vector of size 12
    """
    gen = phipredictor.datagen.SampleGen(crop_size=200)

    measurement, pose = gen.genSample()
    assert measurement.shape == (200, 200)
    assert pose.shape == (4, 3)


def testDatagenAddMirror():
    """
    Should change only the values corresponding to the mirror
    """

    gen = phipredictor.datagen.SampleGen()
    measurement = np.ndarray((gen.size_surface, gen.size_surface))
    gen._addSegments(measurement)
    for start_x, start_y in gen.mirror_positions:
        mirror = measurement[start_x:gen.size_segment + start_x, start_y:gen.size_segment + start_y]
        value_in_mirror = np.sum(mirror != 0)
        assert value_in_mirror == gen.size_segment**2
