import pytest
import numpy as np
import phipredictor.datagen


def testDatagenSize():
    """
    Should return a tuple with a matrix of the correct size
    and a vector of size 12
    """
    gen = phipredictor.datagen.PhiGen(out_size=1024)

    measurement, pose = gen.generateSample()
    assert measurement.shape == (1024, 1024)
    assert pose.shape == (12, )


def testDatagenAddMirror():
    """
    Should change only the values corresponding to the mirror
    """

    size = 512
    gen = phipredictor.datagen.PhiGen(out_size=size)
    measurement = np.ndarray((size, size))

    for start_x, start_y in gen.mirror_pos:
        gen._applyMirror(measurement, (start_x, start_y), 1, 0, 0)
        value_in_mirror = np.sum(measurement[start_x:gen.mirror_size_w +
                                             start_x, start_y:gen.mirror_size_h + start_y])
        assert value_in_mirror == gen.mirror_size_h * gen.mirror_size_w
        assert int(np.sum(measurement)) == np.count_nonzero(measurement > 1e-300)

