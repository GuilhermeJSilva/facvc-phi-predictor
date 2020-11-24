import pytest 
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
    
