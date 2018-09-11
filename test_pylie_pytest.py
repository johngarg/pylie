import pylie
import numpy as np

su2 = pylie.LieAlgebra(pylie.CartanMatrix('SU', 2))
su3 = pylie.LieAlgebra(pylie.CartanMatrix('SU', 3))
su4 = pylie.LieAlgebra(pylie.CartanMatrix('SU', 4))
su5 = pylie.LieAlgebra(pylie.CartanMatrix('SU', 5))
su9 = pylie.LieAlgebra(pylie.CartanMatrix('SU', 9))
sp12 = pylie.LieAlgebra(pylie.CartanMatrix('SP', 12))

def test_weyl_orbit():
    assert su2._weylOrbit([0]) == [[0]]
    assert su3._weylOrbit([1,0]) == [[1, 0], [-1, 1], [0, -1]]
    assert su3._weylOrbit([1,1]) == [[1, 1], [-1, 2], [2, -1], [1, -2], [-2, 1], [-1, -1]]

def test_dominant_weights():
    assert np.all(su3._dominantWeights([1, 0])[0][0] == np.array([[1, 0]]))
    assert su3._dominantWeights([1, 0])[0][1] == 1

    assert np.all(su3._dominantWeights([1, 1])[0][0] == np.array([[1, 1]]))
    assert su3._dominantWeights([1, 1])[0][1] == 1
    assert np.all(su3._dominantWeights([1, 1])[1][0] == np.array([[0, 0]]))
    assert su3._dominantWeights([1, 1])[1][1] == 2

def test_casimir():
    su3_irreps = [[0, 0], [1, 0], [0, 1], [0, 2], [2, 0], [1, 1], [3, 0], [0, 3], [2, 1], [1, 2], [4, 0], [0, 4], [0, 5], [5, 0], [1, 3], [3, 1], [2, 2], [6, 0], [0, 6], [4, 1], [1, 4], [7, 0], [0, 7], [3, 2], [2, 3], [0, 8], [8, 0], [5, 1], [1, 5], [9, 0], [0, 9], [2, 4], [4, 2], [1, 6], [6, 1], [3, 3], [10, 0], [0, 10], [0, 11], [11, 0], [7, 1], [1, 7], [5, 2], [2, 5], [4, 3], [3, 4], [12, 0], [0, 12], [8, 1], [1, 8]]

    su2_casimir = [su2.casimir([n]) for n in range(1,101)]
    su3_casimir = [su3.casimir(el) for el in su3_irreps]
    assert su2_casimir == [3/4, 2, 15/4, 6, 35/4, 12, 63/4, 20, 99/4, 30, 143/4, 42, 195/4, 56, 255/4, 72, 323/4, 90, 399/4, 110, 483/4, 132, 575/4, 156, 675/4, 182, 783/4, 210, 899/4, 240, 1023/4, 272, 1155/4, 306, 1295/4, 342, 1443/4, 380, 1599/4, 420, 1763/4, 462, 1935/4, 506, 2115/4, 552, 2303/4, 600, 2499/4, 650, 2703/4, 702, 2915/4, 756, 3135/4, 812, 3363/4, 870, 3599/4, 930, 3843/4, 992, 4095/4, 1056, 4355/4, 1122, 4623/4, 1190, 4899/4, 1260, 5183/4, 1332, 5475/4, 1406, 5775/4, 1482, 6083/4, 1560, 6399/4, 1640, 6723/4, 1722, 7055/4, 1806, 7395/4, 1892, 7743/4, 1980, 8099/4, 2070, 8463/4, 2162, 8835/4, 2256, 9215/4, 2352, 9603/4, 2450, 9999/4, 2550]
    assert su3_casimir == [0, 4/3, 4/3, 10/3, 10/3, 3, 6, 6, 16/3, 16/3, 28/3, 28/3, 40/3, 40/3, 25/3, 25/3, 8, 18, 18, 12, 12, 70/3, 70/3, 34/3, 34/3, 88/3, 88/3, 49/3, 49/3, 36, 36, 46/3, 46/3, 64/3, 64/3, 15, 130/3, 130/3, 154/3, 154/3, 27, 27, 20, 20, 58/3, 58/3, 60, 60, 100/3, 100/3]
