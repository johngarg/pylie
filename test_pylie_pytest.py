import pylie
import numpy as np
import sympy as sym

su2 = pylie.LieAlgebra(pylie.CartanMatrix('SU', 2))
su3 = pylie.LieAlgebra(pylie.CartanMatrix('SU', 3))
su4 = pylie.LieAlgebra(pylie.CartanMatrix('SU', 4))
su5 = pylie.LieAlgebra(pylie.CartanMatrix('SU', 5))
su9 = pylie.LieAlgebra(pylie.CartanMatrix('SU', 9))
so10 = pylie.LieAlgebra(pylie.CartanMatrix('SO', 10))
so11 = pylie.LieAlgebra(pylie.CartanMatrix('SO', 11))
sp10 = pylie.LieAlgebra(pylie.CartanMatrix('SP', 10))
sp12 = pylie.LieAlgebra(pylie.CartanMatrix('SP', 12))

su3_irreps = [[0, 0], [1, 0], [0, 1], [0, 2], [2, 0], [1, 1], [3, 0], [0, 3], [2, 1],
              [1, 2], [4, 0], [0, 4], [0, 5], [5, 0], [1, 3], [3, 1], [2, 2], [6, 0],
              [0, 6], [4, 1], [1, 4], [7, 0], [0, 7], [3, 2], [2, 3], [0, 8], [8, 0],
              [5, 1], [1, 5], [9, 0], [0, 9], [2, 4], [4, 2], [1, 6], [6, 1], [3, 3],
              [10, 0], [0, 10], [0, 11], [11, 0], [7, 1], [1, 7], [5, 2], [2, 5], [4, 3],
              [3, 4], [12, 0], [0, 12], [8, 1], [1, 8]]

su9_irreps = [[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1],
              [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [2, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 2], [1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0]]

def test_weyl_orbit():
    assert su2._weylOrbit([0]) == [[0]]
    assert su3._weylOrbit([1,0]) == [[1, 0], [-1, 1], [0, -1]]
    assert su3._weylOrbit([1,1]) == [[1, 1], [-1, 2], [2, -1], [1, -2], [-2, 1], [-1, -1]]

def test_dominant_weights():
    # [[np.array([[1, 0]]), 1]]
    assert np.all(su3._dominantWeights([1, 0])[0][0] == np.array([[1, 0]]))
    assert su3._dominantWeights([1, 0])[0][1] == 1

    # [[np.array([[1, 1]]), 1], [np.array([[0, 0]]), 2]]
    assert np.all(su3._dominantWeights([1, 1])[0][0] == np.array([[1, 1]]))
    assert su3._dominantWeights([1, 1])[0][1] == 1
    assert np.all(su3._dominantWeights([1, 1])[1][0] == np.array([[0, 0]]))
    assert su3._dominantWeights([1, 1])[1][1] == 2

def test_casimir():
    su2_casimirs = ['3/4', '2', '15/4', '6', '35/4', '12', '63/4', '20', '99/4', '30',
                    '143/4', '42', '195/4', '56', '255/4', '72', '323/4', '90', '399/4',
                    '110', '483/4', '132', '575/4', '156', '675/4', '182', '783/4', '210',
                    '899/4', '240', '1023/4', '272', '1155/4', '306', '1295/4', '342',
                    '1443/4', '380', '1599/4', '420', '1763/4', '462', '1935/4', '506',
                    '2115/4', '552', '2303/4', '600', '2499/4', '650', '2703/4', '702',
                    '2915/4', '756', '3135/4', '812', '3363/4', '870', '3599/4', '930',
                    '3843/4', '992', '4095/4', '1056', '4355/4', '1122', '4623/4', '1190',
                    '4899/4', '1260', '5183/4', '1332', '5475/4', '1406', '5775/4', '1482',
                    '6083/4', '1560', '6399/4', '1640', '6723/4', '1722', '7055/4', '1806',
                    '7395/4', '1892', '7743/4', '1980', '8099/4', '2070', '8463/4', '2162',
                    '8835/4', '2256', '9215/4', '2352', '9603/4', '2450', '9999/4', '2550']

    su3_casimirs = ['0', '4/3', '4/3', '10/3', '10/3', '3', '6', '6', '16/3', '16/3', '28/3',
                    '28/3', '40/3', '40/3', '25/3', '25/3', '8', '18', '18', '12', '12', '70/3',
                    '70/3', '34/3', '34/3', '88/3', '88/3', '49/3', '49/3', '36', '36', '46/3',
                    '46/3', '64/3', '64/3', '15', '130/3', '130/3', '154/3', '154/3', '27', '27',
                    '20', '20', '58/3', '58/3', '60', '60', '100/3', '100/3']

    su9_casimirs = ['0', '40/9', '40/9', '70/9', '70/9', '88/9', '88/9', '9', '10', '10']

    su2_casimir = [su2.casimir([n]) for n in range(1,101)]
    su3_casimir = [su3.casimir(el) for el in su3_irreps]
    su9_casimir = [su9.casimir(el) for el in su9_irreps]

    assert su2_casimir == [sym.Rational(i) for i in su2_casimirs]
    assert su3_casimir == [sym.Rational(i) for i in su3_casimirs]
    assert su9_casimir == [sym.Rational(i) for i in su9_casimirs]

def test_dimr():
    su2_dimr = [su2.dimR([n]) for n in range(1,101)]
    su3_dimr = [su3.dimR(el) for el in su3_irreps]
    su9_dimr = [su9.dimR(el) for el in su9_irreps]

    assert su2_dimr == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                        38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                        89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
    assert su3_dimr == [1, 3, 3, 6, 6, 8, 10, 10, 15, 15, 15, 15, 21, 21, 24, 24, 27, 28, 28,
                        35, 35, 36, 36, 42, 42, 45, 45, 48, 48, 55, 55, 60, 60, 63, 63, 64,
                        66, 66, 78, 78, 80, 80, 81, 81, 90, 90, 91, 91, 99, 99]
    assert su9_dimr == [1, 9, 9, 36, 36, 45, 45, 80, 84, 84]

def test_rep_index():
    irreps_to_check_su3 = [[1,n] for n in range(12)]
    rep_index = [su3._representationIndex(np.array([el])) for el in irreps_to_check_su3]
    assert rep_index == [1, 6, 20, 50, 105, 196, 336, 540, 825, 1210, 1716, 2366]

def test_reps_up_to_dim_n():
    assert su2.repsUpToDimN(5) == [[0], [1], [2], [3], [4]]
    assert su3.repsUpToDimN(10) == [[0, 0], [1, 0], [0, 1], [0, 2], [2, 0], [1, 1], [0, 3], [3, 0]]
    assert su5.repsUpToDimN(10) == [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]

def test_group_with_rank_nsq():
    assert su5._getGroupWithRankNsqr(8) == [('A', 8), ('D', 8), ('B', 8), ('C', 8), ('E', 8)]
    assert su5._cmToFamilyAndSeries() == ('A', 4)

def test_conjugacy_class():
    su5_irreps = su5.repsUpToDimN(10)

    so10_irreps = so10.repsUpToDimN(100)
    so11_irreps = so11.repsUpToDimN(100)

    sp10_irreps = sp10.repsUpToDimN(10)
    sp12_irreps = sp12.repsUpToDimN(100)

    assert [su5._conjugacyClass(el) for el in su5_irreps] == [[0], [1], [4], [2], [3]]
    assert [so10._conjugacyClass(el) for el in so10_irreps[2:]] == [[1, 1], [1, 3], [0, 0], [0, 0]]
    assert [so11._conjugacyClass(el) for el in so11_irreps] == [[0], [0], [1], [0], [0]]
    assert [sp10._conjugacyClass(el) for el in sp10_irreps] == [[0], [1]]
    assert [sp12._conjugacyClass(el) for el in sp12_irreps] == [[0], [1], [0], [0]]

def test_conjugate_irrep():
    assert np.all(su5.conjugateIrrep([1,0,0,0]) == np.array([0, 0, 0, 1]))
    assert np.all(su5.conjugateIrrep([1,1,0,1]) == np.array([1, 0, 1, 1]))

def test_weights():
    assert su2._weights(np.array([1])) == [[np.array([1]), 1], [np.array([-1]), 1]]
    assert sym.flatten(su3._weights(np.array([0, 1]))) == sym.flatten([np.array([np.array([-1,  0]), 1], dtype=object),
                                                                       np.array([np.array([ 1, -1]), 1], dtype=object),
                                                                       np.array([np.array([0, 1]), 1], dtype=object)])
    assert sym.flatten(su3._weights(np.array([1, 1]))) == sym.flatten([[np.array([1, 1]), 1], [np.array([ 2, -1]), 1],
                                                                       [np.array([-1,  2]), 1], [np.array([0, 0]), 2],
                                                                       [np.array([ 1, -2]), 1], [np.array([-2,  1]), 1],
                                                                       [np.array([-1, -1]), 1]])
    assert sym.flatten(su3._weights(np.array([1, 0]))) == sym.flatten([[np.array([1, 0]), 1], [np.array([-1,  1]), 1],
                                                                       [np.array([ 0, -1]), 1]])

def test_rep_minimal_matrices():
    res = su2.repMinimalMatrices(np.array([[2]]))
    assert True
    # assert res[0][0] == sym.matrices.sparse.MutableSparseMatrix([[0, sym.sqrt(2), 0], [0, 0, sym.sqrt(2)], 0, 0, 0])
