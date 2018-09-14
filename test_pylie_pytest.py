import pylie
import numpy as np
import sympy as sym

from test_data import *

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
    su2_casimirs = test_casimir_su2_casimirs
    su3_casimirs = test_casimir_su3_casimirs
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

    assert su2_dimr == test_dimr_su2_dimr
    assert su3_dimr == test_dimr_su3_dimr
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
    assert sym.flatten(su3._weights(np.array([0, 1]))) == sym.flatten(test_weights_res_su3_01)
    assert sym.flatten(su3._weights(np.array([1, 1]))) == sym.flatten(test_weights_res_su3_11)
    assert sym.flatten(su3._weights(np.array([1, 0]))) == sym.flatten(test_weights_res_su3_10)

def test_rep_minimal_matrices():
    res_su2 = su2.repMinimalMatrices(np.array([[2]]))
    res_su3 = su3.repMinimalMatrices(np.array([[1,1]]))
    res_su5 = su5.repMinimalMatrices(np.array([[1, 0, 0, 0]]))
    res_sp12 = sp12.repMinimalMatrices(np.array([[1, 0, 0, 0, 0, 0]]))

    assert sym.flatten(res_su2) == sym.flatten(test_rep_minimal_matrices_res_su2)
    assert sym.flatten(res_su3) == sym.flatten(test_rep_minimal_matrices_res_su3)
    assert sym.flatten(res_su5) == sym.flatten(test_rep_minimal_matrices_res_su5)
    assert sym.flatten(res_sp12) == sym.flatten(test_rep_minimal_matrices_res_sp12)

def test_rep_matrices():
    assert sym.flatten(su2.repMatrices([1])) == sym.flatten(
        [sym.Matrix([[0, sym.Rational(1,2)],
                     [sym.Rational(1,2), 0]]),
         sym.Matrix([[0, -sym.sqrt(sym.Rational(-1,4))],
                     [sym.sqrt(sym.Rational(-1,4)), 0]]),
         sym.Matrix([[sym.Rational(1,2), 0],
                     [0, sym.Rational(-1,2)]])])
    # TODO more tests to add here from examples

def test_invariants():
    a = sym.IndexedBase('a')
    b = sym.IndexedBase('b')
    c = sym.IndexedBase('c')
    d = sym.IndexedBase('d')

    assert su2.invariants([[1], [1]], conj=[False, False]) == [a[1]*b[2] - a[2]*b[1]]
    assert su2.invariants([[1], [1]], conj=[False, True]) == [a[1]*b[1] + a[2]*b[2]]
    assert su2.invariants([[1], [1]], conj=[True, False]) == [a[1]*b[1] + a[2]*b[2]]
    assert su2.invariants([[1], [1]], conj=[True, True]) == [a[1]*b[2] - a[2]*b[1]]

    assert su3.invariants([[1,0], [0,1]]) == [a[1]*b[1] + a[2]*b[2] + a[3]*b[3]]
    assert su3.invariants([[1,0], [1,0], [1,0]]) == test_invariants_res_su3_333
    assert su3.invariants([[1,0], [0,1], [1,1]]) == test_invariants_res_su3_33b8
    assert su4.invariants([[1,1,0], [1,1,0]], conj=[False, True]) == test_invariants_res_su4
