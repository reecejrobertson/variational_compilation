import numpy as np

from .tebd import ising_hamiltonian_quimb, longitudinal_ising_hamiltonian_quimb, heisenberg_hamiltonian_quimb, \
    mbl_hamiltonian_quimb
from .trotter import trotter_evolution_optimized_nn_ising_tn, trotter_evolution_optimized_nn_heisenberg_tn, \
    trotter_evolution_optimized_nn_mbl_tn, trotter_evolution_optimized_ising_nnn_tn


def get_hamiltonian(config):
    hamiltonian = config['hamiltonian']

    if 'L' in config.keys():
        L = config['L']
    else:
        Lx, Ly = config['Lx'], config['Ly']
        L = Lx * Ly

    trotter_start = config.get('trotter_start', False)
    trotter_start_order = config.get('trotter_start_order', 1)

    tebd_test_steps = config.get('tebd_test_steps', 20)
    bc = None
    if hamiltonian == 'ising':
        g = config.get('g', 1.0)
        H = lambda x: ising_hamiltonian_quimb(x, 1.0, g)
        hamiltonian_path = f"{hamiltonian}_g_{g:1.2f}"
        trotter_initialization = lambda s, d: trotter_evolution_optimized_nn_ising_tn(L, 1.0, g, 0.0,
                                                                                      s / d,
                                                                                      d,
                                                                                      p=trotter_start_order)
        get_Utrotter = lambda s: trotter_evolution_optimized_nn_ising_tn(L, 1.0, g, 0.0, s / tebd_test_steps,
                                                                         tebd_test_steps, p=2)
    elif hamiltonian == 'longitudinal_ising':
        jx = config.get('jx', 1.0)
        jz = config.get('jz', 1.0)
        H = lambda x: longitudinal_ising_hamiltonian_quimb(x, 1.0, jz, jx)
        hamiltonian_path = f"{hamiltonian}_jz_{jz:1.2f}_jx_{jx:1.2f}"
        trotter_initialization = lambda s, d: trotter_evolution_optimized_nn_ising_tn(L, 1.0, jz, jx,
                                                                                      s / d,
                                                                                      d,
                                                                                      p=trotter_start_order)
        get_Utrotter = lambda s: trotter_evolution_optimized_nn_ising_tn(L, 1.0, jz, jx, s / tebd_test_steps,
                                                                         tebd_test_steps, p=2)
    elif hamiltonian == 'heisenberg':
        H = lambda x: heisenberg_hamiltonian_quimb(x)
        hamiltonian_path = f"{hamiltonian}"
        trotter_initialization = lambda s, d: trotter_evolution_optimized_nn_heisenberg_tn(L,
                                                                                           s / d,
                                                                                           d,
                                                                                           p=trotter_start_order)
        get_Utrotter = lambda s: trotter_evolution_optimized_nn_heisenberg_tn(L, s / tebd_test_steps,
                                                                              tebd_test_steps, p=2)
    elif hamiltonian == 'heisenberg_2d':
        bc = config.get('boundary_condition', (False, False))
        H = lambda x: (Lx, Ly, bc)
        bc_str = ('_closed_Lx' if bc[0] else '') + ('_closed_Ly' if bc[1] else '')
        hamiltonian_path = f"{hamiltonian}/{Lx}x{Ly}{bc_str}"
        trotter_initialization = lambda s, d: NotImplementedError
        get_Utrotter = lambda s: NotImplementedError
        assert not trotter_start, 'Trotter start not possible for 2D models'

    elif hamiltonian == 'mbl':
        sigma = config.get('sigma', 1.0)
        delta = config.get('delta', 1.0)
        dh = (2.0 * np.random.rand(L) - 1.0) * sigma
        H = lambda x: mbl_hamiltonian_quimb(x, delta, dh)
        hamiltonian_path = f"{hamiltonian}_sigma_{sigma:1.3f}"
        trotter_initialization = lambda s, d: trotter_evolution_optimized_nn_mbl_tn(L, delta, dh,
                                                                                    s / d,
                                                                                    d,
                                                                                    p=trotter_start_order)
        get_Utrotter = lambda s: trotter_evolution_optimized_nn_mbl_tn(L, delta, dh, s / tebd_test_steps,
                                                                       tebd_test_steps, p=2)
    elif hamiltonian == 'ising_nnn':
        J = config.get('J', -1.0)
        V = config.get('V', 1.0)
        # define the model, notice the extra factor of 4 and 2
        # assert not L % 2, "Number of sites must be even for NNN Ising"

        trotter_initialization = lambda s, d: trotter_evolution_optimized_ising_nnn_tn(L, J, V,
                                                                                       s / d,
                                                                                       d,
                                                                                       p=trotter_start_order)
        H = lambda x: (x, J, V)
        hamiltonian_path = f"{hamiltonian}_J_{J:1.3f}_V_{V:1.3f}"

    else:
        raise NotImplementedError
    return L, H, hamiltonian_path, trotter_initialization, get_Utrotter, bc
