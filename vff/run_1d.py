import torch
from torch import optim

CUDA_ENABLED = torch.cuda.is_available()
print(f"Is GPU available? {CUDA_ENABLED}")
if CUDA_ENABLED:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(f"Device {device}")

import numpy as np
import matplotlib.pyplot as plt

import os
import random
import pickle
import re
import warnings

warnings.simplefilter("ignore", UserWarning)

from misc.plot import plot_losses
from typing import Iterable
# Everything here is based on QUIMB and PyTorch
from tn.hamiltonians import get_hamiltonian
from training.utils import training_loop
from tn.mps_circuit import TNModel, qmps_brick, qmps_brick_quasi_1d, create_targets, load_gates
from tn.data_states import random_product_state, \
    random_mps_state, get_make_data_set_fn, random_U1_state
from tn.trotter import compress_trotterization_into_circuit
from tn.hst import hst
# For next-nearest neighbors we use Tenpy
import quimb.tensor as qtn


def main(config):
    # 0.) CONFIG PARSING

    # META PARAMS
    TRAIN = config['TRAIN']
    PLOT = config['PLOT']
    PRINT = config.get('PRINT', True)
    SHOW = config['SHOW']
    TEST = config['TEST']
    TEST_UNITARY = config['TEST_UNITARY']
    HST = config.get('HST', False)
    SCRATCH_PATH = config.get('SCRATCH_PATH', None)
    SEED = config['SEED']
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    GET_PATH = config.get('GET_PATH', False)

    test_size = config.get('test_size', 100)
    # Model properties
    if 'L' in config.keys():
        L = config['L']
    else:
        Lx, Ly = config['Lx'], config['Ly']
        L = Lx * Ly
    hamiltonian = config['hamiltonian']
    t = config['t']

    trotter_start = config.get('trotter_start', False)
    trotter_start_order = config.get('trotter_start_order', 1)
    # assert
    L, H, hamiltonian_path, trotter_initialization, get_Utrotter, bc = get_hamiltonian(config)
    if PRINT:
        print(f"Model {hamiltonian} of size {L} at time {t:1.3f}\n")
    granularity_from_t = int(np.round(t / 0.05))
    tebd_granularity = config.get('tebd_granularity', granularity_from_t)  # How fine is the TEBD grid
    tebd_cutoff = config.get('tebd_cutoff', -1)  # What is the SVD cutoff for TEBD
    tebd_max_bond = config.get('max_bond', 20)  # What is the bond dimension cutoff for TEBD
    tebd_opts = {'cutoff': tebd_cutoff, 'max_bond': tebd_max_bond}
    ctg = config.get('ctg', False)
    if PRINT:
        print(f"TEBD steps: {tebd_granularity}")
        print(f"TEBD max bond dimension: {tebd_max_bond}")
        print(f"TEBD cutoff: {tebd_cutoff}\n")

    # Training properties
    circuit_name = config['circuit_name']  # What circuit ansatz
    circuit_translation = config.get('circuit_translation', False)
    training_states = config['training_states']  # Type of training states
    num_steps = config['num_steps']  # Number of max iterations in training
    num_samples = config['num_samples']  # Number of training states

    make_data_set = get_make_data_set_fn(hamiltonian, H(L), tebd_granularity, tebd_opts, PRINT)

    assert num_samples > 0
    training_strategy = config['training_strategy']  # Strategy for optimizing the circuit
    if training_states == 'product':
        get_training_state = lambda x, s: random_product_state(x, s)
        training_state_path = f"{training_states}"
    elif training_states == 'mps':
        train_state_bond_dim = config.get('train_state_bond_dim', 2)
        get_training_state = lambda x, s: random_mps_state(x, train_state_bond_dim, s)
        training_state_path = f"mps_bd_{train_state_bond_dim}"
    elif training_states == 'u1':
        num_particles = config.get('num_particles')
        train_state_bond_dim = config.get('train_state_bond_dim', 3)
        assert train_state_bond_dim > 2, 'If train state bond dimension is <= 2 all states are equal.'
        get_training_state = lambda x, s: random_U1_state(x, train_state_bond_dim, num_particles, s)
        training_state_path = f"u1_bd_{train_state_bond_dim}_np_{num_particles}"
    else:
        raise NotImplementedError
    if hamiltonian == 'heisenberg_2d':
        qmps_ansatz = qmps_brick_quasi_1d
    else:
        qmps_ansatz = qmps_brick

    if training_strategy == 'hotstart':
        depth_min = config.get('depth_min')
        depth_max = config.get('depth_max')
        depth_step = config['depth_step']

        training_strategy_path = f"{training_strategy}_dstep_{depth_step}"
        path_name = f'UNITARY_COMPILATION/{hamiltonian_path}/L_{L}/' \
                    f'{training_strategy_path}/t_{t:1.3f}/Nsteps_{num_steps}_' \
                    f'{circuit_name}' \
                    f'{"_translation" if circuit_translation else ""}' \
                    f'{"_trotter_init_" if trotter_start else ""}' \
                    f'{trotter_start_order if trotter_start else ""}/' \
                    f'SEED_{SEED}/Ns_{num_samples}_{training_state_path}_bd_max_{tebd_max_bond}'
        if PRINT:
            print(f"(hotstart) Min depth = {depth_min}")
            print(f"(hotstart) Max depth = {depth_max}")
            print(f"(hotstart) Depth step= {depth_step}")
            print(f"(hotstart) Trotter start= {trotter_start}")
            print(f"(hotstart) Trotter p = {trotter_start_order}\n")

    elif training_strategy == 'double_time':
        max_factor = config.get('max_factor')
        initial_depth = config.get('initial_depth')
        assert not initial_depth % 2, f'`initial_depth` must be even, received {initial_depth}'
        training_strategy_path = f"{training_strategy}_initial_depth_{initial_depth}"
        path_name = f'UNITARY_COMPILATION/{hamiltonian_path}/L_{L}/{training_strategy_path}/' \
                    f'Nsteps_{num_steps}_{circuit_name}' \
                    f'{"_translation" if circuit_translation else ""}/' \
                    f'SEED_{SEED}/t_start_{t:1.3f}/Ns_{num_samples}_{training_state_path}_bd_max_{tebd_max_bond}'
        if PRINT:
            print(f"(double_time) Initial depth = {initial_depth}")
            print(f"(double_time) Max factor = {max_factor}\n")
    elif training_strategy == 'multiple_times':
        times = config.get('times')
        assert isinstance(times, Iterable), f'`times` must be a list an iterable integers, received {type(times)}'
        assert all(isinstance(t_i, float) for t_i in times)
        t = times[0]
        depth = config.get('depth')
        training_strategy_path = f"{training_strategy}"
        path_name = f'UNITARY_COMPILATION/{hamiltonian_path}/L_{L}/{training_strategy_path}/' \
                    f'Nsteps_{num_steps}_{circuit_name}' \
                    f'{"_translation" if circuit_translation else ""}/' \
                    f'SEED_{SEED}/t_start_{t:1.3f}/Ns_{num_samples}_{training_state_path}_bd_max_{tebd_max_bond}'
        if PRINT:
            print(f"(multiple_times) Depth = {depth}")
            print(f"(multiple_times) Times = {times}\n")
    elif training_strategy == 'double_space':
        assert circuit_translation, '`double_space` requires translation invariant circuits'
        assert hamiltonian != 'heisenberg_2d', 'Does not work for 2D models yet'
        max_factor = config.get('max_factor')
        depth = config['depth']
        training_strategy_path = f"L_{L}/{training_strategy}_depth_{depth}"
        path_name = f'UNITARY_COMPILATION/{hamiltonian_path}/{training_strategy_path}/' \
                    f'Nsteps_{num_steps}_{circuit_name}' \
                    f'{"_translation" if circuit_translation else ""}/' \
                    f'SEED_{SEED}/t_{t:1.3f}/L_start_{L}/Ns_{num_samples}_{training_state_path}_bd_max_{tebd_max_bond}'
        if PRINT:
            print(f"(double_space) Depth = {depth}")
            print(f"(double_space) Max factor = {max_factor}\n")
    else:
        raise NotImplementedError
    learning_rate = config['learning_rate']  # Learning rate schedule
    learning_rate_scheduler = config['learning_rate_schedule']  # Learning rate schedule
    if PRINT:
        print(f"Circuit: {circuit_name}")
        print(f"Translation: {circuit_translation}")
        print(f"Type of training state: {training_states}")
        print(f"Number of steps: {num_steps}")
        print(f"Number of samples: {num_samples}")
        print(f"Training strategy: {training_strategy}\n")

    ### 2.) Paths ###
    if SCRATCH_PATH is not None:
        save_path = SCRATCH_PATH + f'/data/{path_name}/'
        fig_path = SCRATCH_PATH + f'/data/{path_name}/figures/'
    else:
        save_path = f'./data/{path_name}/'
        fig_path = f'./data/{path_name}/figures/'

    if GET_PATH:
        return save_path
    elif TRAIN:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path + 'config'):
            with open(save_path + 'config', 'w') as file:
                for k, v in config.items():
                    file.write(f'{k} = {v}\n')

        if training_strategy == 'hotstart':
            # Create or Load Train data
            if not os.path.exists(save_path + 'psi0.pickle') or not os.path.exists(save_path + 'psit.pickle'):

                print("Creating training data set")
                if not hamiltonian == 'heisenberg_2d':
                    psi0_list_train, psit_list_train, tebd_errors = make_data_set(lambda x: get_training_state(L, x), t,
                                                                                  num_samples, SEED)
                else:
                    psi0_list_train, psit_list_train, tebd_errors = make_data_set(
                        lambda x: get_training_state(Lx * Ly, x), t,
                        num_samples, SEED)
                with open(save_path + 'psi0.pickle', 'wb') as file:
                    pickle.dump(psi0_list_train, file)
                with open(save_path + 'psit.pickle', 'wb') as file:
                    pickle.dump(psit_list_train, file)
                np.save(save_path + 'tebd_errors_train', np.array(tebd_errors))
            else:
                with open(save_path + 'psi0.pickle', 'rb') as file:
                    psi0_list_train = pickle.load(file)
                with open(save_path + 'psit.pickle', 'rb') as file:
                    psit_list_train = pickle.load(file)
                tebd_errors = np.load(save_path + 'tebd_errors_train.npy')
                print("Restored train dataset from file")
                assert len(psi0_list_train) == num_samples
                assert len(psit_list_train) == num_samples
            print(f"TEBD error: {np.mean(tebd_errors)}")

            depth_list = list(range(depth_min, depth_max + 1, depth_step))
            for d_i, depth in enumerate(depth_list):
                save_path_depth = save_path + f"depth_{depth}/"
                save_path_depth_previous = save_path + f"depth_{depth - depth_step}/"
                save_path_depth_ckpts = save_path_depth + "ckpts/"
                save_path_depth_ckpts_previous = save_path_depth_previous + "ckpts/"
                if not os.path.exists(save_path_depth + "train_loss.npy"):

                    if not os.path.exists(save_path_depth_ckpts):
                        os.makedirs(save_path_depth_ckpts)
                    # # If we're in the first layer, start with the identity, otherwise, perturb.
                    if (depth == depth_min) and trotter_start:
                        tn = trotter_initialization(-t, 1)
                        new_tn = compress_trotterization_into_circuit(L, tn)
                        start_even = 'Even' in new_tn.tensors[0].tags
                        num_layers = len(list(filter(lambda x: bool(re.match(r'^L\d+$', x)), new_tn.tags)))
                        assert num_layers == depth_min, f"Minmial number of layers is {depth_min}," \
                                                        f" but must be equal to the number of layers " \
                                                        f" in the trotterization: {num_layers}"
                        if hamiltonian != 'heisenberg_2d':
                            psi_pqc = qmps_ansatz(L, in_depth=num_layers, rand=False, val_iden=0.01,
                                                  start_even=start_even)
                        else:
                            psi_pqc = qmps_ansatz(Lx, Ly, in_depth=num_layers, rand=False, val_iden=0.01,
                                                  boundary_condition=bc)
                        load_gates(psi_pqc, new_tn, transpose=True)
                    else:
                        if hamiltonian != 'heisenberg_2d':
                            psi_pqc = qmps_ansatz(L, in_depth=depth, rand=False, val_iden=0.01)
                        else:
                            psi_pqc = qmps_ansatz(Lx, Ly, in_depth=depth, rand=False, val_iden=0.01,
                                                  boundary_condition=bc)

                    psi, psi_tars = create_targets(L, psi_pqc, psi0_list_train, psit_list_train, device=device)
                    model = TNModel(psi, psi_tars, translation=circuit_translation, ctg=ctg)
                    if depth > depth_min:
                        # Load previous parameters, strict=false means we don't need the number of parameters to match
                        model.load_state_dict(torch.load(save_path_depth_ckpts_previous + "parameters.ckpt",
                                                         map_location=torch.device(device)), strict=False)
                    model.eval()
                    print(f"Start loss = {model.forward():1.12f}")
                    model.to(device)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action='ignore',
                            message='.*trace might not generalize.*',
                        )
                        model = torch.jit.trace_module(model, {"forward": []})
                    lr = learning_rate
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    scheduler = learning_rate_scheduler(optimizer)

                    model = training_loop(optimizer, model, scheduler, num_steps, show_progress=PRINT)

                    # Save the model state.
                    torch.save(model.state_dict(), save_path_depth_ckpts + "parameters.ckpt")

                    final_loss = model().cpu().detach().numpy()
                    np.save(save_path_depth + 'train_loss', final_loss)
                else:
                    previous_loss = np.load(save_path_depth + 'train_loss.npy')
                    print(f"Data for depth {depth} exists, loss = {previous_loss}")
                if TEST:
                    if not os.path.exists(save_path_depth + f'test_loss_{test_size}.npy'):
                        # Create or Load test data
                        if not os.path.exists(save_path + 'psi0_test.pickle') or not os.path.exists(
                                save_path + 'psit_test.pickle'):
                            print("Creating test dataset")
                            psi0_list_test, psit_list_test, tebd_errors = make_data_set(
                                lambda x: get_training_state(L, x), t,
                                test_size, SEED + 10 ** 6)
                            np.save(save_path + 'tebd_errors_test', np.array(tebd_errors))
                            with open(save_path + 'psi0_test.pickle', 'wb') as file:
                                pickle.dump(psi0_list_test, file)
                            with open(save_path + 'psit_test.pickle', 'wb') as file:
                                pickle.dump(psit_list_test, file)
                        else:
                            with open(save_path + 'psi0_test.pickle', 'rb') as file:
                                psi0_list_test = pickle.load(file)
                            with open(save_path + 'psit_test.pickle', 'rb') as file:
                                psit_list_test = pickle.load(file)
                            tebd_errors = np.load(save_path + 'tebd_errors_test.npy')
                            print("Restored test dataset from file")
                            assert len(psi0_list_test) == test_size
                            assert len(psit_list_test) == test_size
                        print(f"TEBD error: {np.mean(tebd_errors)}")
                        # If we're in the first layer, start with the identity, otherwise, perturb.
                        if hamiltonian != 'heisenberg_2d':
                            psi_pqc = qmps_ansatz(L, in_depth=depth, rand=False, val_iden=0.01)
                        else:
                            psi_pqc = qmps_ansatz(Lx, Ly, in_depth=depth, rand=False, val_iden=0.01,
                                                  boundary_condition=bc)
                        psi, psi_tars = create_targets(L, psi_pqc, psi0_list_test, psit_list_test, device=device)
                        model = TNModel(psi, psi_tars, translation=circuit_translation, ctg=ctg)
                        # Load previous parameters,
                        # strict=false means we don't need the number of parameters to match
                        model.load_state_dict(torch.load(save_path_depth_ckpts + "parameters.ckpt",
                                                         map_location=torch.device(device)), strict=True)
                        model.eval()
                        model.to(device)
                        test_loss = model().cpu().detach().numpy()
                        np.save(save_path_depth + f'test_loss_{test_size}', test_loss)
                    else:
                        test_loss = np.load(save_path_depth + f'test_loss_{test_size}.npy')

                    print(f"Test loss for {test_size} samples = {test_loss}")
                if TEST_UNITARY:
                    if not os.path.exists(save_path_depth + f'unitary_loss.npy') & 0:
                        if hamiltonian != 'heisenberg_2d':
                            psi_pqc = qmps_ansatz(L, in_depth=depth)
                        else:
                            psi_pqc = qmps_ansatz(Lx, Ly, in_depth=depth, boundary_condition=bc)
                        # Last index is outer leg
                        outer_indices = []
                        for l in range(L):
                            outer_indices.append(psi_pqc.tensors[l].inds[-1])

                        psi = psi_pqc.tensors[L]
                        for i in range(L + 1, len(psi_pqc.tensors)):
                            psi = psi & psi_pqc.tensors[i]

                        psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
                        circuit = TNModel(psi, translation=circuit_translation)
                        # Load the trained parameters
                        circuit.load_state_dict(torch.load(save_path_depth_ckpts + "parameters.ckpt",
                                                           map_location=torch.device(device)), strict=True)
                        if circuit_translation:
                            params = {}
                            for i in range(circuit.number_of_gates):
                                su4_tag = next(filter(lambda x: 'SU4' in x, circuit.skeleton.tensors[i].tags))
                                _, q1, q2, layer_i = su4_tag.split('_')
                                params[i] = circuit.torch_params[layer_i].detach().numpy()
                        else:
                            params = {int(i): p.detach().numpy() for i, p in list(circuit.torch_params.items())}

                        Ucircuit = qtn.unpack(params, circuit.skeleton)

                        if not os.path.exists(save_path + 'Utrotter.pickle'):
                            print("Creating Trotterization tensor network")
                            Utrotter = get_Utrotter(t)
                            with open(save_path + 'Utrotter.pickle', 'wb') as file:
                                pickle.dump(Utrotter, file)
                        else:
                            with open(save_path + 'Utrotter.pickle', 'rb') as file:
                                Utrotter = pickle.load(file)
                            print("Restored Trotterization tensor network from disk")

                        Ucircuit = Ucircuit.isometrize(method='qr')
                        Ucircuit = Ucircuit.reindex(dict(zip(outer_indices, [f"b{i}" for i in range(L)], )))
                        unitary_loss = 1 - (abs((Ucircuit & Utrotter) ^ all) / 2 ** L) ** 2
                        np.save(save_path_depth + "unitary_loss", unitary_loss)
                    else:
                        unitary_loss = np.load(save_path_depth + f'unitary_loss.npy')
                    print(f"Unitary loss for = {unitary_loss}")
                elif HST:
                    if not os.path.exists(save_path_depth + f'hst_loss.npy'):
                        if hamiltonian != 'heisenberg_2d':
                            psi_pqc = qmps_ansatz(L, in_depth=depth)
                        else:
                            psi_pqc = qmps_ansatz(Lx, Ly, in_depth=depth, boundary_condition=bc)
                        # Last index is outer leg
                        outer_indices = []
                        for l in range(L):
                            outer_indices.append(psi_pqc.tensors[l].inds[-1])

                        psi = psi_pqc.tensors[L]
                        for i in range(L + 1, len(psi_pqc.tensors)):
                            psi = psi & psi_pqc.tensors[i]

                        psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
                        circuit = TNModel(psi, translation=circuit_translation)
                        # Load the trained parameters
                        circuit.load_state_dict(torch.load(save_path_depth_ckpts + "parameters.ckpt",
                                                           map_location=torch.device(device)), strict=True)
                        if circuit_translation:
                            params = {}
                            for i in range(circuit.number_of_gates):
                                layer_tag = next(
                                    filter(lambda x: re.match(r'^L\d+$', x), circuit.skeleton.tensors[i].tags))
                                layer_i = layer_tag[1:]
                                params[i] = circuit.torch_params[layer_i].detach().numpy()
                        else:
                            # raise NotImplementedError
                            params = {int(i): p.detach().numpy() for i, p in list(circuit.torch_params.items())}
                        Ucircuit = qtn.unpack(params, circuit.skeleton)
                        Ucircuit = Ucircuit.isometrize(method='qr')
                        V = get_Utrotter(t)
                        hst_loss = (1 - hst(L, Ucircuit, V, cutoff=1e-8, trans_inv=circuit_translation))
                        np.save(save_path_depth + "hst_loss", hst_loss)
                    else:
                        hst_loss = np.load(save_path_depth + f'hst_loss.npy')
                    print(f"HST loss = {hst_loss}")
        elif training_strategy == 'double_time':
            for i, t_i in enumerate([t * 2 ** f for f in range(max_factor)]):
                print(f"Training for t={t_i:1.3f}")
                save_path_t_i = save_path + f't_{t_i:1.3f}/'
                save_path_t_i_previous = save_path + f't_{t_i / 2:1.3f}/'
                if not os.path.exists(save_path_t_i):
                    os.makedirs(save_path_t_i)
                # Create or Load Train data
                if not os.path.exists(save_path_t_i + 'psi0.pickle') or not os.path.exists(
                        save_path_t_i + 'psit.pickle'):

                    print("Creating training data set")
                    psi0_list_train, psit_list_train, tebd_errors = make_data_set(lambda x: get_training_state(L, x),
                                                                                  t_i,
                                                                                  num_samples, SEED)

                    with open(save_path_t_i + 'psi0.pickle', 'wb') as file:
                        pickle.dump(psi0_list_train, file)
                    with open(save_path_t_i + 'psit.pickle', 'wb') as file:
                        pickle.dump(psit_list_train, file)
                    np.save(save_path_t_i + 'tebd_errors_train', np.array(tebd_errors))
                else:
                    with open(save_path_t_i + 'psi0.pickle', 'rb') as file:
                        psi0_list_train = pickle.load(file)
                    with open(save_path_t_i + 'psit.pickle', 'rb') as file:
                        psit_list_train = pickle.load(file)
                    tebd_errors = np.load(save_path_t_i + 'tebd_errors_train.npy')
                    print("Restored train dataset from file")
                    assert len(psi0_list_train) == num_samples
                    assert len(psit_list_train) == num_samples
                print(f"TEBD error: {np.mean(tebd_errors)}")

                depth = initial_depth * 2 ** i
                save_path_depth = save_path_t_i + f"depth_{depth}/"
                save_path_depth_previous = save_path_t_i_previous + f"depth_{depth // 2}/"
                save_path_depth_ckpts = save_path_depth + "ckpts/"
                save_path_depth_ckpts_previous = save_path_depth_previous + "ckpts/"
                print(f"DEPTH = {depth}")

                if not os.path.exists(save_path_depth + "train_loss.npy"):
                    if not os.path.exists(save_path_depth_ckpts):
                        os.makedirs(save_path_depth_ckpts)
                    # # If we're in the first layer, start with the identity, otherwise, perturb.
                    if hamiltonian != 'heisenberg_2d':
                        psi_pqc = qmps_ansatz(L, in_depth=depth, rand=False, val_iden=0.01)
                    else:
                        psi_pqc = qmps_ansatz(Lx, Ly, in_depth=depth, rand=False, val_iden=0.01, boundary_condition=bc)
                    psi, psi_tars = create_targets(L, psi_pqc, psi0_list_train, psit_list_train, device=device)
                    model = TNModel(psi, psi_tars, translation=circuit_translation, ctg=ctg)
                    if i > 0:
                        # Load previous parameters, strict=false means we don't need the number of parameters to match
                        torch_params = torch.load(save_path_depth_ckpts_previous + "parameters.ckpt",
                                                  map_location=torch.device(device))
                        new_torch_params = dict()
                        for c_i in range(2):
                            for k, v in torch_params.items():
                                index = int(k.split(".")[-1])
                                new_torch_params[f"torch_params.{index + c_i * len(torch_params)}"] = v.clone()
                        model.load_state_dict(new_torch_params, strict=True)
                    model.eval()
                    print(f"Start loss = {model.forward():1.12f}")
                    model.to(device)

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action='ignore',
                            message='.*trace might not generalize.*',
                        )
                        model = torch.jit.trace_module(model, {"forward": []})
                    print(f"Start loss = {model.forward():1.12f}")

                    lr = learning_rate
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    scheduler = learning_rate_scheduler(optimizer)

                    model = training_loop(optimizer, model, scheduler, num_steps, show_progress=PRINT)

                    # Save the model state.
                    torch.save(model.state_dict(), save_path_depth_ckpts + "parameters.ckpt")

                    final_loss = model().cpu().detach().numpy()
                    np.save(save_path_depth + 'train_loss', final_loss)
                else:
                    previous_loss = np.load(save_path_depth + 'train_loss.npy')
                    print(f"Data for depth {depth} exists, loss = {previous_loss}")
                if TEST:
                    if not os.path.exists(save_path_depth + f'test_loss_{test_size}.npy'):
                        # Create or Load test data
                        if not os.path.exists(save_path_t_i + 'psi0_test.pickle') or not os.path.exists(
                                save_path_t_i + 'psit_test.pickle'):
                            print("Creating test dataset")
                            psi0_list_test, psit_list_test, tebd_errors = make_data_set(
                                lambda x: get_training_state(L, x), t_i,
                                test_size, SEED + 10 ** 6)

                            np.save(save_path_t_i + 'tebd_errors_test', np.array(tebd_errors))
                            with open(save_path_t_i + 'psi0_test.pickle', 'wb') as file:
                                pickle.dump(psi0_list_test, file)
                            with open(save_path_t_i + 'psit_test.pickle', 'wb') as file:
                                pickle.dump(psit_list_test, file)
                        else:
                            with open(save_path_t_i + 'psi0_test.pickle', 'rb') as file:
                                psi0_list_test = pickle.load(file)
                            with open(save_path_t_i + 'psit_test.pickle', 'rb') as file:
                                psit_list_test = pickle.load(file)
                            tebd_errors = np.load(save_path_t_i + 'tebd_errors_test.npy')
                            print("Restored test dataset from file")
                            assert len(psi0_list_test) == test_size
                            assert len(psit_list_test) == test_size
                        print(f"TEBD error: {np.mean(tebd_errors)}")
                        # If we're in the first layer, start with the identity, otherwise, perturb.
                        if hamiltonian != 'heisenberg_2d':
                            psi_pqc = qmps_ansatz(L, in_depth=depth, rand=False, val_iden=0.01)
                        else:
                            psi_pqc = qmps_ansatz(Lx, Ly, in_depth=depth, rand=False, val_iden=0.01,
                                                  boundary_condition=bc)
                        psi, psi_tars = create_targets(L, psi_pqc, psi0_list_test, psit_list_test, device=device)
                        model = TNModel(psi, psi_tars, translation=circuit_translation, ctg=ctg)
                        # Load previous parameters,
                        # strict=false means we don't need the number of parameters to match

                        model.load_state_dict(torch.load(save_path_depth_ckpts + "parameters.ckpt",
                                                         map_location=torch.device(device)), strict=True)
                        model.eval()
                        model.to(device)
                        test_loss = model().cpu().detach().numpy()
                        np.save(save_path_depth + f'test_loss_{test_size}', test_loss)
                    else:
                        test_loss = np.load(save_path_depth + f'test_loss_{test_size}.npy')
                    print(f"Test loss for {test_size} samples = {test_loss}")
        elif training_strategy == 'multiple_times':
            for i, t_i in enumerate(times):
                print(f"Training for t={t_i:1.3f}")
                save_path_t_i = save_path + f't_{t_i:1.3f}/'
                save_path_t_i_previous = save_path + f't_{times[i - 1]:1.3f}/'
                if not os.path.exists(save_path_t_i):
                    os.makedirs(save_path_t_i)
                # Create or Load Train data
                if not os.path.exists(save_path_t_i + 'psi0.pickle') or not os.path.exists(
                        save_path_t_i + 'psit.pickle'):

                    print("Creating training data set")
                    psi0_list_train, psit_list_train, tebd_errors = make_data_set(lambda x: get_training_state(L, x),
                                                                                  t_i,
                                                                                  num_samples, SEED)

                    with open(save_path_t_i + 'psi0.pickle', 'wb') as file:
                        pickle.dump(psi0_list_train, file)
                    with open(save_path_t_i + 'psit.pickle', 'wb') as file:
                        pickle.dump(psit_list_train, file)
                    np.save(save_path_t_i + 'tebd_errors_train', np.array(tebd_errors))
                else:
                    with open(save_path_t_i + 'psi0.pickle', 'rb') as file:
                        psi0_list_train = pickle.load(file)
                    with open(save_path_t_i + 'psit.pickle', 'rb') as file:
                        psit_list_train = pickle.load(file)
                    tebd_errors = np.load(save_path_t_i + 'tebd_errors_train.npy')
                    print("Restored train dataset from file")
                    assert len(psi0_list_train) == num_samples
                    assert len(psit_list_train) == num_samples
                print(f"TEBD error: {np.mean(tebd_errors)}")

                save_path_depth = save_path_t_i + f"depth_{depth}/"
                save_path_depth_previous = save_path_t_i_previous + f"depth_{depth}/"
                save_path_depth_ckpts = save_path_depth + "ckpts/"
                save_path_depth_ckpts_previous = save_path_depth_previous + "ckpts/"
                print(f"DEPTH = {depth}")

                if not os.path.exists(save_path_depth + "train_loss.npy"):
                    if not os.path.exists(save_path_depth_ckpts):
                        os.makedirs(save_path_depth_ckpts)
                    # # If we're in the first layer, start with the identity, otherwise, perturb.
                    if hamiltonian != 'heisenberg_2d':
                        psi_pqc = qmps_ansatz(L, in_depth=depth, rand=False, val_iden=0.01)
                    else:
                        psi_pqc = qmps_ansatz(Lx, Ly, in_depth=depth, rand=False, val_iden=0.01, boundary_condition=bc)
                    psi, psi_tars = create_targets(L, psi_pqc, psi0_list_train, psit_list_train, device=device)
                    model = TNModel(psi, psi_tars, translation=circuit_translation, ctg=ctg)
                    if i > 0:
                        # Load previous parameters, strict=false means we don't need the number of parameters to match
                        torch_params = torch.load(save_path_depth_ckpts_previous + "parameters.ckpt",
                                                  map_location=torch.device(device))
                        new_torch_params = dict()
                        for k, v in torch_params.items():
                            index = int(k.split(".")[-1])
                            new_torch_params[f"torch_params.{index}"] = v.clone()
                        model.load_state_dict(new_torch_params, strict=True)
                    model.eval()
                    print(f"Start loss = {model.forward():1.12f}")
                    model.to(device)

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action='ignore',
                            message='.*trace might not generalize.*',
                        )
                        model = torch.jit.trace_module(model, {"forward": []})
                    print(f"Start loss = {model.forward():1.12f}")

                    lr = learning_rate
                    if i==0:
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                    else:
                        optimizer = optim.SGD(model.parameters(), lr=lr)

                    scheduler = learning_rate_scheduler(optimizer)

                    model = training_loop(optimizer, model, scheduler, num_steps, show_progress=PRINT)

                    # Save the model state.
                    torch.save(model.state_dict(), save_path_depth_ckpts + "parameters.ckpt")

                    final_loss = model().cpu().detach().numpy()
                    np.save(save_path_depth + 'train_loss', final_loss)
                else:
                    previous_loss = np.load(save_path_depth + 'train_loss.npy')
                    print(f"Data for depth {depth} exists, loss = {previous_loss}")
                if TEST:
                    if not os.path.exists(save_path_depth + f'test_loss_{test_size}.npy'):
                        # Create or Load test data
                        if not os.path.exists(save_path_t_i + 'psi0_test.pickle') or not os.path.exists(
                                save_path_t_i + 'psit_test.pickle'):
                            print("Creating test dataset")
                            psi0_list_test, psit_list_test, tebd_errors = make_data_set(
                                lambda x: get_training_state(L, x), t_i,
                                test_size, SEED + 10 ** 6)

                            np.save(save_path_t_i + 'tebd_errors_test', np.array(tebd_errors))
                            with open(save_path_t_i + 'psi0_test.pickle', 'wb') as file:
                                pickle.dump(psi0_list_test, file)
                            with open(save_path_t_i + 'psit_test.pickle', 'wb') as file:
                                pickle.dump(psit_list_test, file)
                        else:
                            with open(save_path_t_i + 'psi0_test.pickle', 'rb') as file:
                                psi0_list_test = pickle.load(file)
                            with open(save_path_t_i + 'psit_test.pickle', 'rb') as file:
                                psit_list_test = pickle.load(file)
                            tebd_errors = np.load(save_path_t_i + 'tebd_errors_test.npy')
                            print("Restored test dataset from file")
                            assert len(psi0_list_test) == test_size
                            assert len(psit_list_test) == test_size
                        print(f"TEBD error: {np.mean(tebd_errors)}")
                        # If we're in the first layer, start with the identity, otherwise, perturb.
                        if hamiltonian != 'heisenberg_2d':
                            psi_pqc = qmps_ansatz(L, in_depth=depth, rand=False, val_iden=0.01)
                        else:
                            psi_pqc = qmps_ansatz(Lx, Ly, in_depth=depth, rand=False, val_iden=0.01,
                                                  boundary_condition=bc)
                        psi, psi_tars = create_targets(L, psi_pqc, psi0_list_test, psit_list_test, device=device)
                        model = TNModel(psi, psi_tars, translation=circuit_translation, ctg=ctg)
                        # Load previous parameters,
                        # strict=false means we don't need the number of parameters to match

                        model.load_state_dict(torch.load(save_path_depth_ckpts + "parameters.ckpt",
                                                         map_location=torch.device(device)), strict=True)
                        model.eval()
                        model.to(device)
                        test_loss = model().cpu().detach().numpy()
                        np.save(save_path_depth + f'test_loss_{test_size}', test_loss)
                    else:
                        test_loss = np.load(save_path_depth + f'test_loss_{test_size}.npy')
                    print(f"Test loss for {test_size} samples = {test_loss}")
        elif training_strategy == 'double_space':
            for i, L_i in enumerate([L * 2 ** f for f in range(max_factor)]):
                print(f"Training for t={t:1.3f} - L={L_i}")
                make_data_set = get_make_data_set_fn(hamiltonian, H(L_i), tebd_granularity, tebd_opts, PRINT)

                save_path_L_i = save_path + f'L_{L_i}/'
                save_path_L_i_previous = save_path + f'L_{L_i // 2}/'
                if not os.path.exists(save_path_L_i):
                    os.makedirs(save_path_L_i)
                # Create or Load Train data
                if not os.path.exists(save_path_L_i + 'psi0.pickle') or not os.path.exists(
                        save_path_L_i + 'psit.pickle'):
                    print("Creating training data set")
                    psi0_list_train, psit_list_train, tebd_errors = make_data_set(lambda x: get_training_state(L_i, x),
                                                                                  t,
                                                                                  num_samples, SEED)
                    with open(save_path_L_i + 'psi0.pickle', 'wb') as file:
                        pickle.dump(psi0_list_train, file)
                    with open(save_path_L_i + 'psit.pickle', 'wb') as file:
                        pickle.dump(psit_list_train, file)
                    np.save(save_path_L_i + 'tebd_errors_train', np.array(tebd_errors))
                else:
                    with open(save_path_L_i + 'psi0.pickle', 'rb') as file:
                        psi0_list_train = pickle.load(file)
                    with open(save_path_L_i + 'psit.pickle', 'rb') as file:
                        psit_list_train = pickle.load(file)
                    tebd_errors = np.load(save_path_L_i + 'tebd_errors_train.npy')
                    print("Restored train dataset from file")
                    assert len(psi0_list_train) == num_samples
                    assert len(psit_list_train) == num_samples
                print(f"TEBD error: {np.mean(tebd_errors)}")

                save_path_depth = save_path_L_i + f"depth_{depth}/"
                save_path_depth_previous = save_path_L_i_previous + f"depth_{depth}/"
                save_path_depth_ckpts = save_path_depth + "ckpts/"
                save_path_depth_ckpts_previous = save_path_depth_previous + "ckpts/"
                print(f"DEPTH = {depth}")

                if not os.path.exists(save_path_depth + "train_loss.npy"):
                    if not os.path.exists(save_path_depth_ckpts):
                        os.makedirs(save_path_depth_ckpts)
                    # # If we're in the first layer, start with the identity, otherwise, perturb.
                    psi_pqc = qmps_ansatz(L_i, in_depth=depth, rand=False, val_iden=0.01)
                    psi, psi_tars = create_targets(L_i, psi_pqc, psi0_list_train, psit_list_train, device=device)
                    model = TNModel(psi, psi_tars, translation=circuit_translation, ctg=ctg)
                    if i > 0:
                        model.load_state_dict(torch.load(save_path_depth_ckpts_previous + "parameters.ckpt",
                                                         map_location=torch.device(device)), strict=True)
                    model.eval()

                    model.to(device)

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action='ignore',
                            message='.*trace might not generalize.*',
                        )
                        model = torch.jit.trace_module(model, {"forward": []})
                    start_loss = model.forward()
                    print(f"Start loss = {start_loss:1.12f}")
                    lr = learning_rate
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    scheduler = learning_rate_scheduler(optimizer)

                    model = training_loop(optimizer, model, scheduler, num_steps, show_progress=PRINT)

                    # Save the model state.
                    torch.save(model.state_dict(), save_path_depth_ckpts + "parameters.ckpt")

                    final_loss = model().cpu().detach().numpy()
                    np.save(save_path_depth + 'train_loss_start', start_loss.cpu().detach().numpy())
                    np.save(save_path_depth + 'train_loss', final_loss)
                else:
                    previous_loss_start = np.load(save_path_depth + 'train_loss_start.npy')
                    previous_loss = np.load(save_path_depth + 'train_loss.npy')
                    print(f"Data for depth {depth} exists, loss = {previous_loss}")
                if TEST:
                    if not os.path.exists(save_path_depth + f'test_loss_{test_size}.npy'):
                        # Create or Load test data
                        if not os.path.exists(save_path_L_i + 'psi0_test.pickle') or not os.path.exists(
                                save_path_L_i + 'psit_test.pickle'):
                            print("Creating test dataset")
                            psi0_list_test, psit_list_test, tebd_errors = make_data_set(
                                lambda x: get_training_state(L_i, x), t,
                                test_size, SEED + 10 ** 6)

                            np.save(save_path_L_i + 'tebd_errors_test', np.array(tebd_errors))
                            with open(save_path_L_i + 'psi0_test.pickle', 'wb') as file:
                                pickle.dump(psi0_list_test, file)
                            with open(save_path_L_i + 'psit_test.pickle', 'wb') as file:
                                pickle.dump(psit_list_test, file)
                        else:
                            with open(save_path_L_i + 'psi0_test.pickle', 'rb') as file:
                                psi0_list_test = pickle.load(file)
                            with open(save_path_L_i + 'psit_test.pickle', 'rb') as file:
                                psit_list_test = pickle.load(file)
                            tebd_errors = np.load(save_path_L_i + 'tebd_errors_test.npy')
                            print("Restored test dataset from file")
                            assert len(psi0_list_test) == test_size
                            assert len(psit_list_test) == test_size
                        print(f"TEBD error: {np.mean(tebd_errors)}")
                        # If we're in the first layer, start with the identity, otherwise, perturb.
                        psi_pqc = qmps_ansatz(L_i, in_depth=depth, rand=False, val_iden=0.01)
                        psi, psi_tars = create_targets(L_i, psi_pqc, psi0_list_test, psit_list_test, device=device)
                        model = TNModel(psi, psi_tars, translation=circuit_translation, ctg=ctg)
                        # Load previous parameters,
                        # strict=false means we don't need the number of parameters to match

                        model.load_state_dict(torch.load(save_path_depth_ckpts + "parameters.ckpt",
                                                         map_location=torch.device(device)), strict=True)
                        model.eval()
                        model.to(device)
                        test_loss = model().cpu().detach().numpy()
                        np.save(save_path_depth + f'test_loss_{test_size}', test_loss)
                    else:
                        test_loss = np.load(save_path_depth + f'test_loss_{test_size}.npy')
                    print(f"Test loss for {test_size} samples = {test_loss}")
    if PLOT:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        if training_strategy == 'hotstart':
            train_losses = []
            test_losses = []
            unitary_losses = []
            for depth in depth_list:
                save_path_depth = save_path + f"depth_{depth}/"
                train_losses.append(np.load(save_path_depth + 'train_loss.npy'))
                try:
                    test_losses.append(np.load(save_path_depth + f'test_loss_{test_size}.npy'))
                except FileNotFoundError:
                    test_losses.append(np.nan)
                try:
                    unitary_losses.append(np.load(save_path_depth + 'unitary_loss.npy'))
                except FileNotFoundError:
                    unitary_losses.append(np.nan)
            train_losses = np.array(train_losses).flatten()
            test_losses = np.array(test_losses).flatten()
            unitary_losses = np.array(unitary_losses).flatten()

            plot_losses(depth_list, train_losses, test_losses, unitary_losses, fig_path)
            if SHOW:
                plt.show()
        elif training_strategy == 'double_time':
            train_losses = []
            test_losses = []
            unitary_losses = []
            depth_list = []
            for i, t_i in enumerate([t * 2 ** f for f in range(max_factor)]):
                save_path_t_i = save_path + f't_{t_i:1.3f}/'
                depth = initial_depth * 2 ** i
                depth_list.append(depth)
                save_path_depth = save_path_t_i + f"depth_{depth}/"
                train_losses.append(np.load(save_path_depth + 'train_loss.npy'))
                try:
                    test_losses.append(np.load(save_path_depth + f'test_loss_{test_size}.npy'))
                except FileNotFoundError:
                    test_losses.append(np.nan)
                try:
                    unitary_losses.append(np.load(save_path_depth + 'unitary_loss.npy'))
                except FileNotFoundError:
                    unitary_losses.append(np.nan)
            train_losses = np.array(train_losses).flatten()
            test_losses = np.array(test_losses).flatten()
            unitary_losses = np.array(unitary_losses).flatten()
            plot_losses(depth_list, train_losses, test_losses, unitary_losses, fig_path)
            if SHOW:
                plt.show()
