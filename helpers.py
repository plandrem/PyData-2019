import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import ortho_group


def get_least_norm_matrices(show=False):
    np.random.seed(1)

    N = 16  # Number of Modes

    # Generate some random eigenvalue magnitudes
    a = 1 - 0.2 ** np.linspace(3, 5, N // 2)

    # Generate some random frequencies
    ϕ = np.logspace(-2, 0, N // 2)
    np.random.shuffle(ϕ)

    # Generate some random phase shifts
    δ = np.random.rand(N // 2) * 2

    # Create eigenvalues
    λ = a * np.exp(1j*ϕ)

    T = 1000
    ts = np.arange(T)

    if show:
        # Plot the oscillatory modes
        fig, axs = plt.subplots(N // 2, sharex=True, figsize=(10, 10))
        plt.subplots_adjust(hspace=0)

        for λi, δi, ax in zip(λ, δ, axs):
            
            xs = λi**ts * np.exp(1j*δi)
            
            ax.plot(ts, xs)
            ax.set_yticks([])

        plt.xlim(0, T)
        plt.xlabel('Time (s)')
        plt.show()    

    # Construct A matrix

    # Build blocks for real modal form
    blocks = []
    for λi in λ:
        block = np.array([[ λi.real, λi.imag],
                          [-λi.imag, λi.real]])
        
        blocks.append(block)
        
    Λ = sp.linalg.block_diag(*blocks)

    # - randomly rotate to get a non-modal matrix
    Q = ortho_group.rvs(dim=N)

    A = Q.T @ Λ @ Q

    # Build Linear Dynamic System matrices
    C = np.random.rand(2, A.shape[0])
    B = np.random.randn(N, 2)

    # Give the system a kick and see what happens - the initial state will be an arbitrary displacement from x = 0
    if show:
        x = np.zeros(N)
        x[4] = 1

        ys = np.empty((len(ts), 2))
        for t in ts:
            x = A @ x
            ys[t, :] = C @ x


        plt.figure(figsize=(10, 4))
        plt.plot(ts, ys[:, 0])
        plt.xlim(0, 350)
        plt.ylim(-1, 1)
        plt.xlabel('Time (s)')
        # plt.legend(['$y_1$', '$y_2$'])

        plt.show()

    return A, B, C, N