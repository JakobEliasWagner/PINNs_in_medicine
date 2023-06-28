import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

LAMBDA = 5
LEFT_BOUNDARY = -np.pi / 2
RIGHT_BOUNDARY = np.pi / 2
SAMPLES = 200
MAX_ITERS = 10000


def jacobi_iteration(u: torch.Tensor, alpha: float):
    """
    1d jakobi solver
    :param A:
    :param u:
    :param b:
    :return:
    """
    u_new = u
    for i in range(1, u.shape[0] - 1):
        u_new[i] = alpha * (u[i + 1] + u[i - 1])
    return u_new


if __name__ == "__main__":
    x = torch.linspace(LEFT_BOUNDARY, RIGHT_BOUNDARY, SAMPLES)
    u = torch.zeros((SAMPLES,))  # uniform grid
    u[0] = np.sin(LAMBDA * LEFT_BOUNDARY)
    u[-1] = np.sin(LAMBDA * RIGHT_BOUNDARY)

    # assemble system matrix
    h = (RIGHT_BOUNDARY - LEFT_BOUNDARY) / SAMPLES  # step size
    alpha = -1 / ((LAMBDA**2 - 2 / h**2) * h**2)  # from system discretization

    for i in tqdm.tqdm(range(MAX_ITERS)):
        u = jacobi_iteration(u, alpha)

    # plotting stuff
    fig, ax = plt.subplots()
    ax.plot(x.detach().numpy(), u.detach().numpy(), "b")
    ax.plot(x.detach().numpy(), torch.sin(LAMBDA * x).detach().numpy(), "r--")

    plt.show()
