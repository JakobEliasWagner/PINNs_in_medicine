"""
A IS NOT DIAGONALLY DOMINANT!!! DOES NOT CONVERGE
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

LAMBDA = 5
OMEGA = 2 / 3  # weighted jacobi relaxation parameter
LEFT_BOUNDARY = -np.pi / 2
RIGHT_BOUNDARY = np.pi / 2
SAMPLES = 1000
MAX_ITERS = 100000


def generate_plot(x: torch.Tensor, u: torch.Tensor):
    # plotting stuff
    fig, ax = plt.subplots()
    ax.plot(x.detach().numpy(), u.detach().numpy(), "b")
    ax.plot(x.detach().numpy(), torch.sin(LAMBDA * x).detach().numpy(), "r--")

    return fig


if __name__ == "__main__":
    x = torch.linspace(LEFT_BOUNDARY, RIGHT_BOUNDARY, SAMPLES)
    u = torch.zeros((SAMPLES,))  # uniform grid
    u[0] = np.sin(LAMBDA * LEFT_BOUNDARY)
    u[-1] = np.sin(LAMBDA * RIGHT_BOUNDARY)

    # assemble system matrix
    h = (RIGHT_BOUNDARY - LEFT_BOUNDARY) / SAMPLES  # step size
    INV_D = torch.diag(1 / ((LAMBDA**2) * (h**2) - 2) * torch.ones((SAMPLES - 2,)))
    LU = torch.diag(torch.ones((SAMPLES - 3,)), diagonal=1) + torch.diag(
        torch.ones((SAMPLES - 3,)), diagonal=-1
    )
    b = torch.zeros((SAMPLES - 2,))
    b[0] = -np.sin(LAMBDA * LEFT_BOUNDARY)
    b[-1] = -np.sin(LAMBDA * RIGHT_BOUNDARY)

    writer = SummaryWriter()

    plot_every = MAX_ITERS // 100
    for i in tqdm.tqdm(range(MAX_ITERS)):
        # u[1:-1] = torch.matmul(INV_D, b - torch.matmul(LU, u[1:-1]))  # jacobi
        u[1:-1] = (
            OMEGA * torch.matmul(INV_D, b - torch.matmul(LU, u[1:-1]))
            + (1 - OMEGA) * u[1:-1]
        )
        if i % plot_every == 0:
            writer.add_figure("solution", generate_plot(x, u), i)
    writer.close()
