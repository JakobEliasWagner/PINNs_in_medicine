import matplotlib.pyplot as plt
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

LEFT_BOUNDARY = 0
RIGHT_BOUNDARY = 5

OMEGA = 2 / 3
SAMPLES = 100
MAX_ITERS = 10000


def get_ground_truth(x: torch.Tensor):
    length = RIGHT_BOUNDARY - LEFT_BOUNDARY
    return torch.ones(x.shape) - 1 / length * x


def generate_plot(x: torch.Tensor, u: torch.Tensor):
    # plotting stuff
    fig, ax = plt.subplots()
    ax.plot(x.detach().numpy(), u.detach().numpy(), "b")
    ax.plot(x.detach().numpy(), get_ground_truth(x).detach().numpy(), "r--")

    return fig


if __name__ == "__main__":
    t = torch.linspace(LEFT_BOUNDARY, RIGHT_BOUNDARY, SAMPLES)

    # sim
    u = torch.zeros((SAMPLES,))  # uniform grid
    u[0] = 1.0

    # assemble system matrix
    INV_D = torch.diag(1 / (-2.0) * torch.ones((SAMPLES - 2,)))
    LU = torch.diag(torch.ones((SAMPLES - 3,)), diagonal=1) + torch.diag(
        torch.ones((SAMPLES - 3,)), diagonal=-1
    )
    b = torch.zeros((SAMPLES - 2,))
    b[0] = -u[0]

    writer = SummaryWriter()

    plot_every = MAX_ITERS // 100
    for i in tqdm.tqdm(range(MAX_ITERS)):
        u[1:-1] = torch.matmul(INV_D, b - torch.matmul(LU, u[1:-1]))  # jacobi
        # u[1:-1] = OMEGA * torch.matmul(INV_D, b - torch.matmul(LU, u[1:-1])) + (1 - OMEGA) * u[1:-1]
        if i % plot_every == 0:
            writer.add_figure("solution", generate_plot(t, u), i)
    writer.close()
