import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

N = 63
num_plots = 240
epochs = 4500


# TODO scale solutions the same way!


def generate_plots(x: torch.Tensor, epoch: int):
    vals = x.detach().clone()
    vals = torch.reshape(vals, (N, N))

    F = torch.fft.fft2(vals)
    F = torch.log10(torch.abs(F))

    fig, ax = plt.subplots()
    values = F.detach().numpy()
    scale = max(np.abs(values.min()), np.abs(values.max()))
    ax.imshow(values, vmin=-scale, vmax=scale, cmap="binary")
    plt.savefig(f"plots/fourier/fourier_{epoch}.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    values = vals.detach().numpy()
    scale = max(np.abs(values.min()), np.abs(values.max()))
    ax.imshow(values, vmin=-scale, vmax=scale, cmap="bwr")
    plt.savefig(f"plots/pred/res_{epoch}.png")
    plt.close(fig)


if __name__ == "__main__":
    h_x = 1 / (N + 1)
    h_y = 1 / (N + 1)

    alpha = -2 * (1 / h_x**2 + 1 / h_y**2)
    beta = 1 / h_x**2
    gamma = 1 / h_y**2

    # generate A
    diag_matrix = (
        alpha * torch.eye(N)
        + beta * torch.diag(torch.ones((N - 1)), 1)
        + beta * torch.diag(torch.ones((N - 1)), -1)
    )
    A = gamma * (
        torch.diag(torch.ones(N**2 - N), N) + torch.diag(torch.ones(N**2 - N), -N)
    )
    for i in range(N):
        start_row = i * N
        start_col = i * N
        end_row = start_row + N
        end_col = start_col + N
        A[start_row:end_row, start_col:end_col] = diag_matrix

    # generate b
    b = torch.zeros((N**2,))
    for i in range(1, N):
        for j in range(1, N):
            x = i / (N + 1)
            y = j / (N + 1)
            b[j + i * N] = (
                -2
                * np.pi**2
                * np.sin(np.pi * x)
                * np.sin(4 * np.pi * x)
                * np.sin(np.pi * y)
            )

    D_inv = 1 / alpha * torch.eye(N**2)
    LU = A - alpha * torch.eye(N**2)

    x = torch.rand((N**2))
    for i in tqdm(range(epochs)):
        x = torch.matmul(D_inv, b - torch.matmul(LU, x))
        if i % (epochs // num_plots) == 0:
            generate_plots(x, i)
            print(f"Residual: {torch.abs(torch.mean(torch.matmul(A, x) - b)):.4f}")

    x = torch.reshape(x, (N, N)).detach().numpy()

    plt.imshow(x)
    plt.show()
