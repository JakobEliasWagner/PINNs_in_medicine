import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

N = 100
num_plots = 240
epochs = 1000


class AdaptiveSwish(nn.Module):
    """
    Swish activation function defined as f(x) = x * sigma(beta * x)
    """

    def __init__(self):
        super(AdaptiveSwish, self).__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, input):
        return input * torch.sigmoid(self.beta * input)


class PINN2D(nn.Module):
    def __init__(self, model):
        super(PINN2D, self).__init__()
        self.model = model

    def forward(self, x, y):
        data = torch.stack([x, y], dim=1)
        return self.model(data)


def dirichlet_bc_loss(current: torch.Tensor):
    return torch.mean(torch.square(current))


def heat_pde_loss(current: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    # gradients
    u_x = torch.autograd.grad(
        current,
        x,
        grad_outputs=torch.ones_like(current),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_y = torch.autograd.grad(
        current,
        y,
        grad_outputs=torch.ones_like(current),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_yy = torch.autograd.grad(
        u_y,
        y,
        grad_outputs=torch.ones_like(u_y),
        retain_graph=True,
        create_graph=True,
    )[0]
    residual = (
        u_xx
        + u_yy
        + 2
        * np.pi**2
        * torch.sin(np.pi * x)
        * torch.sin(4 * np.pi * x)
        * torch.sin(np.pi * y)
    )
    return torch.mean(torch.square(residual))


def plot_collocations(x, y):
    fig, ax = plt.subplots()
    ax.plot(
        x["domain"].detach().numpy(), y["domain"].detach().numpy(), "rx", label="domain"
    )
    ax.plot(
        x["bc_top"].detach().numpy(), y["bc_top"].detach().numpy(), ".", label="top"
    )
    ax.plot(
        x["bc_left"].detach().numpy(), y["bc_left"].detach().numpy(), ".", label="left"
    )
    ax.plot(
        x["bc_bottom"].detach().numpy(),
        y["bc_bottom"].detach().numpy(),
        ".",
        label="bottom",
    )
    ax.plot(
        x["bc_right"].detach().numpy(),
        y["bc_right"].detach().numpy(),
        ".",
        label="right",
    )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    plt.show()


def generate_plots(model: nn.Module, epoch: int):
    N_plot = 63
    x_plot, y_plot = torch.meshgrid(
        torch.linspace(0, 1, N_plot), torch.linspace(0, 1, N_plot), indexing="ij"
    )
    x_plot, y_plot = torch.flatten(x_plot), torch.flatten(y_plot)

    pred_plot = model(x_plot, y_plot)

    pred_plot = pred_plot.reshape((N_plot, N_plot))

    fig, ax = plt.subplots()
    values = pred_plot.detach().numpy()
    scale = max(np.abs(values.min()), np.abs(values.max()))
    ax.imshow(values, vmin=-scale, vmax=scale, cmap="bwr")
    plt.savefig(f"plots/pred_pinn/pred_{epoch}.png")
    plt.close(fig)

    F = torch.fft.fft2(pred_plot)
    F = torch.log10(torch.abs(F))

    fig, ax = plt.subplots()
    values = F.detach().numpy()
    scale = max(np.abs(values.min()), np.abs(values.max()))
    ax.imshow(values, vmin=-scale, vmax=scale, cmap="binary")
    plt.savefig(f"plots/fourier_pinn/fourier_{epoch}.png")
    plt.close(fig)


def generate_data_plot(model: nn.Module, x_data: torch.Tensor, y_data: torch.Tensor):
    N_plot = 63
    x_plot, y_plot = torch.meshgrid(
        torch.linspace(0, 1, N_plot), torch.linspace(0, 1, N_plot), indexing="ij"
    )
    x_plot, y_plot = torch.flatten(x_plot), torch.flatten(y_plot)

    pred_plot = model(x_plot, y_plot)

    pred_plot = pred_plot.reshape((N_plot, N_plot))

    fig, ax = plt.subplots()
    values = pred_plot.detach().numpy()
    scale = max(np.abs(values.min()), np.abs(values.max()))
    ax.imshow(
        values, vmin=-scale, vmax=scale, cmap="bwr", extent=[0, 1, 0, 1], alpha=0.6
    )
    ax.plot(x_data.detach().numpy(), y_data.detach().numpy(), "kX")
    plt.savefig(f"plots/fit/pred_{epoch}.png")
    plt.close(fig)


if __name__ == "__main__":
    x = {
        "domain": torch.rand((N,)),
        "bc_top": torch.rand((N // 4,)),
        "bc_left": torch.zeros((N // 4,)),
        "bc_bottom": torch.rand((N // 4,)),
        "bc_right": torch.ones((N // 4,)),
    }

    y = {
        "domain": torch.rand((N,)),
        "bc_top": torch.ones((N // 4,)),
        "bc_left": torch.rand((N // 4,)),
        "bc_bottom": torch.zeros((N // 4,)),
        "bc_right": torch.rand((N // 4,)),
    }

    x_data = np.load("x_correct.npy")
    y_data = np.load("y_correct.npy")
    u_data = np.load("u_correct.npy")
    np.random.seed(42)
    idx = np.random.choice(np.arange(len(x_data)), 5, replace=False)
    x_data = torch.Tensor(x_data[idx])
    y_data = torch.Tensor(y_data[idx])
    u_data = torch.Tensor(u_data[idx])

    data = {}
    for key in x:
        data[key] = torch.stack([x[key], y[key]], dim=1)
        data[key].requires_grad = True
        x[key].requires_grad = True
        y[key].requires_grad = True

    plot_collocations(x, y)

    hidden_size = 100
    model = PINN2D(
        nn.Sequential(
            nn.Linear(2, hidden_size),
            AdaptiveSwish(),
            nn.Linear(hidden_size, hidden_size),
            AdaptiveSwish(),
            nn.Linear(hidden_size, hidden_size),
            AdaptiveSwish(),
            nn.Linear(hidden_size, hidden_size),
            AdaptiveSwish(),
            nn.Linear(hidden_size, 1),
        )
    )

    pde_loss = heat_pde_loss
    loss_top = dirichlet_bc_loss
    loss_left = dirichlet_bc_loss
    loss_bottom = dirichlet_bc_loss
    loss_right = dirichlet_bc_loss

    optim = torch.optim.Adam(model.parameters())

    for epoch in tqdm(range(epochs)):
        optim.zero_grad()

        pred_pde = model(x["domain"], y["domain"])
        loss_pde = pde_loss(pred_pde, x["domain"], y["domain"])

        pred_top_bc = model(x["bc_top"], y["bc_top"])
        loss_top_bc = loss_top(pred_top_bc)

        pred_left_bc = model(x["bc_left"], y["bc_left"])
        loss_left_bc = loss_left(pred_left_bc)

        pred_bottom_bc = model(x["bc_bottom"], y["bc_bottom"])
        loss_bottom_bc = loss_bottom(pred_bottom_bc)

        pred_right_bc = model(x["bc_right"], y["bc_right"])
        loss_right_bc = loss_right(pred_right_bc)

        pred_data = model(x_data, y_data)
        loss_data = torch.mean(torch.square(pred_data - u_data))

        loss_bc = loss_top_bc + loss_left_bc + loss_bottom_bc + loss_right_bc
        loss = loss_data  # loss_pde + 1e+3 * loss_bc + loss_data
        loss.backward()

        optim.step()
        if epoch % (epochs // num_plots) == 0:
            generate_data_plot(model, x_data, y_data)
            continue
            generate_plots(model, epoch)

    # get a prediction
    x_plot, y_plot = torch.meshgrid(
        torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), indexing="ij"
    )
    x_plot, y_plot = torch.flatten(x_plot), torch.flatten(y_plot)

    pred_plot = model(x_plot, y_plot)

    pred_plot = pred_plot.reshape((100, 100))

    fig, ax = plt.subplots()
    ax.imshow(pred_plot.detach().numpy())
    plt.show()
