import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

LAMBDA = 5
LEFT_BOUNDARY = -np.pi / 2
RIGHT_BOUNDARY = np.pi / 2
PLOT_SAMPLES = 300


def Helmholtz1DLoss(
    gt: torch.Tensor,
    output: torch.Tensor,
    collocation_points: torch.Tensor,
    output_collocation: torch.Tensor,
    collocation_boundary: torch.Tensor,
    boundary_gt: torch.Tensor,
    writer: torch.utils.tensorboard.SummaryWriter,
    epoch: int,
    epoch_weight: float = 1.0,
):
    loss = nn.MSELoss()
    # common loss
    loss_common = loss(gt, output)

    # pde loss
    df_dx = torch.autograd.grad(
        output_collocation,
        collocation_points,
        grad_outputs=torch.ones_like(collocation_points),
        retain_graph=True,
        create_graph=True,
    )[0]
    d_df_dx_dx = torch.autograd.grad(
        df_dx,
        collocation_points,
        grad_outputs=torch.ones_like(collocation_points),
        retain_graph=True,
        create_graph=True,
    )[0]
    loss_pde = epoch_weight * loss(-d_df_dx_dx, LAMBDA**2 * output_collocation)

    # boundary loss
    output_boundary = model(collocation_boundary)
    loss_boundary = epoch_weight * loss(output_boundary, boundary_gt)

    writer.add_scalars(
        "losses",
        {
            "gt": loss_common.item(),
            "pde": loss_pde.item(),
            "boundary": loss_boundary.item(),
        },
        epoch,
    )

    return loss_pde + 1e3 * loss_boundary + loss_common


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.model = nn.Sequential(
            # ScaleLayer(LEFT_BOUNDARY, RIGHT_BOUNDARY),
            nn.Linear(1, 60),
            nn.Tanh(),
            nn.Linear(60, 60),
            nn.Tanh(),
            nn.Linear(60, 60),
            nn.Tanh(),
            nn.Linear(60, 60),
            nn.Tanh(),
            nn.Linear(60, 60),
            nn.Tanh(),
            nn.Linear(60, 1),
        )

    def forward(self, x):
        return self.model(x)


def ground_truth(x: torch.tensor) -> torch.tensor:
    return torch.sin(LAMBDA * x)


def generate_plot(
    x: torch.tensor,
    model: nn.Module,
    x_gt: torch.tensor,
    f_gt: torch.tensor,
    x_sample: torch.tensor,
    gt_sample: torch.tensor,
):
    # test model
    f_pred = model(x)

    # plot
    fig, ax = plt.subplots()
    ax.plot(x.detach().numpy(), f_pred.detach().numpy(), "b")  # prediction
    ax.plot(x_gt.detach().numpy(), f_gt.detach().numpy(), "r--")  # ground truth
    ax.plot(x_sample.detach().numpy(), gt_sample.detach().numpy(), ".")  # gt samples

    ax.set_ylim([min(f_gt.detach().numpy()), max(f_gt.detach().numpy())])

    return fig


def generate_fourier1D(
    x: torch.tensor,
    model: nn.Module,
):
    f_pred = model(x)
    # transform prediction
    ff = torch.fft.fft(f_pred).detach().numpy()

    T = 1 / ((RIGHT_BOUNDARY - LEFT_BOUNDARY) * PLOT_SAMPLES)  # sample frequency
    N = x.shape[0]  # samples
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    # plot
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(ff[: N // 2]))

    return fig


if __name__ == "__main__":
    x = torch.linspace(LEFT_BOUNDARY, RIGHT_BOUNDARY, PLOT_SAMPLES).reshape(-1, 1)
    gt = ground_truth(x)

    # draw random samples
    k = 5
    perm = torch.randperm(x.size(0))
    idx = perm[:k]
    x_sample = x[idx]
    gt_sample = gt[idx]

    # create collocation samples
    n = 5000
    collocation_points = (LEFT_BOUNDARY - RIGHT_BOUNDARY) * torch.rand(
        n, 1
    ) + RIGHT_BOUNDARY
    collocation_points.requires_grad = True

    n_b = 100
    collocation_boundary = torch.cat(
        (
            LEFT_BOUNDARY * torch.ones(n_b // 2, 1),
            RIGHT_BOUNDARY * torch.ones(n_b // 2, 1),
        )
    )
    collocation_boundary.requires_grad = True
    boundary_ground_truth = torch.sin(LAMBDA * collocation_boundary)

    # initialize model
    model = PINN()

    # train the network
    epochs = 1000
    optimizer = torch.optim.Adam(model.parameters())

    writer = SummaryWriter()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # get prediction and loss
        out_samples = model(x_sample)
        out_collocation = model(collocation_points)

        loss = Helmholtz1DLoss(
            gt=gt_sample,
            output=out_samples,
            collocation_points=collocation_points,
            output_collocation=out_collocation,
            collocation_boundary=collocation_boundary,
            boundary_gt=boundary_ground_truth,
            writer=writer,
            epoch=epoch,
        )
        loss.backward(retain_graph=True)

        optimizer.step()

        writer.add_scalar("Loss", loss.item(), epoch)
        if epoch % 10 == 0:
            img = generate_plot(x, model, x, gt, x_sample, gt_sample)
            writer.add_figure("prediction", img, epoch)

            img = generate_fourier1D(x, model)
            writer.add_figure("fourier", img, epoch)

    writer.close()
