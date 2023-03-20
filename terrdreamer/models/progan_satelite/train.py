import argparse
import wandb
from typing import List
from math import log2
import torch
from terrdreamer.dataset import AW3D30Dataset
from tqdm import tqdm
from terrdreamer.models.progan_satelite.progan_utils import (
    log_to_wandb,
    save_checkpoint,
    gradient_penalty,
)
from terrdreamer.models.progan_satelite.progan import Generator, Discriminator
from pathlib import Path


def trainer(
    generator,
    discriminator,
    aw3d30_loader,
    step,
    alpha,
    discriminator_optimizer,
    generator_optimizer,
    tensorboard_step,
    device,
    latent_size,
    batch_size,
    lambda_gp,
    progressive_epochs,
    epoch,
    curr_image_size,
):
    # Find the most appropriate device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (real, _) in tqdm(enumerate(aw3d30_loader), total=len(aw3d30_loader)):
        real = real.to(device)

        # images = real[0].to(device)
        # real = torch.randn(images.shape[0], latent_size).to(device)

        # Train Discriminator: max E[discriminator(real)] - E[discriminator(fake)] <-> min -E[discriminator(real)] + E[discriminator(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(real.size(0), latent_size, 1, 1).to(device).detach()

        discriminator.set_requires_grad(True)
        generator.set_requires_grad(False)
        discriminator.zero_grad()

        fake = generator(noise, alpha, step)

        discriminator_real = discriminator(real, alpha, step)
        discriminator_fake = discriminator(fake.detach(), alpha, step)
        gp = gradient_penalty(discriminator, real, fake, alpha, step, device=device)
        loss_discriminator = (
            -(torch.mean(discriminator_real) - torch.mean(discriminator_fake))
            + lambda_gp * gp
            + (0.001 * torch.mean(discriminator_real ** 2))
        )

        loss_discriminator.backward()
        discriminator_optimizer.step()

        # Train Generator: max E[discriminator(gen_fake)] <-> min -E[discriminator(gen_fake)]
        discriminator.set_requires_grad(False)
        generator.set_requires_grad(True)
        generator.zero_grad()

        fake = generator(noise, alpha, step)
        generator_fake = discriminator(fake, alpha, step)
        loss_generator = -torch.mean(generator_fake)

        loss_generator.backward()
        generator_optimizer.step()

        # Update alpha and ensure less than 1
        alpha += batch_size / ((progressive_epochs[step] * 0.5) * len(aw3d30_loader))
        alpha = min(alpha, 1)

    random_noise = torch.randn(6, latent_size, 1, 1).to(device)

    fixed_fakes = generator(random_noise, alpha, step) * 0.5 + 0.5

    # Log the images to wandb
    log_to_wandb(
        curr_image_size,
        loss_discriminator.item(),
        loss_generator.item(),
        real.detach(),
        fixed_fakes.detach(),
        epoch,
    )

    tensorboard_step += 1

    return tensorboard_step, alpha


def train(
    dataset: Path,
    lr: float,
    batch_size: List[int],
    img_channels: int,
    latent_size: int,
    in_channels: int,
    discriminator_iterations: int,
    lambda_gp: float,
    progressive_epochs: List[int],
    beta1: float,
    beta2: float,
):
    # Find the most appropriate device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Your training code here
    # Load the models to train on
    generator = Generator(latent_size, in_channels, img_channels).to(device)
    discriminator = Discriminator(in_channels, img_channels).to(device)

    # Load the optimizers & scalers - we'll stick with the ones used in the paper
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(beta1, beta2)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(beta1, beta2)
    )

    generator.train()
    discriminator.train()
    tensorboard_step = 0

    # Start at step that corresponds to img size that we set in config
    step = 0

    for num_epochs in progressive_epochs[step:]:
        alpha = 1e-5
        curr_image_size = 4 * (2 ** step)
        print(f"Current image size: {curr_image_size}")

        # Load the dataset
        aw3d30_loader = torch.utils.data.DataLoader(
            AW3D30Dataset(
                dataset,
                device,
                limit=1000,
                for_progan=True,
                new_size=curr_image_size,
            ),
            batch_size,
            shuffle=True,
        )

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha = trainer(
                generator,
                discriminator,
                aw3d30_loader,
                step,
                alpha,
                discriminator_optimizer,
                generator_optimizer,
                tensorboard_step,
                device,
                latent_size=latent_size,
                batch_size=batch_size,
                lambda_gp=lambda_gp,
                progressive_epochs=progressive_epochs,
                epoch=epoch,
                curr_image_size=curr_image_size,
            )

            # # Save the models
            # save_checkpoint(generator, generator_optimizer, f"generator_{step}.pth")
            # save_checkpoint(
            #     discriminator, discriminator_optimizer, f"discriminator_{step}.pth"
            # )

        step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # default values
    parser.add_argument("--dataset", type=Path, default="", help="Path to dataset")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--img-channels", type=int, default=3, help="Number of image channels"
    )
    parser.add_argument(
        "--latent-size", type=int, default=256, help="Latent vector size"
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=256,
        help="Number of input channels for the discriminator",
    )
    parser.add_argument(
        "--discriminator-iterations",
        type=int,
        default=1,
        help="Number of discriminator iterations",
    )
    parser.add_argument(
        "--lambda-gp",
        type=float,
        default=10,
        help="Weight of gradient penalty loss term",
    )
    parser.add_argument(
        "--progressive-epochs",
        nargs="+",
        type=int,
        default=[50, 60, 70, 100, 150, 200, 300],
        help="Progressive epoch list",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.0, help="Beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.99, help="Beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="progan-satelite",
        help="Wandb project name",
    )
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project=args.wandb_project, config=args)

    # Call the train function with the parsed arguments
    train(
        args.dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        img_channels=args.img_channels,
        latent_size=args.latent_size,
        in_channels=args.in_channels,
        discriminator_iterations=args.discriminator_iterations,
        lambda_gp=args.lambda_gp,
        progressive_epochs=args.progressive_epochs,
        beta1=args.beta1,
        beta2=args.beta2,
    )
