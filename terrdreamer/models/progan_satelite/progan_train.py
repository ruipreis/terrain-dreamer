import torch
import torch.nn as nn

# Load the model
from progan import Generator, Discriminator

# Load the dataset
from terrdreamer.dataset import AW3D30Dataset, tiff_to_jpg
from pathlib import Path
from tqdm import tqdm

# Load configs & utils
import progan_config
from progan_utils import gradient_penalty, save_checkpoint, plot_to_tensorboard, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from math import log2

# Improves training speed
torch.backends.cudnn.benchmark = True


def trainer(
    generator,
    discriminator,
    aw3d30_loader,
    step,
    alpha,
    discriminator_optimizer,
    generator_optimizer,
    discriminator_scaler,
    generator_scaler,
    tensorboard_step,
    writer,
    curr_image_size,
    curr_batch_size,
):

    for i, (real, _) in tqdm(enumerate(aw3d30_loader), total=len(aw3d30_loader)):

        real = real.to(progan_config.DEVICE)

        # images = real[0].to(progan_config.DEVICE)
        # real = torch.randn(images.shape[0], progan_config.LATENT_SIZE).to(progan_config.DEVICE)

        # Train Discriminator: max E[discriminator(real)] - E[discriminator(fake)] <-> min -E[discriminator(real)] + E[discriminator(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(curr_batch_size, progan_config.LATENT_SIZE, 1, 1).to(
            progan_config.DEVICE
        )

        with torch.cuda.amp.autocast():
            fake = generator(noise, alpha, step)
            discriminator_real = discriminator(real, alpha, step)
            discriminator_fake = discriminator(fake.detach(), alpha, step)
            gp = gradient_penalty(
                discriminator,
                real,
                fake,
                alpha,
                step,
                curr_batch_size=curr_batch_size,
                device=progan_config.DEVICE,
            )
            loss_discriminator = (
                -(torch.mean(discriminator_real) - torch.mean(discriminator_fake))
                + progan_config.LAMBDA_GP * gp
                + (0.001 * torch.mean(discriminator_real**2))
            )

        discriminator_optimizer.zero_grad()
        discriminator_scaler.scale(loss_discriminator).backward()
        discriminator_scaler.step(discriminator_optimizer)
        discriminator_scaler.update()

        # Train Generator: max E[discriminator(gen_fake)] <-> min -E[discriminator(gen_fake)]
        with torch.cuda.amp.autocast():
            generator_fake = discriminator(fake, alpha, step)
            loss_generator = -torch.mean(generator_fake)

        generator_optimizer.zero_grad()
        generator_scaler.scale(loss_generator).backward()
        generator_scaler.step(generator_optimizer)
        generator_scaler.update()

        # Update alpha and ensure less than 1
        alpha += curr_batch_size / (
            (progan_config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(aw3d30_loader)
        )
        alpha = min(alpha, 1)

        if i % 500 == 0:
            with torch.no_grad():
                fixed_fakes = (
                    generator(progan_config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
                )
            plot_to_tensorboard(
                writer,
                loss_discriminator.item(),
                loss_generator.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1

    return tensorboard_step, alpha


def main():

    # Load the models to train on
    generator = Generator(
        progan_config.LATENT_SIZE, progan_config.IN_CHANNELS, progan_config.IMG_CHANNELS
    ).to(progan_config.DEVICE)
    discriminator = Discriminator(
        progan_config.IN_CHANNELS, progan_config.IMG_CHANNELS
    ).to(progan_config.DEVICE)

    # Load the optimizers & scalers - we'll stick with the ones used in the paper
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=progan_config.LEARNING_RATE, betas=(0.0, 0.99)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=progan_config.LEARNING_RATE, betas=(0.0, 0.99)
    )
    generator_scaler = torch.cuda.amp.GradScaler()
    discriminator_scaler = torch.cuda.amp.GradScaler()

    # Launch Tensorboard
    writer = SummaryWriter(f"logs/gan1")

    if progan_config.LOAD_MODEL:
        load_checkpoint(
            progan_config.CHECKPOINT_GENERATOR,
            generator,
            generator_optimizer,
            progan_config.LEARNING_RATE,
        )
        load_checkpoint(
            progan_config.CHECKPOINT_DISCRIMINATOR,
            discriminator,
            discriminator_optimizer,
            progan_config.LEARNING_RATE,
        )

    generator.train()
    discriminator.train()
    tensorboard_step = 0

    # Start at step that corresponds to img size that we set in config
    step = int(log2(progan_config.START_TRAIN_AT_IMG_SIZE / 4))

    for num_epochs in progan_config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        curr_image_size = 4 * 2**step
        print(f"Current image size: {curr_image_size}")

        curr_batch_size = progan_config.BATCH_SIZE
        # Load the dataset
        aw3d30_loader = torch.utils.data.DataLoader(
            AW3D30Dataset(
                progan_config.DATASET,
                progan_config.DEVICE,
                limit=10000,
                for_progan=True,
                new_size=curr_image_size,
            ),
            batch_size=curr_batch_size,
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
                discriminator_scaler,
                generator_scaler,
                tensorboard_step,
                writer,
                curr_image_size,
                curr_batch_size,
            )

            # Save the models
            save_checkpoint(
                generator, generator_optimizer, progan_config.CHECKPOINT_GENERATOR
            )
            save_checkpoint(
                discriminator,
                discriminator_optimizer,
                progan_config.CHECKPOINT_DISCRIMINATOR,
            )

        step += 1


if __name__ == "__main__":
    main()
