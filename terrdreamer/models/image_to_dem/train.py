import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

import wandb

# Load the dataset
from terrdreamer.dataset import AW3D30Dataset

# Load the models to train on
from terrdreamer.models.image_to_dem import DEM_Pix2Pix

LAMBDA = 100


def train(
    dataset_path: Path,
    test_dataset_path: Path,
    pretrained_generator_path,
    pretrained_discriminator_path,
    n_epochs: int = 300,
    batch_size: int = 8,
    beta1: float = 0.5,
    beta2: float = 0.999,
    lr: float = 2e-4,
    dem_to_image: bool = False,
    ndf: int = 64,
    ngf: int = 64,
    label_smoothing: bool = True,
    label_smoothing_factor: float = 0.1,
    sample_size: int = 10000,
    use_transforms: bool = True,
    loss: str = "vanilla",
    lambda_gp: float = 10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    train_dataset = AW3D30Dataset(
        dataset_path, swap=dem_to_image, transforms=use_transforms, limit=sample_size
    )
    test_dataset = AW3D30Dataset(
        test_dataset_path, swap=dem_to_image, limit=sample_size // 10
    )

    aw3d30_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_aw3d30_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    # Load the model to train on
    if dem_to_image:
        pix2pix_model = DEM_Pix2Pix(
            1,
            3,
            lambda_l1=LAMBDA,
            ngf=ngf,
            ndf=ndf,
            label_smoothing=label_smoothing,
            label_smoothing_factor=label_smoothing_factor,
            loss=loss,
            lambda_gp=lambda_gp,
        )
    else:
        pix2pix_model = DEM_Pix2Pix(
            3,
            1,
            lambda_l1=LAMBDA,
            ngf=ngf,
            ndf=ndf,
            label_smoothing=label_smoothing,
            label_smoothing_factor=label_smoothing_factor,
            loss=loss,
            lambda_gp=lambda_gp,
        )

    # Initialize the weights to have mean 0 and standard deviation 0.02
    pix2pix_model.weight_init(mean=0.0, std=0.02)

    # If possible, start training from a pretrained model
    if pretrained_discriminator_path is not None:
        pix2pix_model.load_discriminator(pretrained_discriminator_path)

    if pretrained_generator_path is not None:
        pix2pix_model.load_generator(pretrained_generator_path)

    # Load the optimizers - we'll stick with the ones used in the paper
    D_optimizer = torch.optim.Adam(
        pix2pix_model.discriminator.parameters(), lr=lr, betas=(beta1, beta2)
    )
    G_optimizer = torch.optim.Adam(
        pix2pix_model.generator.parameters(), lr=lr, betas=(beta1, beta2)
    )

    # Place to the model on the GPU
    pix2pix_model.to(device)

    # Train the models
    for epoch in range(n_epochs):
        loss_history = {
            "D_loss": [],
            "D_real_loss": [],
            "D_fake_loss": [],
            "G_loss": [],
            "G_bce_loss": [],
            "G_l1_loss": [],
            "test_D_loss": [],
            "test_G_loss": [],
        }

        epoch_start_time = time.time()

        for x, y in aw3d30_loader:
            # Place the data on the GPU
            x = x.to(device)
            y = y.to(device)

            # Discriminator Step (D)
            pix2pix_model.prepare_discriminator_step()
            d_loss, d_real_loss, d_fake_loss = pix2pix_model.step_discriminator(
                x, y, D_optimizer
            )

            # Generator Step (G)
            pix2pix_model.prepare_generator_step()
            g_loss, g_bce_loss, g_l1_loss = pix2pix_model.step_generator(
                x, y, G_optimizer
            )

            # Add everything to the loss history
            loss_history["D_loss"].append(d_loss)
            loss_history["D_real_loss"].append(d_real_loss)
            loss_history["D_fake_loss"].append(d_fake_loss)
            loss_history["G_loss"].append(g_loss)

            if g_bce_loss is not None:
                loss_history["G_bce_loss"].append(g_bce_loss)

            loss_history["G_l1_loss"].append(g_l1_loss)

        with torch.no_grad():
            for x, y in test_aw3d30_loader:
                # Place the data on the GPU
                x = x.to(device)
                y = y.to(device)

                # Test the model
                _, test_d_loss, test_g_loss = pix2pix_model.test(x, y)

                # Add everything to the loss history
                loss_history["test_D_loss"].append(test_d_loss)
                loss_history["test_G_loss"].append(test_g_loss)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        # Construct a log message
        log_message = {k: torch.mean(torch.tensor(v)) for k, v in loss_history.items()}
        log_message["per_epoch_ptime"] = per_epoch_ptime

        # Start testing the model on random images from the test dataset
        if epoch % 10 == 0:
            pix2pix_model.save(epoch)

            # Use images at specific indexes to see if the model is learning anything,
            interest_indexes = random.sample(range(0, len(test_dataset)), 10)

            with torch.no_grad():
                src_imgs = []
                src_dems = []
                gen_dems = []

                for i in interest_indexes:
                    # Get the image and the DEM
                    sat_rgb_img, dem_img = test_dataset[i]

                    sat_rgb_img = sat_rgb_img.unsqueeze(0).cuda()
                    dem_img = dem_img.unsqueeze(0).cuda()

                    gen_result = pix2pix_model.generator(sat_rgb_img)

                    # Sample the output DEM to see if it makes any sense
                    original_sat = sat_rgb_img[0].detach().cpu()

                    original_dem = dem_img[0].detach().cpu()
                    predicted_dem = gen_result[0].detach().cpu()

                    if dem_to_image:
                        # Get the unnormalized image
                        src_imgs.append(train_dataset.to_gtif(original_sat))
                        src_dems.append(train_dataset.to_img(original_dem))
                        gen_dems.append(train_dataset.to_img(predicted_dem))
                    else:
                        src_imgs.append(train_dataset.to_img(original_sat))
                        src_dems.append(train_dataset.to_gtif(original_dem))
                        gen_dems.append(train_dataset.to_gtif(predicted_dem))

                if dem_to_image:
                    log_message["src_dems"] = [wandb.Image(img) for img in src_imgs]
                    log_message["src_imgs"] = [wandb.Image(img) for img in src_dems]
                    log_message["gen_imgs"] = [wandb.Image(img) for img in gen_dems]
                else:
                    log_message["src_imgs"] = [wandb.Image(img) for img in src_imgs]
                    log_message["src_dems"] = [wandb.Image(img) for img in src_dems]
                    log_message["gen_dems"] = [wandb.Image(img) for img in gen_dems]

        wandb.log(log_message)

    pix2pix_model.save(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--pretrained-generator", type=Path, default=None)
    parser.add_argument("--pretrained-discriminator", type=Path, default=None)
    parser.add_argument("--dem-to-image", action="store_true")
    parser.add_argument("--label-smoothing", action="store_true")
    parser.add_argument("--use-transforms", action="store_true")
    parser.add_argument("--label-smoothing-factor", type=float, default=0.2)
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--wandb-project", type=str, required=True)
    parser.add_argument("--loss", choices=["vanilla", "wgangp"], default="vanilla")
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    args = parser.parse_args()

    if args.pretrained_generator is not None:
        pretrained_generator_path = args.pretrained_generator
    else:
        pretrained_generator_path = None

    if args.pretrained_discriminator is not None:
        pretrained_discriminator_path = args.pretrained_discriminator
    else:
        pretrained_discriminator_path = None

    wandb.init(project=args.wandb_project, config=args)

    # Start training
    train(
        args.train_dataset,
        args.test_dataset,
        pretrained_generator_path,
        pretrained_discriminator_path,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dem_to_image=args.dem_to_image,
        ndf=args.ndf,
        ngf=args.ngf,
        label_smoothing=args.label_smoothing,
        label_smoothing_factor=args.label_smoothing_factor,
        sample_size=args.sample_size,
        use_transforms=args.use_transforms,
        loss=args.loss,
        lambda_gp=args.lambda_gp,
    )

    wandb.finish()
