import wandb
from pathlib import Path
import torch
import time
from terrdreamer.dataset.cropped import CroppedDataset

from terrdreamer.models.infinity_grid import Masked_Pix2Pix
import random
from tqdm import tqdm


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
    ndf: int = 64,
    ngf: int = 64,
    sample_size: int = 10000,
    lambda_gp: float = 10,
    lambda_l1: float = 100,
    # Setup the data related to the input images
    min_crop_factor: float = 0.3,
    max_crop_factor: float = 0.6,
    min_mask_factor: float = 0.01,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    train_dataset = CroppedDataset(
        dataset_path,
        limit=sample_size,
        min_factor=min_crop_factor,
        max_factor=max_crop_factor,
        minimum_mask=min_mask_factor,
    )
    test_dataset = CroppedDataset(
        test_dataset_path,
        limit=sample_size // 10,
        min_factor=min_crop_factor,
        max_factor=max_crop_factor,
        minimum_mask=min_mask_factor,
    )

    aw3d30_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_aw3d30_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    # Load the model to train on
    pix2pix_model = Masked_Pix2Pix(
        lambda_l1=lambda_l1,
        ngf=ngf,
        ndf=ndf,
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
        print("Epoch: ", epoch)
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

        for y, x, mask in tqdm(aw3d30_loader, total=len(aw3d30_loader)):
            # Place the data on the GPU
            x = torch.cat([x, mask], dim=1)
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            # Discriminator Step (D)
            pix2pix_model.prepare_discriminator_step()
            d_loss, d_real_loss, d_fake_loss = pix2pix_model.step_discriminator(
                x, y, D_optimizer
            )

            # Generator Step (G)
            pix2pix_model.prepare_generator_step()
            g_loss, g_bce_loss, g_l1_loss = pix2pix_model.step_generator(
                x, y, G_optimizer, mask
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
            for y, x, mask in test_aw3d30_loader:
                # Place the data on the GPU
                x = torch.cat([x, mask], dim=1)
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)

                # Test the model
                _, test_d_loss, test_g_loss = pix2pix_model.test(x, y, mask)

                # Add everything to the loss history
                loss_history["test_D_loss"].append(test_d_loss)
                loss_history["test_G_loss"].append(test_g_loss)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        # Construct a log message
        log_message = {k: torch.mean(torch.tensor(v)) for k, v in loss_history.items()}
        log_message["per_epoch_ptime"] = per_epoch_ptime

        # Call torch cuda empty cache to free up memory
        torch.cuda.empty_cache()

        # Start testing the model on random images from the test dataset
        if epoch % 20 == 0:
            print("Saving model")
            pix2pix_model.save(epoch)

            # Use images at specific indexes to see if the model is learning anything,
            interest_indexes = random.sample(range(0, len(test_dataset)), 10)

            with torch.no_grad():
                original_sat_images = []
                maked_sat_images = []
                masks = []
                resulting_sat_images = []

                for i in interest_indexes:
                    # Get the image and the DEM
                    original_sat, masked_sat, mask = test_dataset[i]

                    # Concatenate the masked sat and mask
                    sat_masked_img = torch.cat((masked_sat, mask), dim=0)
                    sat_masked_img = sat_masked_img.unsqueeze(0).cuda()
                    gen_sat = pix2pix_model.generator(sat_masked_img)

                    # Now add the images to the list
                    original_sat_images.append(original_sat)
                    maked_sat_images.append(masked_sat)
                    masks.append(mask)
                    resulting_sat_images.append(gen_sat[0].detach().cpu())

                # Now log the images
                log_message["original_sat_images"] = [
                    wandb.Image(img) for img in original_sat_images
                ]
                log_message["masked_sat_images"] = [
                    wandb.Image(img) for img in maked_sat_images
                ]
                log_message["masks"] = [wandb.Image(img) for img in masks]
                log_message["resulting_sat_images"] = [
                    wandb.Image(img) for img in resulting_sat_images
                ]

        wandb.log(log_message)

    pix2pix_model.save(epoch)


if __name__ == "__main__":
    import argparse

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
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--wandb-project", type=str, required=True)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--min-crop-factor", type=float, default=0.3)
    parser.add_argument("--max-crop-factor", type=float, default=0.6)
    parser.add_argument("--min-mask-factor", type=float, default=0.1)
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
        ndf=args.ndf,
        ngf=args.ngf,
        sample_size=args.sample_size,
        lambda_gp=args.lambda_gp,
        lambda_l1=args.lambda_l1,
        min_crop_factor=args.min_crop_factor,
        max_crop_factor=args.max_crop_factor,
        min_mask_factor=args.min_mask_factor,
    )

    wandb.finish()
