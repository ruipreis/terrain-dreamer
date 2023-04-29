import random
import time
from pathlib import Path
from typing import Tuple

import torch
import torchvision
import wandb
from torchvision.transforms import RandomCrop
from tqdm import tqdm

from terrdreamer.dataset import AW3D30Dataset
from terrdreamer.models.infinity_grid import DeepFillV1


def random_bbox(
    height, width, min_scale_factor, max_scale_factor
) -> Tuple[int, int, int, int]:
    img_tensor = torch.ones(1, height, width)

    # Generate a random crop height
    crop_height = random.randint(
        int(height * min_scale_factor), int(height * max_scale_factor)
    )

    # Generate a random crop width
    crop_width = random.randint(
        int(width * min_scale_factor), int(width * max_scale_factor)
    )

    random_crop = RandomCrop(size=(crop_height, crop_width), pad_if_needed=True)
    top, left, _, _ = random_crop.get_params(img_tensor, (crop_height, crop_width))
    return top, left, crop_height, crop_width


def train(
    train_dataset: Path,
    test_dataset: Path,
    save_model_path: Path,
    device: str = "cuda",
    n_epochs: int = 300,
    limit: int = 1000,
    batch_size: int = 16,
    pretrained_generator_path: Path = None,
    pretrained_local_critic_path: Path = None,
    pretrained_global_critic_path: Path = None,
    lr: float = 1e-4,
    beta1: float = 0.5,
    beta2: float = 0.9,
    min_scale_factor: float = 0.3,
    max_scale_factor: float = 0.6,
):
    # Instantiate the dataset
    dataset = AW3D30Dataset(train_dataset, limit=limit)
    test_dataset = AW3D30Dataset(test_dataset, limit=limit // 10)

    # Sample a single image just to get the size of the image
    _, H, W = dataset[0][0].shape

    # Get the matching loader for the dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    deepfill_model = DeepFillV1()

    # If needed, load pretrained weights
    deepfill_model.load_pretrained_if_needed(
        pretrained_generator_path,
        pretrained_local_critic_path,
        pretrained_global_critic_path,
    )

    # Instantiate optimizer for the models, both generator and discriminator
    D_local_optimizer = torch.optim.Adam(
        deepfill_model.local_critic.parameters(), lr=lr, betas=(beta1, beta2)
    )
    D_global_optimizer = torch.optim.Adam(
        deepfill_model.global_critic.parameters(), lr=lr, betas=(beta1, beta2)
    )
    G_optimizer = torch.optim.Adam(
        deepfill_model.inpaint_generator.parameters(), lr=lr, betas=(beta1, beta2)
    )

    # Send the model to the device
    deepfill_model.to(device)

    # Train the model
    for epoch in range(n_epochs):
        print("Starting epoch", epoch, "...")
        loss_history = {}
        epoch_start_time = time.time()

        for x, _ in tqdm(loader, total=len(loader)):
            x = x.to(device)

            # Sample random bbox
            top, left, height, width = random_bbox(
                x.shape[2],
                x.shape[3],
                min_scale_factor=min_scale_factor,
                max_scale_factor=max_scale_factor,
            )
            bbox = (top, left, height, width)

            # Create mask
            mask = torch.zeros(x.size(0), 1, H, W, device=x.device)
            mask[:, :, top : top + height, left : left + width] = 1
            mask = mask.detach()

            # Generator Step (G)
            deepfill_model.prepare_generator_step()
            G_losses_dict = deepfill_model.step_generator(x, mask, bbox, G_optimizer)

            # Discriminator Step (D)
            deepfill_model.prepare_discriminator_step()
            D_losses_dict = deepfill_model.step_discriminator(
                x, mask, bbox, D_local_optimizer, D_global_optimizer
            )

            # Add everything to the loss history
            for k, v in D_losses_dict.items():
                r_k = "D_" + k
                loss_history.setdefault(r_k, [])
                loss_history[r_k].append(v)

            for k, v in G_losses_dict.items():
                r_k = "G_" + k
                loss_history.setdefault(r_k, [])
                loss_history[r_k].append(v)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Construct the loss message
        loss_message = {k: torch.mean(torch.tensor(v)) for k, v in loss_history.items()}
        loss_message["epoch_duration"] = epoch_duration

        with torch.no_grad():
            # Pick 8 random images from the test dataset
            test_images = random.sample(list(range(limit // 10)), 8)
            test_images = [test_dataset[i][0] for i in test_images]
            test_images = torch.stack(test_images).to(device)

            # Sample random bbox and create a mask
            top, left, height, width = random_bbox(
                test_images.shape[2],
                test_images.shape[3],
                min_scale_factor=min_scale_factor,
                max_scale_factor=max_scale_factor,
            )
            bbox = (top, left, height, width)
            mask = torch.zeros(test_images.size(0), 1, H, W, device=x.device)
            mask[:, :, top : top + height, left : left + width] = 1
            mask = mask.detach()

            # Apply the mask to the images
            masked_images = test_images * (1 - mask)
            x1, x2 = deepfill_model.eval(masked_images, mask)

            # Save the images
            grid = torchvision.utils.make_grid(
                torch.cat([masked_images, x1, x2], dim=0),
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
            )

            # Store the image in wandb
            loss_message["samples"] = wandb.Image(grid)

            # Save to file
            torchvision.utils.save_image(grid, f"epoch_{epoch}.png")

        # Ever so often, save the model and sample some images
        if epoch % 10 == 0:
            deepfill_model.save(save_model_path)

            # Perform further sampling just to get the images
            with torch.no_grad():
                pass

        wandb.log(loss_message)

    # Save the model
    deepfill_model.save(save_model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-model-path", type=Path, required=True)
    parser.add_argument("--wandb-project", type=str, required=True)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, config=args)

    train(
        args.train_dataset,
        args.test_dataset,
        args.save_model_path,
        n_epochs=args.n_epochs,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    wandb.finish()
