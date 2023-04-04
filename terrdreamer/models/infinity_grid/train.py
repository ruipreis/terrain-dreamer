import torch
import random
from torchvision.transforms import RandomCrop
from typing import Tuple
from pathlib import Path
from terrdreamer.dataset import AW3D30Dataset
from terrdreamer.models.infinity_grid import DeepFillV1
import time
import wandb


def local_patch(x, bbox):
    """
    Crop local patch according to bbox.
    Args:
        x: input
        bbox: (top, left, height, width)
    Returns:
        torch.Tensor: local patch
    """
    top, left, height, width = bbox
    patch = x[:, :, top : top + height, left : left + width]
    return patch


def random_bbox(
    height, width, min_factor: float, max_factor: float
) -> Tuple[int, int, int, int]:
    img_tensor = torch.ones(1, height, width)
    crop_height = random.randint(int(height * min_factor), int(height * max_factor))
    crop_width = random.randint(int(width * min_factor), int(width * max_factor))
    random_crop = RandomCrop(size=(crop_height, crop_width), pad_if_needed=True)
    top, left, _, _ = random_crop.get_params(img_tensor, (crop_height, crop_width))
    return top, left, crop_height, crop_width


def train(
    train_dataset: Path,
    save_model_path: Path,
    device: str = "cpu",
    n_epochs: int = 300,
    limit: int = 1000,
    batch_size: int = 16,
    pretrained_generator_path: Path = None,
    pretrained_discriminator_path: Path = None,
    lr: float = 1e-4,
    beta1: float = 0.5,
    beta2: float = 0.9,
    min_factor: float = 0.3,
    max_factor: float = 0.6,
):
    # Instantiate the dataset
    dataset = AW3D30Dataset(train_dataset, limit=limit)

    # Get the matching loader for the dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    deepfill_model = DeepFillV1()

    # If needed, load pretrained weights
    deepfill_model.load_pretrained_if_needed(
        pretrained_generator_path,
        pretrained_discriminator_path,
    )

    # Instantiate optimizer for the models, both generator and discriminator
    D_optimizer = torch.optim.Adam(
        deepfill_model.critic.parameters(), lr=lr, betas=(beta1, beta2)
    )
    G_optimizer = torch.optim.Adam(
        deepfill_model.inpaint_generator.parameters(), lr=lr, betas=(beta1, beta2)
    )

    # Send the model to the device
    deepfill_model.to(device)

    # Train the model
    for epoch in range(n_epochs):
        loss_history = {}
        epoch_start_time = time.time()

        for x, _ in loader:
            x = x.to(device)

            # Sample random bbox
            top, left, height, width = random_bbox(
                x.shape[2], x.shape[3], min_factor, max_factor
            )

            # Create mask
            mask = torch.zeros_like(x)
            mask[:, :, top : top + height, left : left + width] = 1

            # Discriminator Step (D)
            deepfill_model.prepare_discriminator_step()
            D_losses_dict = deepfill_model.step_discriminator()

            # Generator Step (G)
            deepfill_model.prepare_generator_step()
            G_losses_dict = deepfill_model.step_generator()

            # Add everything to the loss history
            for k, v in D_losses_dict.items():
                r_k = "D_" + k
                loss_history.setdefault(r_k, [])
                loss_history[r_k].append(v)

            for k, v in G_losses_dict.items():
                r_k = "G_" + k
                loss_history.setdefault(r_k, [])
                loss_history[r_k].append(v)

        with torch.no_grad():
            # Now perform testing
            pass

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Construct the loss message
        loss_message = {k: torch.mean(torch.tensor(v)) for k, v in loss_history.items()}
        loss_message["epoch_duration"] = epoch_duration

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
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-model-path", type=Path, required=True)
    parser.add_argument("--wandb-project", type=str, required=True)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, config=args)

    train(
        args.train_dataset,
        args.save_model_path,
        n_epochs=args.n_epochs,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    wandb.finish()