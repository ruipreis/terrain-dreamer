import argparse

import torch
import torch.nn as nn

# Load the models to train on
from terrdreamer.models.image_to_dem.base_models import BasicDiscriminator, UNetGenerator
from terrdreamer.models.image_to_dem.losses import DiscriminatorLoss, GeneratorLoss

# Load the dataset
from terrdreamer.dataset import AW3D30Dataset, tiff_to_jpg

from pathlib import Path
from tqdm import tqdm
from PIL import Image

def train(
    dataset_path:Path, n_epochs:int=300, batch_size:int=8
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    aw3d30_loader = torch.utils.data.DataLoader(
        AW3D30Dataset(dataset_path, device, limit=10000), 
        batch_size=batch_size, 
        shuffle=True
    )

    # Load the models to train on
    discriminator = BasicDiscriminator()
    generator = UNetGenerator()

    # Load the losses
    discriminator_loss = DiscriminatorLoss()
    generator_loss = GeneratorLoss(loss_lambda=100.0)

    # Load the optimizers - we'll stick with the ones used in the paper
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)

    # Place the parts in the correct device
    discriminator = discriminator.to(device)
    generator = generator.to(device)


    # Train the models
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        for i, (sat_rgb_img, dem_img) in tqdm(enumerate(aw3d30_loader), total=len(aw3d30_loader)):
            # The generator is expected to produce a 256x256 DEM image, from the satellite image

            # Train the discriminator
            discriminator_optimizer.zero_grad()

            # Get the generated image
            generated_img = generator(sat_rgb_img)

            discriminator_real_output = discriminator(dem_img)
            discriminator_fake_output = discriminator(generated_img)

            # Calculate the loss
            discriminator_loss_value = discriminator_loss(
                discriminator_real_output, discriminator_fake_output
            )

            # Backpropagate the loss
            discriminator_loss_value.backward()
            discriminator_optimizer.step()

            # Train the generator
            generator_optimizer.zero_grad()

            # Can't use the generated image from before, because the backward pass
            # removes the generated image from the graph
            generated_img = generator(sat_rgb_img)

            discriminator_fake_output = discriminator(generated_img)

            # Calculate the loss
            generator_loss_value = generator_loss(
                discriminator_fake_output, generated_img, dem_img
                )

            # Backpropagate the loss
            generator_loss_value.backward()
            generator_optimizer.step()

            if i % 100 == 0:
                print(
                    f"Epoch {epoch} | Batch {i} | Discriminator loss: {discriminator_loss_value.item()} | Generator loss: {generator_loss_value.item()}"
                )

                # Save the models
                torch.save(
                    discriminator.state_dict(),
                    f"discriminator_{epoch}_{i}.pt",
                )

                torch.save(
                    generator.state_dict(),
                    f"generator_{epoch}_{i}.pt",
                )
                
                # Sample the output DEM to see if it makes any sense
                original_sat = sat_rgb_img[0].detach().cpu()
                
                # Unnormalize the satellite image
                original_sat = (original_sat*AW3D30Dataset._Satellite_std) + AW3D30Dataset._Satellite_mean
                
                original_dem = dem_img[0].detach().cpu()
                predicted_dem = generated_img[0].detach().cpu()
                
                # Convert the SAT image to a JPG
                Image.fromarray(original_sat.permute(1,2,0).numpy().astype("uint8")).save(f"original_sat_{epoch}_{i}.jpg")
                
                tiff_to_jpg(
                    original_dem,out_path=f"original_dem_{epoch}_{i}.jpg",convert=True
                )
                
                tiff_to_jpg(
                    predicted_dem,out_path=f"generated_dem_{epoch}_{i}.jpg",convert=True
                )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    # Start training
    train(args.dataset, n_epochs=args.n_epochs, batch_size=args.batch_size)