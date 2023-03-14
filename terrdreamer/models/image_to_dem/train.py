import argparse

import torch
import torch.nn as nn

# Load the models to train on
from terrdreamer.models.image_to_dem.base_models import BasicDiscriminator, UNetGenerator

# Load the dataset
from terrdreamer.dataset import AW3D30Dataset, tiff_to_jpg, RGB_MEAN, RGB_STD

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import time
from torch.autograd import Variable


def train(
    dataset_path:Path, test_dataset_path:Path, pretrained_generator_path,pretrained_discriminator_path,n_epochs:int=300, batch_size:int=8, beta1:float=.5, beta2:float=.999,lr:float=2e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    train_dataset = AW3D30Dataset(dataset_path, limit=1000)
    test_dataset = AW3D30Dataset(test_dataset_path, limit=100)
    
    aw3d30_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_aw3d30_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Load the models to train on
    G = UNetGenerator(3, 1)
    D = BasicDiscriminator(3+1) # 3 channels for the satellite image, 1 for the DEM
    
    # If possible, start training from a pretrained model
    if pretrained_generator_path is not None:
        G.load_state_dict(torch.load(pretrained_generator_path))
    else:
        # Initialize the weights to have mean 0 and standard deviation 0.02
        G.weight_init(mean=0.0, std=0.02)
        
    if pretrained_discriminator_path is not None:
        D.load_state_dict(torch.load(pretrained_discriminator_path))
    else:
        D.weight_init(mean=0.0, std=0.02)    
    
    # Load the losses - we'll stick with the ones used in the paper
    BCE_loss = nn.BCELoss().cuda()
    L1_loss = nn.L1Loss().cuda()

    # Load the optimizers - we'll stick with the ones used in the paper
    G_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1,beta2))

    # Place the parts in the correct device
    D = D.to(device)
    G = G.to(device)
    
    # Keep track of the training losses
    train_history = {
        "D_losses": [],
        "G_losses": [],
        "per_epoch_ptimes": [],
    }

    # Train the models
    for epoch in range(n_epochs):
        D_losses = []
        G_losses = []
        
        epoch_start_time = time.time()
        
        for i, (x_, y_) in tqdm(enumerate(aw3d30_loader), total=len(aw3d30_loader)):
            # The generator is expected to produce a 256x256 DEM image, from the satellite image

            # Train the discriminator (D)
            D.zero_grad()

            x_,y_ = Variable(x_.cuda()), Variable(y_.cuda())
            
            D_result = D(x_, y_).squeeze()
            
            D_real_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda()))
            
            G_result = G(x_)
            D_result = D(x_, G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, Variable(torch.zeros(D_result.size()).cuda()))
            
            D_train_loss = (D_real_loss + D_fake_loss)*0.5
            D_train_loss.backward()
            D_optimizer.step()
            
            # Log the discriminator loss
            val_D_train_loss = D_train_loss.item()
            train_history["D_losses"].append(val_D_train_loss)
            D_losses.append(val_D_train_loss)
            
            # Train the generator (G)
            G.zero_grad()
            
            G_result = G(x_)
            D_result = D(x_, G_result).squeeze()
            
            G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda())) + L1_loss(G_result, y_)*100
            G_train_loss.backward()
            G_optimizer.step()
            
            # Log the generator loss
            val_G_train_loss = G_train_loss.item()
            train_history["G_losses"].append(val_G_train_loss)
            G_losses.append(val_G_train_loss)
        

        # Run the model in the test dataset
        print("Running the model in the test dataset")
        
        D.eval()
        G.eval()
        
        with torch.no_grad():
            test_D_losses = []
            test_G_losses = []
            
            for i, (x_, y_) in tqdm(enumerate(test_aw3d30_loader), total=len(test_aw3d30_loader)):
                x_,y_ = Variable(x_.cuda()), Variable(y_.cuda())
                D_result = D(x_, y_).squeeze()
            
                D_real_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda()))
                
                G_result = G(x_)
                D_result = D(x_, G_result).squeeze()
                D_fake_loss = BCE_loss(D_result, Variable(torch.zeros(D_result.size()).cuda()))
                
                D_tet_loss = (D_real_loss + D_fake_loss)*0.5
                
                test_D_losses.append(D_tet_loss.item())
                
                # Now the generator
                G_result = G(x_)
                D_result = D(x_, G_result).squeeze()
                G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda())) + L1_loss(G_result, y_)*100

                test_G_losses.append(G_train_loss.item())
                
        D.train()
        G.train()
            
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        
        print('[%d/%d] - ptime: %.2f, train_loss_d: %.3f, train_loss_g: %.3f, test_loss_d: %.3f, test_loss_g: %.3f' % (
            (epoch + 1), n_epochs, per_epoch_ptime, 
            torch.mean(torch.FloatTensor(D_losses)),
            torch.mean(torch.FloatTensor(G_losses)),
            torch.mean(torch.FloatTensor(test_D_losses)),
            torch.mean(torch.FloatTensor(test_G_losses))
            ))

        if epoch % 20 == 0:
            print("Saving the models after epoch %i" % epoch)

            # Save the models
            torch.save(
                D.state_dict(),
                f"discriminator_{epoch}.pt",
            )

            torch.save(
                G.state_dict(),
                f"generator_{epoch}.pt",
            )
        
            
            G.eval()
            
            print("Sampling the images after epoch %i" % epoch)
            # Use images at specific indexes to see if the model is learning anything,
            interest_indexes = [0, 1, 2, 3]
            
            with torch.no_grad():
                for i in interest_indexes:
                    sat_rgb_img, dem_img = test_dataset[i]
                    
                    sat_rgb_img = Variable(sat_rgb_img.unsqueeze(0).cuda())
                    dem_img = Variable(dem_img.unsqueeze(0).cuda())
                    
                    gen_result = G(sat_rgb_img)

                    # Sample the output DEM to see if it makes any sense
                    original_sat = sat_rgb_img[0].detach().cpu()
                    
                    # Unnormalize the satellite image
                    original_sat = (original_sat+1)*127.5
                        
                    original_dem = dem_img[0].detach().cpu()
                    predicted_dem = gen_result[0].detach().cpu()
                    
                    # Convert the SAT image to a JPG
                    Image.fromarray(original_sat.permute(1,2,0).numpy().astype("uint8")).save(f"original_sat_{epoch}_{i}.jpg")
                    
                    tiff_to_jpg(
                        original_dem,out_path=f"original_dem_{epoch}_{i}.jpg",convert=True
                    )
                    
                    tiff_to_jpg(
                        predicted_dem,out_path=f"generated_dem_{epoch}_{i}.jpg",convert=True
                    )
                    
                    
            G.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pretrained-generator", type=Path, default=None)
    parser.add_argument("--pretrained-discriminator", type=Path, default=None)
    args = parser.parse_args()

    if args.pretrained_generator is not None:
        pretrained_generator_path = args.pretrained_generator
    else:
        pretrained_generator_path = None
        
    if args.pretrained_discriminator is not None:
        pretrained_discriminator_path = args.pretrained_discriminator
    else:
        pretrained_discriminator_path = None
        
    # Start training
    train(args.train_dataset, args.test_dataset, pretrained_generator_path,pretrained_discriminator_path,n_epochs=args.n_epochs, batch_size=args.batch_size,lr=args.lr)