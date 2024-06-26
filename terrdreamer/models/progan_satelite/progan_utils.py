import torch
import torchvision
import wandb


def log_to_wandb(latent_size, loss_critic, loss_gen, real, fake, wandb_step, dem: bool):
    log_message = {
        f"Loss Discriminator": loss_critic,
        f"Loss Generator": loss_gen,
        f"Latent Size": latent_size,
    }

    if wandb_step % 10 == 0:
        if dem:
            # In case of DEM, then there's the need to call the to_gtif method
            img_grid_real = torchvision.utils.make_grid(
                real[:6], normalize=True, value_range=(-1, 1)
            )
            img_grid_fake = torchvision.utils.make_grid(
                fake[:6], normalize=True, value_range=(-1, 1)
            )
        else:
            # take out (up to) 8 examples to plot
            img_grid_real = torchvision.utils.make_grid(real[:6], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake[:6], normalize=True)

        log_message["Real"] = [wandb.Image(img_grid_real)]
        log_message["Fake"] = [wandb.Image(img_grid_fake)]

    wandb.log(log_message)


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    beta = torch.rand((real.size(0), 1, 1, 1)).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
