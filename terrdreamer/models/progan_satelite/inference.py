from terrdreamer.models.progan_satelite.progan import Generator
import torch
import torchvision


def load_generator(checkpoint_path, device, latent_dim, n_channels):
    generator = Generator(latent_dim, latent_dim, n_channels)
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    return generator


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-img", type=str, default="out.png")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--n-channels", type=int, default=3)
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=7)
    args = parser.parse_args()

    G = load_generator(args.checkpoint, args.device, args.latent_dim, args.n_channels)
    random_noise = torch.randn(args.n_images, args.latent_dim, 1, 1, device=args.device)
    start_t = time.time()
    out = G(random_noise, 1, args.n_steps - 1) * 0.5 + 0.5
    end_t = time.time()
    print("Time taken:", end_t - start_t, "s")

    # Now create a grid and save it
    grid = torchvision.utils.make_grid(out, nrow=5)
    torchvision.utils.save_image(grid, args.out_img)