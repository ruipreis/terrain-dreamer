import torch
from math import log2
from pathlib import Path

START_TRAIN_AT_IMG_SIZE = 4
DATASET = Path('terrdreamer/models/progan_satelite/dataset/data/aw3d30')
CHECKPOINT_GENERATOR = "terrdreamer/models/progan_satelite/checkpoints/generator.pth"
CHECKPOINT_DISCRIMINATOR = "terrdreamer/models/progan_satelite/checkpoints/discriminator.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [8, 8, 8, 8, 8, 8, 8]
BATCH_SIZE = 8
IMG_CHANNELS = 3
LATENT_SIZE = 512
IN_CHANNELS = 256
DISCRIMINATOR_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, LATENT_SIZE, 1, 1).to(DEVICE)
NUM_WORKERS = 4