import argparse
import os
import torch
from src.model import DiffusionModel
from src.utils import load_checkpoint, save_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default='outputs/generated_images')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diff_model = DiffusionModel(device=device).to(device)
    ckpt = load_checkpoint(args.checkpoint, device)
    diff_model.unet.load_state_dict(ckpt['diffusion_state_dict'])
    diff_model.adaptor.load_state_dict(ckpt['adaptor_state_dict'])
    diff_model.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    diff_model.unet.eval()
    diff_model.adaptor.eval()

    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    all_imgs = []
    for i in range(num_batches):
        batch_size = min(args.batch_size, args.num_samples - i*args.batch_size)
        imgs = diff_model.sample(batch_size)
        all_imgs.append(imgs)

    all_imgs = torch.cat(all_imgs, dim=0)
    save_images(all_imgs, args.output_dir, prefix='gen')

if __name__ == '__main__':
    main()