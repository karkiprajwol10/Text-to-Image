import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from data_loaders.CUB_200_2011 import CUB200Dataset
from models import get_gmodel
from ganlib.priors import get_sampler_fn
from pathlib import Path
from easydict import EasyDict as edict
from PIL import Image
import tensorflow as tf
from generate_embeddings import generate_embed


model_folder = Path("efficient_models")
epoch = 440
device = torch.device('cpu') # 'cuda' for GPU
m = torch.load(
    model_folder / f"netG_avg_epoch_{epoch}.pth",
    map_location=lambda storage, loc: storage,
)
netG = get_gmodel(**m['netG_params'])
netG.load_state_dict(m['net'])
netG = netG.eval().to(device)
sampler_fn = get_sampler_fn(
    'normal',
    device=device,
    normalize='false',
)

# Load Dataset
def load_dataset():
    tform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    dset = CUB200Dataset(
        'datasets/CUB',
        tform,
        None,
        split='test',
        return_captions=True,
        return_fnames=True,
    )

    return dset

def show_real_img(dset):
    cap_idx = 3 # from 0 to 9
    img, embs, caption, fname, fix = dset[3] # test instance idx
    embs = embs.squeeze(0)[cap_idx]
    emb = torch.tensor(embs[None, ...], dtype=torch.float32)
    print(emb.shape)
    cap = caption[cap_idx]
    print('fname:', fname, 'fix:', fix, 'cap idx:', cap_idx, 'caption:', cap)
    img = img.mul(0.5).add(0.5).clamp(0, 1).mul(255).clamp_(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return emb,cap


def generate_gan_image(emb, samples, random_number):
    z = sampler_fn(samples, 128)
    with torch.no_grad():
        imgs = netG(z.to(device), emb.to(device).repeat(samples, 1))
        imgs = imgs.mul(0.5).add(0.5).clamp(0, 1).mul(255).clamp_(0, 255).byte().permute(0,2,3,1).cpu().numpy()
    for ii in range(samples):
        save_img(imgs[ii], random_number[ii])
    

def save_img(image, random_number):
    tf.keras.utils.save_img(f"static/images/gan_{random_number}.jpg", image)

def plot_img(img, cap):
    fig = plt.figure(figsize=(9, 6))
    plt.imshow(img)
    plt.text(0, -10, cap, fontsize=15)
    plt.show()

def run(text, random_number, samples):
    # dset = load_dataset()
    emb = generate_embed(text)
    generate_gan_image(emb, samples, random_number)
    print("Image Saved")

