# python 3.7
""" This code shows an example of using GANs to generate synthetic images.
The code is adopted from https://github.com/genforce/sefa """

# import the required libraries
import cv2
import torch
import numpy as np
from utils import to_tensor
from utils import postprocess
from utils import load_generator
from models import parse_gan_type
from utils import factorize_weight
from matplotlib import pyplot as plt


def sample(generator, gan_type, num=1, seed=0):
    """Samples latent codes."""

    torch.manual_seed(seed)
    codes = torch.randn(num, generator.z_space_dim).cuda()

    if gan_type == 'pggan':
        codes = generator.layer0.pixel_norm(codes)
    elif gan_type == 'stylegan':
        codes = generator.mapping(codes)['w']
        codes = generator.truncation(codes, trunc_psi=0.7, trunc_layers=8)
    elif gan_type == 'stylegan2':
        codes = generator.mapping(codes)['w']
        codes = generator.truncation(codes, trunc_psi=0.5, trunc_layers=18)

    codes = codes.detach().cpu().numpy()

    return codes


def synthesize(model, gan_type, code):
    """Synthesizes an image with the given code."""
    if gan_type == 'pggan':
        image = model(to_tensor(code))['image']
    elif gan_type in ['stylegan', 'stylegan2']:
        image = model.synthesis(to_tensor(code))['image']
    else:
        print("GAN types is not valid!")
        image = []
    image = postprocess(image)
    return image


def imshow(images, col, viz_size=256):
    """Shows images in one figure."""
    num, height, width, channels = images.shape
    assert num % col == 0
    row = num // col

    fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)

    for idx, image in enumerate(images):
        i, j = divmod(idx, col)
        y = i * viz_size
        x = j * viz_size
        if height != viz_size or width != viz_size:
          image = cv2.resize(image, (viz_size, viz_size))
        fused_image[y:y + viz_size, x:x + viz_size] = image

    fused_image = np.asarray(fused_image, dtype=np.uint8)

    return fused_image


def main():
    num_samples = 8 # num of image to generate (min:1, max:8)
    noise_seed = 121 # noise seed (min:0, max:1000)

    # params of generation (default is 0.0)
    layer_idx = "0-1"  # ['all', '0-1', '2-5', '6-13']
    semantic_1 = 1.4  # min:-3.0, max:3.0
    semantic_2 = -2.9  # min:-3.0, max:3.0
    semantic_3 = -1.2  # min:-3.0, max:3.0
    semantic_4 = 1.2   # min:-3.0, max:3.0
    semantic_5 = -1.4  # min:-3.0, max:3.0

    # select model name, in this example we use "stylegan_car512"
    # ['stylegan_animeface512', 'stylegan_car512', 'stylegan_cat256', 'pggan_celebahq1024',
    # 'stylegan_bedroom256']
    model_name = 'stylegan_car512'

    # load the pretrained model
    generator = load_generator(model_name)
    gan_type = parse_gan_type(generator)

    # samples latent codes
    codes = sample(generator, gan_type, num_samples, noise_seed)

    # generate the synthetic image from the code
    images = synthesize(generator, gan_type, codes)

    # fast implementation to factorize the weight by SeFa
    layers, boundaries, _ = factorize_weight(generator, layer_idx)

    new_codes = codes.copy()
    for sem_idx in range(5):
        boundary = boundaries[sem_idx:sem_idx + 1]
        step = eval(f'semantic_{sem_idx + 1}')
        if gan_type == 'pggan':
            new_codes += boundary * step
        elif gan_type in ['stylegan', 'stylegan2']:
            new_codes[:, layers, :] += boundary * step

    # generate the synthetic image after changing the semantics (params of generation)
    new_images = synthesize(generator, gan_type, new_codes)

    # show the generated images
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(imshow(images, num_samples))
    axarr[0].axis('off')
    axarr[1].imshow(imshow(new_images, num_samples))
    axarr[1].axis('off')
    plt.show()


if __name__ == '__main__':
    main()
