import os
import random
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from gan_module import Generator

parser = ArgumentParser()
parser.add_argument(
    '--image_dir', default='/Downloads/CACD_VS/', help='The image directory')
parser.add_argument(
    '--save_dir', default='/Downloads/CACD_VS/', help='The image saving directory')


@torch.no_grad()
def main():
    args = parser.parse_args()
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if
                   x.endswith('.png') or x.endswith('.jpg')]
    images_paths_ranking = [(path, int(path.split("-")[2].split('.')[0]))
                            for path in image_paths]

    # bubblesort
    for i in range(len(images_paths_ranking) - 1):  # 遍历 len(nums)-1 次
        for j in range(len(images_paths_ranking) - i - 1):  # 已排好序的部分不用再次遍历
            if images_paths_ranking[j][1] > images_paths_ranking[j+1][1]:
                # Python 交换两个数不用中间变量
                images_paths_ranking[j], images_paths_ranking[j +
                                                              1] = images_paths_ranking[j+1], images_paths_ranking[j]
    images_paths = [image[0] for image in images_paths_ranking]
    # print(images_paths)

    model = Generator(ngf=32, n_residual_blocks=9)
    ckpt = torch.load('pretrained_model/state_dict.pth', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # fig, ax = plt.subplots(2, nr_images, figsize=(20, 10))
    # random.shuffle(image_paths)
    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')
        img = trans(img).unsqueeze(0)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        aged_face = aged_face * 255
        aged_face = np.array(aged_face, dtype=np.uint8)
        if np.ndim(aged_face) > 3:
            assert aged_face.shape[0] == 1
            aged_face = aged_face[0]
        # ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        # ax[1, i].imshow(aged_face)
        aged_face = Image.fromarray(aged_face, 'RGB')
        aged_face.save(args.save_dir + '/' + image_path.split("-")
                       [2].split('.')[0] + '.png')
    # plt.show()
    # plt.savefig("mygraph.png")


if __name__ == '__main__':
    main()
