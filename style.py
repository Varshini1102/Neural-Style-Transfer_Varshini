import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize

from Config import Config
from model import AesFA_test
from blocks import test_model_load

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().numpy()
    image = image.transpose(0, 2, 3, 1)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def do_transform(img, osize):
    transform = Compose([Resize(size=osize),
                         CenterCrop(size=osize),
                         ToTensor(),
                         Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return transform(img).unsqueeze(0)

def save_img(config, cont_name, sty_name, content, style, stylized):
    real_A = im_convert(content)
    real_B = im_convert(style)
    trs_AtoB = im_convert(stylized)

    A_image = Image.fromarray((real_A[0] * 255.0).astype(np.uint8))
    B_image = Image.fromarray((real_B[0] * 255.0).astype(np.uint8))
    trs_image = Image.fromarray((trs_AtoB[0] * 255.0).astype(np.uint8))

    cont_name = cont_name.split('/')[-1].split('.')[0]
    sty_name = sty_name.split('/')[-1].split('.')[0]

    A_image.save('{}/{:s}_content.jpg'.format(config.img_dir, cont_name))
    B_image.save('{}/{:s}_style.jpg'.format(config.img_dir, sty_name))
    trs_image.save('{}/stylized_{:s}_{:s}.jpg'.format(config.img_dir, cont_name, sty_name))

def main():
    config = Config()
    if not os.path.exists(config.img_dir):
        os.makedirs(config.img_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Version:', config.file_n)
    print(device)

    with torch.no_grad():
        ## Model load
        model = AesFA_test(config)

        ## Load saved model
        ckpt = config.ckpt_dir + '/main.pth'
        print("checkpoint: ", ckpt)
        model = test_model_load(checkpoint=ckpt, model=model)
        model.to(device)

        ## Style Transfer
        if hasattr(config, "style_img"):
            style_path = config.style_img
        elif hasattr(config, "style_dir"):
            style_path = os.path.join(config.style_dir, "specific_style_image.jpg")  # Replace with your logic
        else:
            raise AttributeError("Config must have either 'style_img' or 'style_dir' defined.")

        real_A = Image.open(config.content_img).convert('RGB')
        style = Image.open(style_path).convert('RGB')

        real_A = do_transform(real_A, config.blend_load_size).to(device)
        style = do_transform(style, config.blend_load_size).to(device)

        stylized, during = model.style_transfer(real_A, style)
        save_img(config, config.content_img, style_path, real_A, style, stylized)
        print("Time:", during)
        print("Time:", during)

if __name__ == '__main__':
    main()
