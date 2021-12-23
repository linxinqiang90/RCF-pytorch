from models import RCF
import numpy as np
import os
from PIL import Image
import torch
import torchvision

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
input = '/home/linxinqiang/Desktop/test/j4';
save_dir = '/home/linxinqiang/Desktop/test/j4_result'
files = os.listdir(input)
counter = 1

def prepare_image_PIL(im):
    im = im[:, :, ::-1] - np.zeros_like(im)  # rgb to bgr
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im

for i in files:
    model = RCF()
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img = np.array(Image.open(input+"/"+i), dtype=np.float32)
    img = prepare_image_PIL(img)
    _, H, W = img.shape
    # img = img.transpose((2, 0, 1))
    img = torch.unsqueeze(torch.from_numpy(img).cpu(), 0)
    results = model(img)

    result = torch.squeeze(results[-1].detach()).cpu().numpy()
    results_all = torch.zeros((len(results), 1, H, W))
    for i in range(len(results)):
      results_all[i, 0, :, :] = results[i]

    torchvision.utils.save_image(1-results_all, os.path.join(save_dir, f"all_{os.path.split(i)[0]}.jpg"))

    result = Image.fromarray((result * 255).astype(np.uint8))
    result.save(os.path.join(save_dir, f"{os.path.split(i)[0]}.jpg" ))
    print("Running test [%d/%d]" % (counter, len(os.listdir(input))))
    counter += 1