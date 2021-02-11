import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ML.models import ResNetEncoder, Discriminator, Generator
from ML.utils.dataset import Data

if __name__ == "__main__":
    vec_shape = 1000

    print(f"Evalutating with VecSize {vec_shape}")

    netG = Generator(device="cpu", noise_dim=500, vec_shape=vec_shape)
    netD = Discriminator()
    netENC = ResNetEncoder(vec_shape)

    netG.load_state_dict(torch.load("Gen.pt", map_location=torch.device("cpu")))
    netENC.load_state_dict(torch.load("RES.pt", map_location=torch.device("cpu")))

    netG.eval()
    netENC.eval()

    numrows = 10
    d = Data(path="../fashiondata/img", batch_size=numrows, size=(64, 64))

    d_loaded = DataLoader(d.folderdata, numrows, shuffle=True)

    # get one random batch of images
    imgs = next(iter(d_loaded))[0]

    with torch.no_grad():
        vector = netENC(imgs)
        fakeImages = netG(vector)

    _, ax = plt.subplots(
        2, numrows, squeeze=False, sharex=True, sharey=True, figsize=(8, 4)
    )

    for i in range(numrows):
        ax[0, i].imshow(
            (fakeImages[i].permute(1, 2, 0).numpy() + [1, 1, 1]) / [2, 2, 2]
        )
        ax[0, i].axis(False)

        ax[1, i].imshow(
            (imgs[i].permute(1, 2, 0).numpy() + [1, 1, 1]) / [2, 2, 2]
        )
        ax[1, i].axis(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
