"""
Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .dataset import Data
from .newModels import Generator, Discriminator
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from time import time

class Train:
	def __init__(self, path, epochs, batch_size, split, display_step=10, noise_dim=100, save_dir="ModelWeights"):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.root = save_dir + "/"

		## Instantiating Generator and Discriminator classes
		self.gen = Generator(device=self.device, noise_dim=noise_dim).to(self.device)
		self.disc = Discriminator().to(self.device)

		## Number of epochs and steps on which to display results
		self.epochs = epochs
		self.display_step = display_step

		## Loss and Optimizer
		self.criterion = nn.BCEWithLogitsLoss()

		beta1 = 0.5
		learning_rate = 0.002
		self.disc_optim = optim.Adam(self.disc.parameters(), lr = learning_rate, betas = (beta1, 0.999))
		self.gen_optim = optim.Adam(self.gen.parameters(), lr = learning_rate, betas=(beta1, 0.999))

		## Loading Data
		data = Data(path=path, batch_size=batch_size, size=(64, 64))
		self.train_loader, self.test_loader, _ = data.getdata(split=split)

		## Initializing generator and discriminator weights
		self.gen = self.gen.apply(self.weights_init)
		self.disc = self.disc.apply(self.weights_init)

		## Storing losses for both models
		self.discLosses = []
		self.genLosses = []

	def trainer(self):
		
		## ????
		self.gen.train()
		self.disc.train()

		mean_disc_loss = mean_gen_loss = 0
		cur_step = 0

		testimage = next(iter(self.testloader))
		testimage = testimage[0].to(self.device)

		for epoch in range(self.epochs):
			print(f"Training on {epoch}'th epoch. ")

			for image, _ in tqdm(self.train_loader):
				## Training the discriminator
				self.disc_optim.zero_grad()

				disc_real_out = self.disc(image)
				disc_fake_out = self.disc(self.gen(batch_size=image.shape[0]).detach())
				# presumably, in our orig model, here we passed the image encoding
				# from which we then got .shape[0] as our batch_size, which we passed to noise generator
				# so here since we don't have any image encoding, I'm just straightaway passing the .shape[0] of the image batch
				# which is presumably number of images in the batch i.e. in the variable 'image'

				real_disc_loss = self.criterion(disc_real_out, torch.ones_like(disc_real_out))
				fake_disc_loss = self.criterion(disc_fake_out, torch.zeros_like(disc_fake_out))

				total_disc_loss = (real_disc_loss + fake_disc_loss) / 2
				total_disc_loss.backward()

				mean_disc_loss += total_disc_loss.item() / self.display_step
				self.disc_optim.step()


				## Training the generator
				self.gen_optim.zero_grad()

				gen_real_out = self.disc(self.gen(batch_size=image.shape[0]))
				gen_real_loss = self.criterion(gen_real_out, torch.ones_like(gen_real_out))

				gen_real_loss.backward()
				mean_gen_loss += gen_real_loss.item() / self.display_step
				self.gen_optim.step()

				if cur_step % self.display_step == 0:
					print(f"Step: {cur_step} Generator Loss: {mean_gen_loss}, \t Discriminator Loss: {mean_disc_loss}")

					fake = self.gen(batch_size=testimage.shape[0])
					self.show_tensor_images(fake)
					self.show_tensor_images(testimage)

					self.discLosses.append(mean_disc_loss)
					self.genLosses.append(mean_gen_loss)

					mean_gen_loss = 0
					mean_disc_loss = 0
				
				cur_step += 1

			print("Saving weights!")

			torch.save(self.gen.state_dict(), self.root + "gen.pt")
			torch.save(self.disc.state_dict(), self.root + "disc.pt")

	def show_tensor_images(self, image_tensor, num_images=64, size=(3, 64, 64)):

		image_tensor = (image_tensor + 1) / 2
		image_unflat = image_tensor.detach().cpu()
		image_grid = make_grid(image_unflat[:num_images], nrow=5)
		plt.imshow(image_grid.permute(1, 2, 0).squeeze())
		plt.axis(False)
		plt.show()

	def weights_init(self, m):
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
			torch.nn.init.normal_(m.weight, 0.0, 0.02)
		if isinstance(m, nn.BatchNorm2d):
			torch.nn.init.normal_(m.weight, 0.0, 0.02)
			torch.nn.init.constant_(m.bias, 0)

	def plot_trainer(self):
		assert len(self.discLosses) != 0 and len(self.genLosses) != 0
		plt.plot(self.discLosses, label="Discriminator Loss")
		plt.plot(self.genLosses, label="Generator Loss")
		plt.legend()
		plt.show()

def main():
	train = train("../../fashiondata/img", epochs=1, batch_size=100)
	train.trainer()

if __name__ == "__main__":
	main()