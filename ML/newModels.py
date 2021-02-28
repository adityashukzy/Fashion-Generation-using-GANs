from torch import nn
from torchvision import models
import torch

class Generator(nn.Module):
	"""
	Generator architecture of the GAN
	"""
	def __init__(self, **kwargs):
		super(Generator, self).__init__()

		self.device = kwargs["device"]
		self.noise_dim = kwargs["noise_dim"]
		
		self.gen = nn.Sequential(
			self.genBlock(
				in_channels=self.noise_dim,
				out_channels=512,
				kernel_size=4,
				stride=1,
				padding=0
				),
			self.genBlock(
				in_channels=512,
				out_channels=350,
				kernel_size=4,
				stride=2,
				padding=1
				),
			self.genBlock(
				in_channels=350,
				out_channels=250,
				kernel_size=4,
				stride=2,
				padding=1
				),
			self.genBlock(
				in_channels=250,
				out_channels=150,
				kernel_size=4,
				stride=2,
				padding=1
				),
			self.genBlock(
				in_channels=150,
				out_channels=3,
				kernel_size=4,
				stride=2,
				padding=1
				),
		)
	
	def genBlock(
		self,
		in_channels,
		out_channels,
		kernel_size,
		stride,
		padding,
		last_layer=False):

		if not last_layer:
			return nn.Sequential(
				nn.ConvTranspose2d(
					in_channels,
					out_channels,
					kernel_size,
					stride,
					padding,
					bias = False),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True)
			)
		else:
			return nn.Sequential(
				nn.ConvTranspose2d(
					in_channels,
					out_channels,
					kernel_size,
					stride,
					padding,
					bias = False),
				nn.Tanh()
			)

	def generateNoise(self, batch_size):
		return torch.randn(batch_size, self.noise_dim, device=self.device)

	def forward(self, batch_size):
		'''
			batch_size dictates how many random samples are to be generated
			and packed into a minibatch
		'''

		random_noise = self.generateNoise(batch_size)
		gen_output = self.gen(random_noise)
		
		return gen_output


class Discriminator(nn.Module):
	"""
	Discriminator architecture of the GAN
	"""
	def __init__(self):
		super(Discriminator, self).__init__()
	
		self.disc = nn.Sequential(
			self.discBlock(
				in_channels=3,
				out_channels=128,
				first_layer=True
			),
			self.discBlock(
				in_channels=128,
				out_channels=256
			),
			self.discBlock(
				in_channels=256,
				out_channels=512
			)
		)

	def discBlock(self, in_channels, out_channels, first_layer=False):
		if first_layer:
			return nn.Sequential(
				nn.Conv2d(
					in_channels=in_channels,
					out_channels=out_channels,
					kernel_size=4,
					stride=2,
					padding=1,
					bias=False
				),
				nn.LeakyReLU(0.2, inplace=True)
			)
		else:
			return nn.Sequential(
				nn.Conv2d(
					in_channels=in_channels,
					out_channels=out_channels,
					kernel_size=4,
					stride=2,
					padding=1,
					bias=False
				),
				nn.BatchNorm2d(out_channels),
				nn.LeakyReLU(0.2, inplace=True)
			)
	
	def forward(self, input_image):
		disc_output = self.disc(input_image)
		return disc_output

def main():
	if torch.cuda.is_available():
		device = "cuda"
	else:
		device = "cpu"
	
	generator = Generator(device=device, noise_dim=100)
	generator = generator.to(device)

	discriminator = Discriminator()

	print(generator(batch_size=25).shape)
	print(discriminator(generator(batch_size=25)).shape)

if __name__ == '__main__':
	main()
