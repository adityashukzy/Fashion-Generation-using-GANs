<p align="center">
<a href="https://dscommunity.in">
	<img src="https://github.com/Data-Science-Community-SRM/template/blob/master/Header.png?raw=true" width=80%/>
</a>
	<h1 align="center">Fashion Generation using Generative Modeling</h1>
	<h4 align="center">Deep Convolutional GAN to generate fashion images<h4>
</p>

---

## Description
Our team of 5 worked on the task of generating novel fashion images from an existing dataset called [DeepFashion]() using the popular generative adversarial network (or GAN) architecture. We experimented with several different architectures, losses, learning rates, training techniques (which are a particular stress point when it comes to building GANs.).

The architecture we have used for the most part is that of the Deep Convolutional GAN or DCGAN. Although we have experimented with things such as adding dropout, changing the dimensionality of some layers, etc., the basic concept of ConvTranspose2d (Upsampling layers) and Conv2d (simple Conv. layers) has been kept.

In addition, we also added an element of giving the generator some meaningful input to start with rather than simple random noise. While the generator is conventionally given some random noise sampled from a normal distribution, we have passed the input image through a ResNet architecture, extracted an encoded vector which will contain some compressed information about the image and then concatenated this encoding with some random noise. This is finally then given to the generator.

The idea behind this method is to give the generator some help in that we are giving it noise which already has some rhyme and reason, which will hopefully make the generator's job easier when it comes to mapping this noise to a proper output which can fool the discriminator.

## Preview (training for ~500 epochs)
### Beginning of training
![Screen Shot 2021-03-05 at 11 00 22](https://user-images.githubusercontent.com/20011207/110071433-05c67c00-7da2-11eb-8d46-13759b7b0161.png)
![Screen Shot 2021-03-05 at 11 01 19](https://user-images.githubusercontent.com/20011207/110071480-1ecf2d00-7da2-11eb-928c-6eeb98583d8a.png)

### One-fourth through training
![Screen Shot 2021-03-05 at 11 03 07](https://user-images.githubusercontent.com/20011207/110071628-5fc74180-7da2-11eb-961d-994845de368c.png)
![Screen Shot 2021-03-05 at 11 04 59](https://user-images.githubusercontent.com/20011207/110071773-a321b000-7da2-11eb-8b40-e9cdd0d289bd.png)

### Halfway through training
![Screen Shot 2021-03-05 at 11 06 18](https://user-images.githubusercontent.com/20011207/110071867-d19f8b00-7da2-11eb-8d0a-e2a9ab443486.png)
![Screen Shot 2021-03-05 at 11 07 43](https://user-images.githubusercontent.com/20011207/110071966-03b0ed00-7da3-11eb-8477-fdf584882dcf.png)

### Three-fouth through training
![Screen Shot 2021-03-05 at 11 09 52](https://user-images.githubusercontent.com/20011207/110072126-5094c380-7da3-11eb-9aa0-8b70bbaa664a.png)
![Screen Shot 2021-03-05 at 11 10 44](https://user-images.githubusercontent.com/20011207/110072212-70c48280-7da3-11eb-858d-db7a30164a18.png)

### End of training
![Screen Shot 2021-03-05 at 11 15 15](https://user-images.githubusercontent.com/20011207/110072612-111aa700-7da4-11eb-936f-108c319495fc.png)
![Screen Shot 2021-03-05 at 11 16 42](https://user-images.githubusercontent.com/20011207/110072747-458e6300-7da4-11eb-84c5-824727757c49.png)

The model did not seem to converge, and the quality of outputs did not increase significantly after this point.
<br>


## Instructions to run

* Pre-requisites:
	-  < insert pre-requisite >
	-  < insert pre-requisite >

* < directions to install > 
```bash
< insert code >
```

* < directions to execute >

```bash
< insert code >
```

## Contributors

<table>
<tr align="center">


<td>

Abhishek Saxena

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/33656173?s=400&u=a411c58cfffec9bf59da192674093abf4b82bd04&v=4"  height="120" alt="Abhishek Saxena">
</p>
<p align="center">
<a href = "https://github.com/saxenabhishek"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/abhibored/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>


<td>

Aditya Shukla

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/20011207?s=400&u=7570f3915eca3bcd55cd72c60038e4f68965db4b&v=4"  height="120" alt="Aditya Shukla">
</p>
<p align="center">
<a href = "https://github.com/adityashukzy"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/person2">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>



<td>

Aradhya Tripathi

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/67282231?s=400&u=d2ee088fed219592ce11e243ca05863c689f2f1e&v=4"  height="120" alt="Aradhya Tripathi">
</p>
<p align="center">
<a href = "https://github.com/Aradhya-Tripathi"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/aradhya-tripathi51/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>

<td>

Harsh Sharma

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/56259814?s=400&u=f08cf73c6051e212a4ffc3a7cef84e872abcf35a&v=4"  height="120" alt="Your Name Here (Insert Your Image Link In Src">
</p>
<p align="center">
<a href = "https://github.com/harshgeek4coder"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/harshsharma27/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>

<td>

Harshit Aggarwal

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/51515489?s=400&u=193dc2f58f7e354e413fadb400e12767fd7ac509&v=4"  height="120" alt="Harshit Aggarwal">
</p>
<p align="center">
<a href = "https://github.com/harshitaggarwal01"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/harshit-a-46b4a0b7/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>

</tr>
  </table>
  
## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

<p align="center">
	Made with ❤️ by <a href="https://dscommunity.in">DS Community SRM</a>
</p>

