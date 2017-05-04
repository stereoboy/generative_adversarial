# generative_adversarial
* Various Generative Adversarial Networks using tensorflow
* Main Reference: https://github.com/shekkizh/EBGAN.tensorflow'
  * copied main optimizer code and most setup codes
  * refactored model build-up and variable-maintaing codes
  * modified some calculations to follow the details of the original paper

## EBGAN
* Generate face images based on CelebA data

## DCGAN
* Generate face images based on CelebA data

* Under epoch #1

<img src="./dc_checkpoints/epoch01/generated00.png" width="100">|<img src="./dc_checkpoints/epoch01/generated01.png" width="100">|<img src="./dc_checkpoints/epoch01/generated02.png" width="100">|<img src="./dc_checkpoints/epoch01/generated03.png" width="100">|<img src="./dc_checkpoints/epoch01/generated04.png" width="100">

* Under epoch #2

<img src="./dc_checkpoints/epoch02/generated00.png" width="100">|<img src="./dc_checkpoints/epoch02/generated01.png" width="100">|<img src="./dc_checkpoints/epoch02/generated02.png" width="100">|<img src="./dc_checkpoints/epoch02/generated03.png" width="100">|<img src="./dc_checkpoints/epoch02/generated04.png" width="100">

* Under epoch #3

<img src="./dc_checkpoints/epoch03/generated00.png" width="100">|<img src="./dc_checkpoints/epoch03/generated01.png" width="100">|<img src="./dc_checkpoints/epoch03/generated02.png" width="100">|<img src="./dc_checkpoints/epoch03/generated03.png" width="100">|<img src="./dc_checkpoints/epoch03/generated04.png" width="100">

* Cherry-picked results

<img src="./dc_checkpoints/cherry-picked/generated19.png" width="100">|<img src="./dc_checkpoints/cherry-picked/generated21.png" width="100">|<img src="./dc_checkpoints/cherry-picked/generated41.png" width="100">|<img src="./dc_checkpoints/cherry-picked/generated65.png" width="100">|<img src="./dc_checkpoints/cherry-picked/generated93.png" width="100">

## InfoGAN
* Generate  hand-written number images based on mnist data
* Need to fine-tune parameters
  * It is delicate to make InfoGan converge. It is easy to make simple GAN converge. But fine-tuning is needed when applying latent codes.
* Use simple adaptive generator optimization.
  * accuracy 0.4
* Use categorical latent code only (continuous latent code)
### Result

* After 14 epochs

<img src="./info_mnist_checkpoints/vis_00_00.png">
<img src="./info_mnist_checkpoints/vis_00_10.png">
<img src="./info_mnist_checkpoints/vis_00_20.png">
<img src="./info_mnist_checkpoints/vis_00_30.png">
<img src="./info_mnist_checkpoints/vis_00_40.png">

* After 15 epochs

<img src="./info_mnist_checkpoints/vis_01_00.png">
<img src="./info_mnist_checkpoints/vis_01_10.png">
<img src="./info_mnist_checkpoints/vis_01_20.png">
<img src="./info_mnist_checkpoints/vis_01_30.png">
<img src="./info_mnist_checkpoints/vis_01_40.png">

* comments
  * '8' to class #0, '1' to class #1, '3' to class #2, and so on.
  * '7', '9', '4' are not distinguishable in this result, this means that encoding of '7', '9', '4' into distinct codes failed.
    * But network can encode these 3 numbers into separate codes in another trial.
    * Whenever I try it, different result comes. Sometimes I succed all, sometime fail 2 numbers.

## Adversarial Autoencoders
