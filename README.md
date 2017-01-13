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

## Adversarial Autoencoders
