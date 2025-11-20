# Ultra Zoom

A fast single image super-resolution (SISR) model for upscaling images with ultra high-quality. Ultra Zoom uses a two-stage "zoom in and enhance" mechanism that utilizes a fast deterministic upscaling algorithm to upscale the image and then enhances it through a steerable residual pathway that operates primarily in the low-resolution subspace of a deep neural network.

## Key Features

- **Fast and scalable**: Ultra Zoom uses a unique "zoom in and enhance" mechanism that combines the speed of deterministic bicubic interpolation with the power of a deep neural network.

- **Controllable enhancements**: Ultra Zoom integrates channel-wise control modules directly into the architecture, allowing you to finely adjust the amount of denoising, deblurring, and deartifacting to suite your image source.

- **Full RGB**: Unlike many efficient SR models that only operate in the luminance domain, Ultra Zoom operates within the full RGB color domain enhancing both luminance and chrominance for the best possible image quality.

## Demos

View at full resolution for best results. More comparisons can be found [here](https://github.com/andrewdalpino/UltraZoom/tree/master/docs/images).

![UltraZoom 2X Comparison](https://raw.githubusercontent.com/andrewdalpino/UltraZoom/master/docs/images/cat-2x-comparison.png)
![UltraZoom 3X Comparison](https://raw.githubusercontent.com/andrewdalpino/UltraZoom/master/docs/images/building-3x-comparison.png)
![UltraZoom 4X Comparison](https://raw.githubusercontent.com/andrewdalpino/UltraZoom/master/docs/images/flower-4x-comparison.png)

This comparison demonstrates the strength of the enhancements (deblurring, denoising, and deartifacting) applied to the upscaled image.

![UltraZoom Ctrl Enhancement Comparison](https://raw.githubusercontent.com/andrewdalpino/UltraZoom/master/docs/images/ctrl-compare-all-3.png)

This comparison demonstrates the individual enhancements applied in isolation.

![UltraZoom Ctrl Enhancement Comparison](https://raw.githubusercontent.com/andrewdalpino/UltraZoom/master/docs/images/ctrl-compare-individual.png)

## Pretrained Models

The following pretrained models are available on HuggingFace Hub.

| Name | Upscale | Num Channels | Encoder Layers | Parameters | Control Modules | Library Version |
|---|---|---|---|---|---|---|
| [andrewdalpino/UltraZoom-2X-Ctrl](https://huggingface.co/andrewdalpino/UltraZoom-2X-Ctrl) | 2X | 48 | 20 | 1.8M | Yes | 0.2.x |
| [andrewdalpino/UltraZoom-2X](https://huggingface.co/andrewdalpino/UltraZoom-2X) | 2X | 48 | 20 | 1.8M | No | 0.1.x |
| [andrewdalpino/UltraZoom-3X](https://huggingface.co/andrewdalpino/UltraZoom-3X) | 3X | 54 | 30 | 3.5M | No | 0.1.x |
| [andrewdalpino/UltraZoom-4X](https://huggingface.co/andrewdalpino/UltraZoom-4X) | 4X | 96 | 40 | 14M | No | 0.1.x |

## Pretrained Examples

If you'd just like to load the pretrained weights and do inference, getting started is as simple as in the examples below.

### Non-control Version

First, you'll need the `ultrazoom` package installed into your project. For the non-control version we'll need library version `0.1.x` to load the pretrained weights. We'll also need the `torchvision` library to do some basic image preprocessing. We recommend using a virtual environment to make package management easier.

```sh
pip install ultrazoom~=0.1.0 torchvision
```

Then, load the weights from HuggingFace Hub, convert the input image to a tensor, and upscale the image.

```python
import torch

from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.v2 import ToDtype, ToPILImage

from ultrazoom.model import UltraZoom


model_name = "andrewdalpino/UltraZoom-2X"
image_path = "./dataset/bird.png"

model = UltraZoom.from_pretrained(model_name)

image_to_tensor = ToDtype(torch.float32, scale=True)
tensor_to_pil = ToPILImage()

image = decode_image(image_path, mode=ImageReadMode.RGB)

x = image_to_tensor(image).unsqueeze(0)

y_pred = model.upscale(x)

pil_image = tensor_to_pil(y_pred.squeeze(0))

pil_image.show()
```

### Control Version

The control version of Ultra Zoom allows you to independently adjust the level of deblurring, denoising, and deartifacting applied to the upscaled image. We accomplish this by conditioning the input image on a Control Vector that gets picked up by control modules embedded into each layer of the encoder. Version `0.2.x` of the library is required for control functionality.

```sh
pip install ultrazoom~=0.2.0 torchvision
```

The `ControlVector` class takes 3 arguments - `gaussian_blur`, `gaussian_noise`, and `jpeg_compression` corresponding to the assumed level of each type of degradation present in the input image. Their values range from 0.0 meaning no degradation is assumed present to 1.0 meaning that the maximum amount of degradation is assumed present.

```python
import torch

from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.v2 import ToDtype, ToPILImage

from ultrazoom.model import UltraZoom
from ultrazoom.control import ControlVector


model_name = "andrewdalpino/UltraZoom-2X-Ctrl"
image_path = "./dataset/bird.png"

model = UltraZoom.from_pretrained(model_name)

image_to_tensor = ToDtype(torch.float32, scale=True)
tensor_to_pil = ToPILImage()

image = decode_image(image_path, mode=ImageReadMode.RGB)

x = image_to_tensor(image).unsqueeze(0)

c = ControlVector(
    gaussian_blur=0.5,      # Higher values indicate more degradation
    gaussian_noise=0.6,     # which increases the strength of the
    jpeg_compression=0.7    # enhancement [0, 1].
).to_tensor()

y_pred = model.upscale(x, c)

pil_image = tensor_to_pil(y_pred.squeeze(0))

pil_image.show()
```

## Training

### Clone the Repository

You'll need the code in the repository to train new models and export them for production.

```sh
git clone https://github.com/andrewdalpino/UltraZoom
```

### Install Project Dependencies

Project dependencies are specified in the `requirements.txt` file. You can install them with [pip](https://pip.pypa.io/en/stable/) using the following command from the project root. We recommend using a virtual environment such as `venv` to keep package dependencies on your system tidy.

```
python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt
```

### Pretraining

Ultra Zoom is trained in two stages. The first stage focuses on building a foundation model for fine-tuning. It aims to jointly minimize the Pixel Loss with high and low frequency perceptual losses from the perspective of a pretrained VGG19 image classifier. To start training with the default settings, add your training and testing images to the `./dataset/train` and `./dataset/test` folders respectively and call the pretraining script like in the example below. If you are looking for good training sets to start with we recommend the `DIV2K` and/or `Flicker2K` datasets.

```sh
python pretrain.py
```

You can customize the upscaler model by adjusting the `num_channels`, `hidden_ratio`, and `num_encoder_layers` hyper-parameters like in the example below.

```sh
python pretrain.py --num_channels=64 --hidden_ratio=2 --num_encoder_layers=24
```

You can also adjust the `batch_size`, `learning_rate`, and `gradient_accumulation_steps` to suite your training setup.

```sh
python pretrain.py --batch_size=16 --learning_rate=5e-4 --gradient_accumulation_steps=8
```

In addition, you can control various training data augmentation arguments such as the brightness, contrast, hue, and saturation jitter.

```sh
python pretrain.py --brightness_jitter=0.5 --contrast_jitter=0.4 --hue_jitter=0.3 --saturation_jitter=0.2
```

#### Training Dashboard

We use [TensorBoard](https://www.tensorflow.org/tensorboard) to capture and display training events such as loss and gradient norm updates. To launch the dashboard server run the following command from the terminal.

```sh
tensorboard --logdir=./runs
```

Then navigate to the dashboard using your favorite web browser.

#### Pretraining Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --train_images_path | "./dataset/train" | str | The path to the folder containing your training images. |
| --test_images_path | "./dataset/test" | str | The path to the folder containing your testing images. |
| --num_dataset_processes | 8 | int | The number of CPU processes to use to preprocess the dataset. |
| --target_resolution | 256 | int | The number of pixels in the height and width dimensions of the training images. |
| --upscale_ratio | 2 | (1, 2, 3, 4) | The upscaling or zoom factor. |
| --min_gaussian_blur | 0.0 | float | The minimum amount of Gaussian blur to apply to the degraded low-resolution image. |
| --max_gaussian_blur | 1.0 | float | The maximum amount of Gaussian blur to apply to the degraded low-resolution image. |
| --min_gaussian_noise | 0.0 | float | The minimum amount of Gaussian noise to add to the degraded low-resolution image. |
| --max_gaussian_noise | 0.1 | float | The maximum amount of Gaussian noise to add to the degraded low-resolution image. |
| --min_compression | 0.0 | float | The minimum amount of JPEG compression to apply to the degraded low-resolution image. |
| --max_compression | 0.8 | float | The maximum amount of JPEG compression to apply to the degraded low-resolution image. |
| --brightness_jitter | 0.1 | float | The amount of jitter applied to the brightness of the training images. |
| --contrast_jitter | 0.1 | float | The amount of jitter applied to the contrast of the training images. |
| --saturation_jitter | 0.1 | float | The amount of jitter applied to the saturation of the training images. |
| --hue_jitter | 0.1 | float | The amount of jitter applied to the hue of the training images. |
| --batch_size | 32 | int | The number of training images to pass through the network at a time. |
| --gradient_accumulation_steps | 4 | int | The number of batches to pass through the network before updating the model weights. |
| --num_epochs | 100 | int | The number of epochs to train for. |
| --learning_rate | 5e-4 | float | The learning rate of the AdamW optimizer. |
| --max_gradient_norm | 2.0 | float | Clip gradients above this threshold norm before stepping. |
| --num_channels | 48 | int | The number of channels within each encoder block. |
| --hidden_ratio | 2 | (1, 2, 4) | The ratio of hidden channels to `num_channels` within the activation portion of each encoder block. |
| --num_encoder_layers | 20 | int | The number of layers within the body of the encoder. |
| --activation_checkpointing | False | bool | Should we use activation checkpointing? This will drastically reduce memory utilization during training at the cost of recomputing the forward pass. |
| --eval_interval | 2 | int | Evaluate the model after this many epochs on the testing set. |
| --checkpoint_interval | 2 | int | Save the model checkpoint to disk every this many epochs. |
| --checkpoint_path | "./checkpoints/checkpoint.pt" | str | The path to the base checkpoint file on disk. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --run_dir_path | "./runs" | str | The path to the TensorBoard run directory for this training session. |
| --device | "cpu" | str | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator. |

### Fine-tuning

This next stage focuses on squeezing extra performance out of the model using an adversarial training framework. Step 2 of training takes the pretrained checkpoint and fine-tunes the model using feedback from an adversarial critic model. The critic is specially optimized to detect slight differences between real images and images generated by Ultra Zoom. It uses feedback from the upscaler to improve its detection rate and in turn the upscaler uses feedback from the critic to improve its fool rate. This stage can be considered fully optimized when the critic can no longer reliably detect fake images i.e. the F1 score is pegged near 0.5. To start fine-tuning your pretrained checkpoint see the example below.

```sh
python fine-tune.py --base_checkpoint_path=./checkpoints/2X-100.pt
```

To adjust the size of the critic model use the `critic_model_size` argument.

```sh
python fine-tune.py --base_checkpoint_path="./checkpoints/2X-100.pt" --critic_model_size=medium
```

#### Fine-tuning Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --base_checkpoint_path | None | str | The path to the pretrained checkpoint. |
| --train_images_path | "./dataset/train" | str | The path to the folder containing your training images. |
| --test_images_path | "./dataset/test" | str | The path to the folder containing your testing images. |
| --num_dataset_processes | 8 | int | The number of CPU processes to use to preprocess the dataset. |
| --target_resolution | 512 | int | The number of pixels in the height and width dimensions of the training images. |
| --min_gaussian_blur | 0.0 | float | The minimum amount of Gaussian blur to apply to the degraded low-resolution image. |
| --max_gaussian_blur | 1.0 | float | The maximum amount of Gaussian blur to apply to the degraded low-resolution image. |
| --min_gaussian_noise | 0.0 | float | The minimum amount of Gaussian noise to add to the degraded low-resolution image. |
| --max_gaussian_noise | 0.1 | float | The maximum amount of Gaussian noise to add to the degraded low-resolution image. |
| --min_compression | 0.0 | float | The minimum amount of JPEG compression to apply to the degraded low-resolution image. |
| --max_compression | 0.8 | float | The maximum amount of JPEG compression to apply to the degraded low-resolution image. |
| --brightness_jitter | 0.1 | float | The amount of jitter applied to the brightness of the training images. |
| --contrast_jitter | 0.1 | float | The amount of jitter applied to the contrast of the training images. |
| --saturation_jitter | 0.1 | float | The amount of jitter applied to the saturation of the training images. |
| --hue_jitter | 0.1 | float | The amount of jitter applied to the hue of the training images. |
| --batch_size | 8 | int | The number of training images to pass through the network at a time. |
| --gradient_accumulation_steps | 16 | int | The number of batches to pass through the network before updating the model weights. |
| --upscaler_learning_rate | 1e-4 | float | The learning rate of the AdamW optimizer. |
| --upscaler_max_gradient_norm | 1.0 | float | Clip gradients above this threshold norm before stepping. |
| --critic_learning_rate | 5e-4 | float | The learning rate of the AdamW optimizer. |
| --critic_max_gradient_norm | 5.0 | float | Clip gradients above this threshold norm before stepping. |
| --num_epochs | 100 | int | The number of epochs to train for. |
| --critic_warmup_epochs | 1 | int | Train the critic model for this many epochs before using it to train the upscaler. |
| --critic_model_size | "small" | str | The size of the critic model. Choice of small, medium, and large. |
| --activation_checkpointing | False | bool | Should we use activation checkpointing? This will drastically reduce memory utilization during training at the cost of recomputing the forward pass. |
| --eval_interval | 2 | int | Evaluate the model after this many epochs on the testing set. |
| --checkpoint_interval | 2 | int | Save the model checkpoint to disk every this many epochs. |
| --checkpoint_path | "./checkpoints/checkpoint.pt" | str | The path to the base checkpoint file on disk. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --run_dir_path | "./runs" | str | The path to the TensorBoard run directory for this training session. |
| --device | "cpu" | str | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator. |

### Test Comparison

You can use the provided `test-compare.py` script to generate upscaled images from the trained model at the default checkpoint like in the example below.

```sh
python test-compare.py --image_path="./example.jpg"
```

To generate images using a different checkpoint you can use the `checkpoint_path` argument like in the example below.

```sh
python test-compare.py --checkpoint_path="./checkpoints/fine-tuned.pt" --image_path="./example.jpg"
```

You can adjust the level of enhancements applied to the image by setting the `gaussian_blur`, `gaussian_noise`, and `jpeg_compression` arguments like in the example below. Each value has been normalized such that 0 means no enhancement and 1 means full enhancement.

```sh
python test-compare.py --guassian_blur=0.5 --gaussian_noise=0.7 --jpeg_compression=0.9 --image_path="./example.jpg" 
```

#### Test compare Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --image_path | None | str | The path to the image file to be upscaled by the model. |
| --checkpoint_path | "./checkpoints/fine-tuned.pt" | str | The path to the base checkpoint file on disk. |
| --gaussian_blur | 0.5 | float | The strength of gaussian blur removal from the image, between 0 and 1. |
| --gaussian_noise | 0.5 | float | The strength of gaussian noise removal from the image, between 0 and 1. |
| --jpeg_compression | 0.5 | float | The strength of JPEG compression artifact removal from the image, between 0 and 1. |
| --device | "cpu" | str | The device to run the computation on. |

## References

>- J. Song, et. al Gram-GAN: Image Super-Resolution Based on Gram Matrix and Discriminator Perceptual Loss, Sensors, 2023.
>- Z. Liu, et al. A ConvNet for the 2020s, 2022.
>- A. Jolicoeur-Martineau. The Relativistic Discriminator: A Key Element Missing From Standard GAN, 2018.
>- J. Yu, et al. Wide Activation for Efficient and Accurate Image Super-Resolution, 2018.
>- J. Johnson, et al. Perceptual Losses for Real-time Style Transfer and Super-Resolution, 2016.
>- W. Shi, et al. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, 2016.
>- T. Salimans, et al. Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks, OpenAI, 2016.
>- T. Miyato, et al. Spectral Normalization for Generative Adversarial Networks, ICLR, 2018.
>- E. Perez, et. al. FiLM: Visual Reasoning with a General Conditioning Layer, Association for the Advancement of Artificial Intelligence, 2018.
