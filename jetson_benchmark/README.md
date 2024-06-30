requirements.
Docker container: 'l4tml-32.7.1'

The inf2.py file has to be run inside the docker. This file performs the inference using tensorrt (.trt file of corrosponding module)
This code performs the inference for imagenet pretrained model.
We need to have 'tiny-imagenet' dataset on our device, from which we will get the test images. This dataset is easily available on internet.

monitor.py file has to be RUN OUTSIDE THE DOCKER.

1. Run monitor.py in one shell
2. Run inf2.py in other shell (docker shell)

After monitor.py file will record the resource consumption while inference is running. Refer to my comments in these files.
