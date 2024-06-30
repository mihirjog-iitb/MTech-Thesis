This repo is for performing federated learning experiments with the library. Note that in this, both client and server has to run on the PC.

Steps to run:
1. run pvserver.py in a terminal
2. Run as many numbers of pvclient.py as you wish in separate terminals. MOdify the number of clients in pvserver.py accordingly.
3. you shall see the round-wise accuracy results in the pvserver terminal once the rounds are finished.

   Main requirements:
   Flwr library
   Pytorch

For more details, refer to the comments in the code that start with 'MJ'
