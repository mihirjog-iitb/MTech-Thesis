This repo is for performing federated learning experiments with the library. Note that in this, both client and server has to run on the PC.

Steps to run:
1. run pvserver.py in a terminal. It takes 3 arguments
   i. train/fed: want to do feretated learning or just training
   ii. even/ odd: which subset of data to be given (this will work for 2 clients) run one client with 'even' argument and the other with 'odd'.
   iii. Plant dataset: currently it will support 'Apple', 'Onion' and 'Tomato'. Apple and tomato are from PV. Onion is fro TIH, and is not publically available. contact respective authorities for permission.
3. Run as many numbers of pvclient.py as you wish in separate terminals. MOdify the number of clients in pvserver.py accordingly.
4. you shall see the round-wise accuracy results in the pvserver terminal once the rounds are finished.

   Main requirements:
   Flwr library
   Pytorch

For more details, refer to the comments in the code that start with 'MJ'
