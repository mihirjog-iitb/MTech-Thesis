This is to do the federated learning with jetson nano. You need to run both server and client code.
Client code will run on the jetson nano.
Prerequisits for jetson nano: 
-Jetpack 4.6.1 (comes with sdcard image)
- you will need to install docker container 'l4tml-32.7.1' from nvidia ngc container. The python files will run inside the container, UNLESS STATED OTHERWISE.
- The py_process.py will run inside the container with python3.
- You need to run 'scptrial.py' OUTSIDE the docker. The talk.txt file will server as messanger between py_process.py and 'scp_trial.py'. scptrial.py will do the task of file transfer. Make sure that you change the file names of the destination, when you copy these files to second client. For example, 'net1.pt' in one client and 'net2.pt' for other clients. Refer my comment in this file for understanding better.
- You also need to run POWMODE.py file OUTSIDE THE DOCKER. This is to control the power mode. it will also use a text file to communicate with the app.
- make sure to run py_process.py file only after running the server socket opening cell in server code notebook

make sure that for the backbone model, you have corrosponding .onnx model file copied to the jetson.
Once you copy the .onnx file, you need to convert it to .trt (tensorrt) using terexec.

For server, the elaborate instructions are placed in the markup cells.
