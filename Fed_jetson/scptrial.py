import subprocess
import time
fp="talk.txt"
while(True):
	with open(fp,'r') as fd:
		char= fd.read(1)
	if char=='r':       #send the file to server
		print("r is found performing scp")
		#MJ: make sure to use proper username, and IP address in the command (as per server machine)
		command = "scp net1.pt vip-lab@10.107.47.157:/home/vip-lab/Downloads/workspace/net1.pt" # MJ: make sure when you use this file in second client (device), you change the destination file name. eg. net2.pt 

		output = subprocess.run(command, shell=True, check=True)

		sleep(1)

		command = "scp feat_dataset.pkl vip-lab@10.107.47.157:/home/vip-lab/Downloads/workspace/feat_dataset_1.pkl" # MJ:make sure when you use this file in second client (device), you change the destination file name. eg. feat_dataset_2.pkl 

		
		output = subprocess.run(command, shell=True, check=True)

		with open(fp,'w') as fd:
			ack='c'
			fd.write(ack)

	time.sleep(1)
