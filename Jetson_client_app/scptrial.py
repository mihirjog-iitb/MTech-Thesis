import subprocess
import time
fp="talk.txt"
while(True):
	with open(fp,'r') as fd:
		char= fd.read(1)
	if char=='r':       #send the file to server
		print("r is found performing scp")
		command = "scp net1.pt vip-lab@10.107.47.157:/home/vip-lab/Downloads/workspace/net1.pt"

		output = subprocess.run(command, shell=True, check=True)

		sleep(1)

		command = "scp feat_dataset.pkl vip-lab@10.107.47.157:/home/vip-lab/Downloads/workspace/feat_dataset_1.pkl"
		
		output = subprocess.run(command, shell=True, check=True)

		with open(fp,'w') as fd:
			ack='c'
			fd.write(ack)

	time.sleep(1)
