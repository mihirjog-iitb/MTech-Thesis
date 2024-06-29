import subprocess
from time import sleep
fp="powtalk.txt"
mode=0
while(True):
	with open(fp,'r') as fd:
		char= fd.read(1)
	if char=='H' or char=='L':       #send the file to server
		print("switching the power mode")
		
		if(char=='H'):
			mode='0'
			modechar="HIGH"
		else:
			mode='1'
			modechar="LOW"
		try:	
			subprocess.run(['sudo', 'nvpmodel', '-m', mode], check=True)
			print(f"Power mode set to {modechar}")
		except subprocess.CalledProcessError as e:
			print(f"Error setting power mode: {e}")

		with open(fp,'w') as fd:
			ack='c'
			fd.write(ack)

	sleep(1)
# if char =='w':      #get the file from the source
#     print("w is found performing reverse scp")
#     command = "scp vip-lab@10.107.47.157:/home/vip-lab/Downloads/workspace/net.pkl /"
#     output = subprocess.run(command, shell=True, check=True)

#     with open(fp,'w') as fd:
#         ack='c'
#         fd.write(ack)

# command = "scp subp1.py vip-lab@10.107.47.157:/home/vip-lab/Downloads/workspace/"
# output = subprocess.run(command, shell=True, check=True) 