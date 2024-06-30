file_path = 'talk.txt'

from jtop import jtop
import csv
import time
import pickle
import psutil
from time import sleep

ramlist=[]
gpulist=[]
powerlist=[]
swaplist=[]
cpulist=[]
modelname='mobilenet'
BATCH_SIZE=16
performance_list=[]

energy_list=[]

energy_bw_dict={'ene':[],'bw':[],'timestamp':[]}

def readFile(path):
	with open(path, 'r') as file:
    		# Read data from the file
    		data = file.read()
    		return data

char=0

# read the file until 'S' is not found

while(char!='S'):
	char=readFile(file_path)

init_time=time.time()
bwinitial= psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv

tot_energy=0
# while(True):
while(char!='E'):
	
	start_time=time.time()
	char=readFile(file_path)
	with jtop() as jetson:
    # jetson.ok() will provide the proper update frequency
    		#while jetson.ok():
        		# Read tegra stats
			mydir=jetson.stats
			#print(jetson.stats)
			#print(jetson.power)
			ramlist.append(100*mydir["RAM"])
			powerlist.append(mydir["power avg"])
			swaplist.append(100*mydir["SWAP"])
			gpulist.append(mydir["GPU1"])
			cpulist.append(0.25*mydir["CPU1"]+0.25*mydir["CPU2"]+ 0.25*mydir["CPU3"]+0.25*mydir["CPU4"])
			
	end_time=time.time()	
	tot_time= end_time-start_time
	tot_energy=tot_energy + (tot_time*mydir["power avg"])
	energy_list.append(tot_energy)
	energy_bw_dict['ene'].append(tot_energy)
	energy_bw_dict['timestamp'].append(start_time-init_time)
	#for BW
	bwvalue = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
	energy_bw_dict['bw'].append((bwvalue-bwinitial)/1000)	#saving in KBytes

	sleep(0.1)


performance_list.append(max(ramlist))
performance_list.append(max(powerlist))
performance_list.append(max(swaplist))
performance_list.append(max(gpulist))
performance_list.append(max(cpulist))
performance_list.append(tot_energy)

with open('energy.pkl','wb') as f:
	pickle.dump(energy_bw_dict,f)

#append list to csv
csv_path='performance_measures/'+str(modelname)+str(BATCH_SIZE)+'.csv'
with open(csv_path, 'a', newline='') as csv_file:
         writer = csv.writer(csv_file)
         writer.writerow(performance_list)

print("performance list is:")
print(performance_list)
print("\n")
print("total energy is (Joules) :")
print(tot_energy/1000)
print("\n")
print(len(powerlist))

			


