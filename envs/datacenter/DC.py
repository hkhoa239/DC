import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
from metrics.powermodels.PMLinear import PMLinear
from host.Host import Host

class DC():
	def __init__(self, num_hosts=6, num_vms=100, environment=None):
		self.num_hosts = num_hosts
		self.env = environment
		self.host_conf = {
			'cpu' : ['a1', 'a1', 'a1', 'a1', 'a1', 'a1', 'a1'], # MIPS
			'core' : [64, 64, 64, 128, 128, 128, 256], # GB
			# 'RAMRead' : [3000, 2000, 3000],
			# 'RAMWrite' : [3000, 2000, 3000],
			'ram' : [262144, 262144, 262144, 262144, 524288, 524288, 524288],
			# 'DiskRead' : [2000, 2000, 3000],
			# 'DiskWrite' : [2000, 2000, 3000],
			'bwUp' : [5000, 5000, 5000, 5000, 5000, 5000, 5000],
			'bwDown': [5000, 5000, 5000, 5000, 5000, 5000, 5000],
			'PwIdle': [10, 10, 10, 10, 10, 10, 10]
 		}
		
		self.vm_conf = {
			"LCMI": {
				"IPS": [10, 50, 100, 200, 1000, 10000],
				"RAMSize": [256, 256, 512, 512, 2048, 16384],
				"DiskSize": [256, 256, 512, 512, 2048, 16384],
				'BwUp' : [1000, 1000, 1000, 1000, 1000, 1000],
				'BwDown': [1000, 1000, 1000, 1000, 1000, 1000],
			},
			"CI": {
				"IPS": [4000, 10000, 20000],
				"RAMSize": [4096, 16384, 16384],
				"DiskSize": [16384, 16384, 16384],
				'BwUp' : [1000, 1000, 1000, 1000, 1000, 1000],
				'BwDown': [1000, 1000, 1000, 1000, 1000, 1000],
			},
			"MI": {
				"IPS": [1000, 4000, 50000],
				"RAMSize": [16384, 32764, 524288],
				"DiskSize": [16384, 32764, 524288],

				'BwUp' : [2000, 2000, 2000, 2000, 2000, 2000],
				'BwDown': [2000, 2000, 2000, 2000, 2000, 2000],
			},
			"HCMI": {
				"IPS": [100000, 150000],
				"RAMSize": [262144, 524288],
				"DiskSize": [262144, 524288],

				'BwUp' : [2000, 2000, 2000, 2000, 2000, 2000],
				'BwDown': [2000, 2000, 2000, 2000, 2000, 2000],
			}

			# "IPS": [50, 100, 200, 1000, 10000, 4000, 10000, 20000, 1000, 4000, 10000, 20000],
			# "RAMSize": [256, 512, 512, 2048, 16384, 4096, 16384, 16384, 16384, 32764, 524288, 131072],
			# "DiskSize": [256, 512, 512, 2048, 16384, 16384, 16384, 16384, 16384, 32764, 524288, 131072],
			# "Types": ["LCMI", "LCMI", "LCMI", "LCMI", "LCMI", "CI", "CI", "CI", "MI", "MI", "MI", "HCMI"]
		}
		


	def generateHosts(self):
		hosts = []
		for i in range(self.num_hosts):
			typeID = i%7 # np.random.randint(0,3) # i%3 #
			Cpu = self.host_conf['cpu'][typeID]
			Core = self.host_conf['core'][typeID]
			Ram = self.host_conf['ram'][typeID]
			# Disk_ = self.host_conf['DiskSize'][typeID]
			Bw_Up = self.host_conf['bwUp'][typeID]
			Bw_Down = self.host_conf['bwDown'][typeID]
			Power = PMLinear(idle=self.host_conf['PwIdle'][typeID], max=Core*10)
			# latency = 0
			# host = Host(ID=len(self.hostlist),IPS=IPS,RAM=Ram,Disk=Disk_,BwUp=Bw_Up,BwDown=Bw_Down,Latency=latency,Powermodel=Power,Envionment=self.environment)
			hosts.append((Cpu, Core, Ram, Bw_Up, Bw_Down, Power))
			# hosts.append((IPS, Ram, Disk_, Bw, 0, Power))
		return hosts
	
	# def generateVMs(self):
	# 	vms = []
	# 	for i in range(int(self.num_vms*0.7)):
	# 		LCMItype = i%6
	# 		Ips = self.vm_types["LCMI"]["IPS"][LCMItype]
	# 		Ram = self.vm_types["LCMI"]["RAMSize"][LCMItype]	
	# 		Disk_ = self.vm_types["LCMI"]["DiskSize"][LCMItype]
	# 		Bw_Up = self.host_types['BwUp'][LCMItype]
	# 		Bw_Down = self.host_types['BwDown'][LCMItype]
	# 		vms.append((Ips, Ram, Disk_, "LCMI", Bw_Up, Bw_Down))
			

	# 	for i in range(int(self.num_vms*0.1)):
	# 		CItype = i%3
	# 		Ips = self.vm_types["CI"]["IPS"][CItype]
	# 		Ram = self.vm_types["CI"]["RAMSize"][CItype]	
	# 		Disk_ = self.vm_types["CI"]["DiskSize"][CItype]
	# 		Bw_Up = self.host_types['BwUp'][CItype]
	# 		Bw_Down = self.host_types['BwDown'][CItype]
	# 		vms.append((Ips, Ram, Disk_, "CI", Bw_Up, Bw_Down))
			
		
	# 	for i in range(int(self.num_vms*0.1)):
	# 		MItype = i%3
	# 		Ips = self.vm_types["MI"]["IPS"][MItype]
	# 		Ram = self.vm_types["MI"]["RAMSize"][MItype]	
	# 		Disk_ = self.vm_types["MI"]["DiskSize"][MItype]
	# 		Bw_Up = self.host_types['BwUp'][MItype]
	# 		Bw_Down = self.host_types['BwDown'][MItype]
	# 		vms.append((Ips, Ram, Disk_, "MI", Bw_Up, Bw_Down))
			
		
	# 	for i in range(int(self.num_vms*0.1)):
	# 		HCMItype = i%1
	# 		Ips = self.vm_types["HCMI"]["IPS"][HCMItype]
	# 		Ram = self.vm_types["HCMI"]["RAMSize"][HCMItype]	
	# 		Disk_ = self.vm_types["HCMI"]["DiskSize"][HCMItype]
	# 		Bw_Up = self.host_types['BwUp'][HCMItype]
	# 		Bw_Down = self.host_types['BwDown'][HCMItype]
	# 		vms.append((Ips, Ram, Disk_, "HCMI", Bw_Up, Bw_Down))
			

	# 	return vms

		