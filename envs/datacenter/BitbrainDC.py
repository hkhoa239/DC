import numpy as np
from host.Disk import Disk
from host.RAM import RAM
from host.Bandwitdh import Bandwidth
from powermodels.PMConstant import PMConstant

class BitbrainDC():
	def __init__(self, num_hosts):
		self.num_hosts = num_hosts
		self.types = {
			'IPS' : [5000, 10000, 50000], # Some containers have IPS requirement as high as 40k
			'RAMSize' : [3000, 4000, 8000],
			'RAMRead' : [3000, 2000, 3000],
			'RAMWrite' : [3000, 2000, 3000],
			'DiskSize' : [30000, 40000, 80000],
			'DiskRead' : [2000, 2000, 3000],
			'DiskWrite' : [2000, 2000, 3000],
			'BwUp' : [1000, 2000, 5000],
			'BwDown': [2000, 4000, 10000],
			'Power' : [1, 1, 1]
 		}
		self.hosts = []

	def generateHosts(self):
		hosts = []
		for i in range(self.num_hosts):
			typeID = i%3 # np.random.randint(0,3) # i%3 #
			IPS = self.types['IPS'][typeID]
			Ram = RAM(self.types['RAMSize'][typeID], self.types['RAMRead'][typeID], self.types['RAMWrite'][typeID])
			Disk_ = Disk(self.types['DiskSize'][typeID], self.types['DiskRead'][typeID], self.types['DiskWrite'][typeID])
			Bw = Bandwidth(self.types['BwUp'][typeID], self.types['BwDown'][typeID])
			Power = PMConstant(self.types['Power'][typeID])
			hosts.append((IPS, Ram, Disk_, Bw, 0, Power))
		return hosts