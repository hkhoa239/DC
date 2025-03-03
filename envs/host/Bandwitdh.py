
class Bandwidth():
	# Downlink = Downlink speed in MBps
	# Uplink = Uplink speed in MBps
	def __init__(self, Downlink, Uplink):
		self.downlink = Downlink
		self.uplink = Uplink
		self.used_downlink, self.used_uplink = 0, 0

	def getAvailableDownlink(self):
		return self.downlink - self.used_downlink
	def getAvailableUplink(self):
		return self.uplink - self.used_uplink

	def allocate(self, downlink, uplink):
		if self.getAvailableDownlink() >= downlink and self.getAvailableUplink() >= uplink:
			self.used_downlink += downlink
			self.used_uplink += uplink 
   
	def release(self, downlink, uplink):
		self.used_downlink = max(0, self.used_downlink - downlink)
		self.used_uplink = max(0, self.used_uplink - uplink)