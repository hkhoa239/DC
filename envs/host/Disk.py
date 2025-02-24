
class Disk():
	# Size = Size of disk in MB
	# Read = Read speed in MBps
	# Write = Write speed in MBps
	def __init__(self, Size, Read, Write):
		self.size = Size
		self.read = Read
		self.write = Write
		self.used_size, self.used_read, self.used_write = 0, 0, 0
	
	def getAvailableSize(self):
		return self.size - self.used_size
	def getAvailableRead(self):
		return self.read - self.used_read
	def getAvailableWrite(self):
		return self.write - self.used_write

	def allocate(self, size, read, write):
		if self.getAvailableSize() >= size and self.getAvailableRead() >= read and self.getAvailableWrite() >= write:
			self.used_size += size
			self.used_read += read
			self.used_write += write
   
	def release(self, size, read, write):
		self.used_size = max(0, self.used_size - size)
		self.used_read = max(0, self.used_read - read)
		self.used_write = max(0, self.used_write - write)