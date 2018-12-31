import os
from subprocess import Popen, PIPE
import tempfile

def tag(sentence: str):
	fd, path = tempfile.mkstemp()
	
	try:
		with os.fdopen(fd, 'w') as tmp:
			tmp.write(sentence)
	
		command = [
			'wcrft',
			'nkjp_e2.ini',
			'-d', 
			'/usr/local/lib/python2.7/dist-packages/wcrft-1.0.0-py2.7.egg/wcrft/model/model_nkjp10_wcrft_e2/', 
			'-i',
			'txt',
			path, 
			'-o',
			'iob-chan'
		]

		proc = Popen(' '.join(command), shell=True, stdout=PIPE)
		output, _ = proc.communicate()
		output = output.decode('utf-8')
		output = [tuple(entry.strip().split('\t')) for entry in output.strip().split('\n')]
		output = {u: (b, p) for u, b, p in output}
		
		return output

	finally:
		os.remove(path)
