import os
import sys

utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)
from download_util import download_file_from_google_drive

# import download_data
#
#
# file_id = "1VV3kw1ToYGAZHXemRRMgayd-zhkCnJPL"   #34 clients
file_id = "1JSjbZFhUQIc8l65QoEcLrPUQ0wvuXK1s"     #71 clients
destination = './dataset.zip'
download_file_from_google_drive(file_id, destination)
os.system('mkdir -p ../data')
os.system('unzip -o ' + destination + " -d ../data")
os.system('rm ' + destination)
