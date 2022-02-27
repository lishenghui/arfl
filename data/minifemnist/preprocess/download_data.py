import os
import sys

utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)
from download_util import download_file_from_google_drive

# file_id = "1jxZXAX12vOA_USU9g4AzXt1pMmf-rDUS"    #full dataset
file_id = "1gaw4VXhch2U1vOTmAhmeW8UtFHoQaD8i"  # 1000 clients
# file_id = "1gMCvgs4u0jhZ0isActSMaOdgAV6BntLO"  # 119 clients
destination = './femnist.zip'
download_file_from_google_drive(file_id, destination)
os.system('mkdir -p ../data')
os.system('unzip -o ' + destination + " -d ../data")
os.system('rm ' + destination)
