import os, random, shutil


def copyFile(fileDir):
    # 1
	pathDir = os.listdir(fileDir)

    # 2
	sample = random.sample(pathDir, 500)
	print(sample)
	
	# 3
	for name in sample:
		shutil.copyfile(fileDir+name, tarDir+name)
if __name__ == '__main__':
	fileDir = "/home/pmcn/workspace/CPE_AI/CPE_data/Cpe_cell2.0/RD/"
	tarDir = '/home/pmcn/workspace/CPE_AI/CPE_data/Cpe_cell2.0/RD_500/'
	copyFile(fileDir)