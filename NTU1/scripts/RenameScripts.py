import os
SRC = r"D:\\OpenSource\\DoAnTotNghiep\\H_SignDatasets\\3\\"
path = os.chdir(SRC)
print(path)
index = 1
for file in os.listdir(path):
    new_file_name = "turn_right_{}.jpg".format(index)
    os.rename(file, new_file_name)
    index = index + 1
print('Done')
