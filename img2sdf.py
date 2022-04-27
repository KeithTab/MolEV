import os
import torch
import argparse
from img2mol.inference import *

os.listdir("model/")
# point to our own training dataset
device = "cuda:0"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

img2mol = Img2MolInference(model_ckpt = "model/model.ckpt", device=device)

cddd_server = CDDDRequest()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
                help = "path to our in processed files")
args = vars(ap.parse_args())

class BatchRename():
    def __init__(self):
        self.path = 'RAD51_molp/'  

    def rename(self):
        filelist = os.listdir(self.path)   
        total_num = len(filelist)  
        i = 1  
        for item in filelist:
            if item.endswith('.jpg') or item.endswith('.png'):  
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), '0' + format(str(i), '0>2s') + '.jpg')#处理后的格式也为jpg格式的，当然这里可以改成png格式
                # restart to define our picture's name label
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()

os.system('cat > smiles_out.txt && chmod +x smiles_out.txt')
f = open("smiles_out.txt",'w+')
f.close
def read_path(file_pathname):
    
    for filename in os.listdir(file_pathname):
        res = img2mol(file_pathname+'/'+filename, cddd_server=cddd_server)
        res["mol"]
        print(res["smiles"],file=f)

read_path("args["input"]") 
"""
if here cannot run correctly, please change args["input"] to .png or .jpg filepath which you have stored  
"""
