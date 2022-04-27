import numpy as np
from numpy import *
import argparse

def normalize(x,a,b):
    max_x=max(x)
    min_x=min(x)
    return a+(x-min_x)/(max_x-min_x)*(b-a)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
                help = "path to our in processed files")
ap.add_argument("-o", "--output", required=True,
                help="path to output processed files")
args = vars(ap.parse_args())

if __name__=='__main__':
    x=np.loadtxt(args["input"])
    x=np.array(x)
    normalize_x=normalize(x,0,255)
    y = np.rint(normalize_x)
    np.savetxt(args["output"],y, fmt = '%g')
    print('x:',x,'\n',
          'normalize x:',normalize_x)