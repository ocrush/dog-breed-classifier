# utility script to copy representative file of predicted image.  For now, just grab the first file
from glob import glob
import os
from shutil import copyfile
results = [item for item in sorted(glob("../../../dogImages/train/*/"))]

rep_files = [os.path.join(path,os.listdir(path)[0]) for path in results]
c = os.getcwd()
[copyfile(src, os.getcwd() + "/" + os.path.basename(src).rsplit("_", 1)[0] + ".jpg" ) for src in rep_files]
print("Done")