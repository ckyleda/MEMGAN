import PIL
from PIL import Image
import os

count = 1
error_count = 0
files = os.listdir("../LSUN/lsun/living_room/all/")
for file in files:
    try:
        #if '.webp' in file:
            #print (file)
            #raise Exception("Incompatible file")
        image = Image.open("../LSUN/lsun/living_room/all/" + file)
    except:
        os.remove("../LSUN/lsun/living_room/all/" + file)
        error_count += 1
        print("Files removed: {}".format(error_count))
    if count % 100 == 0:
        print("Checked {}".format(count))
    count+=1


