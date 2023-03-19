import cv2
import os, json

version = cv2.__version__.replace(".","-")
flagfile = "FLAGS"+version+".json"

if os.path.exists(flagfile):
   pass
else:
    FLAGS = {
    i.replace("COLOR_",""): eval("cv2." + i) for i in dir(cv2) if i.startswith("COLOR_")
}
    with open(flagfile, 'w') as f:
        json.dump(FLAGS, f, indent=4, sort_keys=True)