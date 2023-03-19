import cv2

import numpy as np
import scipy.ndimage as ni
import os, sys
import re, json

COLOR_PREFIX = "COLOR_"
COLOR_SPACE = [
    "GRAY",
    "BGR",
    "RAW",
]

FLAGS = None
try:
    FLAGS = json.load(open("FLAGS"+cv2.__version__.replace(".","-")+".json"))
except:
    raise ValueError("FLAGS.json not opened.")
        
# with open("FLAGS.json") as f:
    # FLAGS = json.load(f)
        

class Image():
    """
    ## Wrapper of OpenCV in sight of translation function of image.

    ###  Categories of functions

    + Changing colorspaces
    + Geometric transformations
    + Image thresholding
    + Smoothing
    + Morphological tarnsformations
    + Image Gradients
    + Canny Edge Detection
    + Image Pyramids
    + Contours in OpenCV
    + Histgrams in OpenCV
    + Image Transforms in OpenCV
    ---
    + Template Matching
    + Hough Line Transform
    + Hough Circle Transform
    + Image Segmentation with Watershed Algorithm
    + Interactive Foreground Extarction GrabCut Algorithm
    """

    def __init__(self, name:str, flag:int=1, *arg, **kwargs):
        super().__init__()
        self.name = name
        self.raw = cv2.imread(name, flag)
        self.prod = self.raw.copy()
        self.COLOR = COLOR_SPACE[flag]
        
    ###### Image Property

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        return self
    
    @property
    def raw(self):
        return self._raw
    
    @raw.setter
    def raw(self, value):
        self._raw = value
        return self
    
    @property
    def prod(self):
        return self._prod
    
    @prod.setter
    def prod(self, value):
        self._prod = value
        return self

    @property
    def COLOR(self):
        return self._color
    
    @COLOR.setter
    def COLOR(self, value):
        self._color = value
        return self
    
    @property
    def histgram(self):
        return self._hisgtram
    
    @histgram.setter
    def histgram(self, value):
        self._histgram = value
        return self
    
    @property
    def ret(self):
        return self._ret
    
    @ret.setter
    def ret(self, value):
        self._ret = value
        return self
    
    @property
    def contours(self):
        return self._contours
    
    @contours.setter
    def contours(self, value):
        self._contours = value
        return self
    
    @property
    def hierarchy(self):
        return self._hierarchy
    
    @hierarchy.setter
    def hierarchy(self, value):
        self._hierarchy = value
        return self
    
    
    ###### Util
    
    def _getPosition(self, position):
        rown, coln, ch = self.prod.shape
        px = (coln - 1) * position[1] / 100
        py = (rown - 1) * position[0] / 100
        return (px, py)
    
    def bitwiseNot(self):
        self.prod = cv2.bitwise_not(self.prod)
        return self
    
    def fillHole(self):
        #Fill Holes処理
        self.prod = ni.binary_fill_holes(self.prod).astype("uint8") * 255
        return self
    
    ###### IO
    
    def show(self, name:str="prod"):
        if name == "prod":
            cv2.imshow(name, self.prod)
        else:
            cv2.imshow(name, self.raw)
        
        ### use async io
        while True:
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
        return self

    def write(self, name:str="image.png"):
        cv2.imwrite(name, self.prod)
        return self
    
    ###### Color

    def cvtColor(self, flag:str):
        self.prod = cv2.cvtColor(self.prod, FLAGS[self.COLOR+"2"+flag])
        self.COLOR = re.sub(self.COLOR+"2","",flag)
        return self
    
    ###### Geometry
    
    def scale(self, x=1.0):
        if x >= 1.0:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA
            
        self.prod= cv2.resize(self.prod, None, fx=x, fy=x, interpolation=interpolation)
        return self
    
    def resize(self, x=1.0, y=1.0):
        if x >= 1.0 and y >= 1.0:
            interpolation = cv2.INTER_CUBIC
        else:
            interpoltion = cv2.INTER_LINEAR
            
        self.prod = cv2.resize(self.prod, None, fx=x, fy=y, interpolation=interpolation)
        return self
        
    def translate(self, pt:list):
        tx, ty = self._getPosition(pt)
        M = np.float32([[1,0,tx],[0,1,ty]])
        
        self.prod = cv2.warpAffine(self.prod, M, self.prod.shape[1::-1])
        return self
        
    def rotate(self, cp:list, degree=0.0, scale=1.0):
        cp = self._getPosition(cp)
        M = cv2.getRotationMatrix2D(c, degree, scale)
        self.prod= cv2.warpAffine(self.prod, M, self.prod.shape[1::-1])
        return self
    
    def affineTransform(self):
        print("Not yet implemented")
        return self
        
    def perspectiveTransform(self, pt1:list, pt2:list, pt3:list, pt4:list):
        pts_pre = np.float32([
            self._getPosition(pt1),
            self._getPosition(pt2),
            self._getPosition(pt3),
            self._getPosition(pt4),
        ])
        pts_aft = np.float32([
            self._getPosition((0,0)),
            self._getPosition((100,0)),
            self._getPosition((0,100)),
            self._getPosition((100,100)),
        ])
        M = cv2.getPerspectiveTransform(pts_pre, pts_aft)
        self.prod= cv2.warpPerspective(self.prod, M, self.prod.shape[1::-1])
        return self
    
    def distanceTransform(self):
        self.prod = cv2.distanceTransform(self.prod, cv2.DIST_L2, 5)
        return self
    ###### Theresholding
    
    def threshBinary(self, ret=127, max=255, inv=False):
        if not inv:
            ret, thresh = cv2.threshold(self.prod, ret, max, cv2.THRESH_BINARY)
        else:
            ret, thresh = cv2.threshold(self.prod, ret, max, cv2.THRESH_BINARY_INV)
        
        self.prod= thresh
        return self
    
    def threshToZero(self, ret=127, max=255, inv=False):
        if not inv:
            ret, thresh = cv2.threshold(self.prod, ret, max, cv2.THRESH_TOZERO)
        else:
            ret, thresh = cv2.threshold(self.prod, ret, max, cv2.threshold_TOZERO)
            
        self.prod= thresh
        return self
    
    def threshTrunc(self, ret=127, max=255):
        ret, thresh = cv2.threshold(self.prod, ret, max, cv2.THRESH_TRUNC)
        self.prod= thresh
        return self

    
    def threshOTSU(self, ret=0, max=255):
        ret, thresh = cv2.threshold(self.prod, ret, max, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        self.prod= thresh
        return self
        
    ## implement later
    def adaptiveThreshMean(self):
        pass
    
    
    ###### Smoothing
    
    def blur(self, kernel_shape:list=(5,5)):
        self.prod= cv2.blur(self.prod, kernel_shape)
        return self
    
    def gaussianBlur(self, kernel_shape:list=(5,5), deviation:list=(0,0)):
        self.prod = cv2.GaussianBlur(self.prod, kernel_shape, *deviation)
        return self
    
    ###### Morphological Transformations
    
    def erode(self, kernel_shape=(5,5), iter=1):
        kernel = np.ones(kernel_shape, np.uint8)
        self.prod = cv2.erode(self.prod, kernel, iterations=iter)
        return self
    
    def dilate(self, kernel_shape=(5,5), iter=1):
        kernel = np.ones(kernel_shape, np.uint8)
        self.prod = cv2.dilate(self.prod, kernel, iterations=iter)
        return self
    
    def opening(self, kernel_shape=(5,5), iter=1):
        kernel = np.ones(kernel_shape, np.uint8)
        self.prod = cv2.morphologyEx(self.prod, cv2.MORPH_OPEN, kernel, iterations=iter)
        return self
    
    def closing(self, kernel_shape=(5,5), iter=1):
        kernel = np.ones(kernel_shape, np.uint8)
        self.prod = cv2.morphologyEx(self.prod, cv2.MORPH_CLOSE, kernel, iterations=iter)
        return self
    
    def morphGrad(self, kernel_shape=(5,5)):
        kernel = np.ones(kernel_shape, np.uint8)
        self.prod = cv2.morphologyEx(self.prod, cv2.MORPH_GRADIENT, kernel)
        return self
    
    def morphTopHat(self, kernel_shape=(5,5)):
        kernel = np.ones(kernel_shape, np.uint8)
        self.prod = cv2.morphologyEx(self.prod, cv2.MORPH_TOPHAT, kernel)
        return self
    
    def morphBlackHat(self, kernel_shape=(5,5)):
        kernel = np.ones(kernel_shape, np.uint8)
        self.prod = cv2.morphologyEx(self.prod, cv2.MORPH_BLACKHAT, kernel)
        return self
    
    ###### Contours
    
    def findContours(self, flag=cv2.CHAIN_APPROX_SIMPLE):
        # self.contours, self.hierarchy = cv2.findContours(self.prod, cv2.RETR_TREE, flag)
        self.contours, self.hierarchy = cv2.findContours(self.prod, cv2.RETR_TREE,flag)
        return self
    
    def drawContours(self, contour=-1, color=(0,255,255)):
        if contour == -1:
            self.prod = cv2.drawContours(self.raw, self.contours, contour, color, 3)
        else:
            contour = self.contours[contour]
            self.prod = cv2.drawContours(self.raw, contour, 0, color, 3)
            
        return self
    
#     def watershed(self):
#         self.
#         .opening()\
#         .
        