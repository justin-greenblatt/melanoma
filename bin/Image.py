import cv2
import pandas
import os
import json
from settings import config
import shutil

class Image:
    metadata = pandas.read_csv(config["paths"]["metadata"])
    def __init__(self, directorie):
        self.format = directorie.split('.')[-1]
        self.imageId = directorie.split('/')[-1].split('.')[0]
        self.dirDict = {self.format: directorie}

        dataDict = self.__class__.metadata[self.__class__.metadata["image_name"] == self.imageId].to_dict()
        self.data = {k : list(dataDict[k].values())[0] for k in dataDict}

        img = cv2.imread(self.dirDict[self.format])
        self.shape = img.shape
        self.pixelValue = img.dtype
        self.children = []

    def setId(self,newId):
        self.imageId = newId

    def setDirDict(self,newDirDict):
        self.dirDict = newDirDict

    def getHist(self, key = None):
        if key == None:
            key = self.format
        img = cv2.imread(self.dirDict[key])
        b = cv2.calcHist([img],[0],None,[256],[0,256])
        g = cv2.calcHist([img],[0],None,[256],[0,256])
        r = cv2.calcHist([img],[0],None,[256],[0,256])
        path = os.path.join(config["paths"]["histograms"], self.imageId + ".json")
        outFile = open(path, 'w')
        json.dump({"b":b.tolist(), "g":g.tolist(), "r":r.tolist()}, outFile)
        outFile.close()
        self.dirDict["histogram"] = path

    def getHistBw(self, key = None):

        if key == None:
            key = self.format
        img = cv2.imread(self.dirDict[key], cv2.IMREAD_GRAYSCALE)
        bw = cv2.calcHist([img],[0],None,[256],[0,256])
        path = os.path.join(config["paths"]["histograms_bw"], self.imageId + ".json")
        outFile = open(path, 'w')
        json.dump({"bw":bw.tolist()}, outFile)
        outFile.close()
        self.dirDict["histogram_bw"] = path

    def getOtsuBinary(self, key = None):
        
        path = os.path.join(config["paths"]["otsu"], self.imageId + "." + self.format)
        if key == None:
            key = self.format
        img = cv2.imread(self.dirDict[key], cv2.IMREAD_GRAYSCALE)
        (thresh, otsu) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(path, otsu)
        self.dirDict["otsu"] = path

    
    def cropImage(self, key = None):

        if key == None:
            key = self.format
        img = cv2.imread(self.dirDict[self.format])
        h = self.shape[0]
        l = self.shape[1]
        path = os.path.join(config["paths"]["croped"], self.imageId + "." + self.format)

        if h < l:
            s = round((l - h)/2)
            croped = img[0:h, s:s + h]
            cv2.imwrite(path, croped)
            self.dirDict["croped"] = path
 
        elif h > l:
            s = round((h - l)/2)
            croped = img[s:s + l, 0:l]
            cv2.imwrite(path, croped)
            self.dirDict["croped"] = path
 
        else:
            self.dirDict["croped"] = self.dirDict[self.format]

    def resizeImage(self,side = 64, key = None):
        if key == None:
            key = self.format
        img = cv2.imread(self.dirDict[key])
        resized = cv2.resize(img, (side, side), interpolation = cv2.INTER_LINEAR)
        path = os.path.join(config["paths"]["resized"], self.imageId + "." + self.format)
        cv2.imwrite(path, resized)
        self.dirDict["resized"] = path

    def organize(self, originKey, destinationKey):
        folder = os.path.join(config["paths"][destinationKey], "class_" + self.data['benign_malignant'])
        newImageDir = os.path.join(folder, self.data['benign_malignant']  + '_' + self.imageId + "." + self.format)

        if not os.path.isdir(folder):
            os.mkdir(folder)
        shutil.copyfile(self.dirDict[originKey], newImageDir)
        self.dirDict[destinationKey] = newImageDir

    def remove(self, key):
        os.remove(self.dirDict.pop(key))

    def rotate(self, degrees, key = None, flip = None):
        if key == None:
            key = self.format
        
        img = cv2.imread(self.dirDict[key])
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), degrees, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        if flip != None:
            rotated = cv2.flip(rotated, flip)
        path = os.path.join(config["paths"]["rotated"], self.imageId + "." + self.format)
        cv2.imwrite(path, rotated)
        self.dirDict["rotated"] = path
        

    def copy(self):
        child = Image(self.dirDict[self.format])
        newId = f'{self.imageId}_{len(self.children) + 1}'.strip('\'')
        child.setId(newId)
        self.children.append(child)
        return child

    def augmentRotateFlip(self, degrees, key, outDir, flip = None):
        new = self.copy()
        new.setDirDict(self.dirDict)
        new.rotate(degrees, key, flip)
        new.organize("rotated", outDir)

