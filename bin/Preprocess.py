from Image import Image
from settings import config
import pandas
from tensorflow import keras
import os
from random import choice, randint

class Otsu64:
    def __init__(self, rawDataKey, outDataKey):
        imageList =  [os.path.join(config["paths"][rawDataKey],a) for a in os.listdir(config["paths"][rawDataKey])]
        
        self.images = list([Image(d) for d in imageList])
        rConfig = list([(kv, n, r, s, f) for kv,n,r,s,f in 
                       zip(config["data_augmentation"]["rotation_targets"].split(','),
                           config["data_augmentation"]["rotation_replicates"].split(','),
                           config["data_augmentation"]["rotation_degrees_range"].split(','),
                           config["data_augmentation"]["rotation_degree_step"].split(','),
                           config["data_augmentation"]["rotation_flip_random"].split(','))])

        self.augmented = [] 
        for i in self.images:

            i.cropImage()
            i.getOtsuBinary("croped")
            i.remove("croped")
            i.resizeImage(key = "otsu")
            i.remove("otsu")
            i.organize("resized", outDataKey)
       
            for c in rConfig:
                k,v = c[0].split(':')
                degreesList = list(range(*list([int(d) for d in c[2].split(':')]),int(c[3])))
                flip = None
                if c[4].lower() == "yes":
                    flip = choice([0,1,-1,None])
                if all([i.data[key] == target for key,target in zip(k.split('^'), v.split('^'))]):

                    for replica in range(int(c[1])):
                        degrees = choice(degreesList)
                        degreesList.remove(degrees)
                        self.augmented.append(i.augmentRotateFlip(degrees,"resized", outDataKey, flip))
        self.train, self.validation = keras.utils.image_dataset_from_directory(
                    config["paths"][outDataKey],
                    labels = 'inferred',
                    label_mode = config["dataset"]["label_mode"],
                    color_mode = config["dataset"]["color_mode"],
                    batch_size = int(config["dataset"]["batch_size"]),
                    image_size = list([int(a) for a in config["dataset"]["image_size"].split(',')]),
                    validation_split = float(config["dataset"]["validation_split"]),
                    subset = "both",
                    seed = randint(0,200000000)
                    )

