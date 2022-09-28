from Image import Image
from settings import config

if __name__ == "__main__":
    testImage = Image(config["paths"]["image_unit_test"])
    print(testImage.imageId)
    print(testImage.data)
    testImage.cropImage()
    testImage.getOtsuBinary("croped")
    testImage.resizeImage(key = "otsu")
    testImage.organize("resized","processed")
    testImage.augmentRotate(70,"resized","processed")
    testImage.augmentRotate(180,"resized","processed")
