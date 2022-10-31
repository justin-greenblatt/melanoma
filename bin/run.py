from Model import ConvPool
from Preprocess import Otsu64

if __name__ == "__main__":
    model = ConvPool()
    data = Otsu64("jpeg", "processed")
    model.train(data)
