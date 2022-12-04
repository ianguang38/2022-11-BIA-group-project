import matplotlib.pyplot as plt
import random
def predict(fn):
    img=plt.imread(fn)
    print(img.shape)
    return(random.random())
