from prepare_label import *
from Segnet import Segnet

data, label = prep_data((256,256), 5, 2)

segnet = Segnet()
segnet.train(data, label, 50, 18)
