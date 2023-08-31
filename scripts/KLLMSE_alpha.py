import tensorflow as tf
import keras
from ncps.tf import LTC
from ncps import wirings
import pandas as pd
import numpy as np

train_ds = pd.read_csv("data/train.csv")