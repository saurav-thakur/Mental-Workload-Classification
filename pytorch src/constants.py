# file paths
DATA_FOLDER_PATH = "../dataset"
DATASET_PATH = "../dataset/WLDataCW.mat"
MODEL_PATH = "../trained_models/model_001.pt"

# can be only fft or pca
PREPROCESSED = "fft"

# hyperparameters
IMAGE_HEIGHT = 62
IMAGE_WIDTH = 62
LEARNING_RATE = 0.01
NUM_EPOCHS = 20
HIDDEN_LAYERS = 10
BATCH_SIZE = 32
NUMBER_OF_CLASS = 2

# plots and save model
PLOTS = True
SAVE_MODEL = False

if PREPROCESSED == "pca":
    IMAGE_CHANNELS = 1
else:
    IMAGE_CHANNELS = 2