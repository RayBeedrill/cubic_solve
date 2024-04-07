from detection import detectColors
from model_solve import nn_solve_cube
from model import get_model

encoding = detectColors()
encodingFormated = " ".join(encoding)
model = get_model()
model.load_weights("auto.weights.h5")
nn_solve_cube(encodingFormated, model)