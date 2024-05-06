# this is estimated to take about several hours to run
from utils_imports import *

with open("models/gp_model.pkl", "rb") as file:
    gpr = pickle.load(file)

map_from_model(gpr, region="asia", show_uncertainties=True, verbose=True, resolution=10)

map_from_model(gpr, region="world", show_uncertainties=True, verbose=True, resolution=10)