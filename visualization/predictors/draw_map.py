# this is estimated to take about several hours to run
from utils_imports import *

with open("models/best_gp.pkl", "rb") as file:
    gpr = pickle.load(file)

map_from_model(gpr, region="world", show_uncertainties=True, verbose=True, resolution=10)