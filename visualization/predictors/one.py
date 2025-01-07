import sys
sys.path.append('./utils')
from utils.utils_data import get_points
from utils.utils_map import raster_from_model
import pickle
from utils.utils_models import fit_gpr_silent


# train
res = 10
points = get_points("dump.sqlite")
points["lon"] = points.geometry.x
points["lat"] = points.geometry.y

X = points[["lon", "lat"]].values
y = points["wait"].values
X.shape, y.shape

with open("models/kernel.pkl", "rb") as file:
    gpr = pickle.load(file)
    
gpr.regressor.optimizer = None

gpr = fit_gpr_silent(gpr, X, y)
raster_maker = raster_from_model(
    gpr,
    "world",
    res,
    show_uncertainties=True,
    version="dev",
    verbose=True,
)

with open("one.pkl", "wb") as file:
    pickle.dump(raster_maker, file)