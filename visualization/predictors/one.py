import sys
sys.path.append('./utils')
from utils_imports import *
# train
res = 10
points = get_points("dump.sqlite", until=pd.Timestamp("2024-01-30"))
points["lon"] = points.geometry.x
points["lat"] = points.geometry.y

X = points[["lon", "lat"]].values
y = points["wait"].values
X.shape, y.shape

with open("models/kernel.pkl", "rb") as file:
    gpr = pickle.load(file)
    
print(gpr.regressor.optimizer)
gpr.regressor.optimizer = None
print(gpr.regressor.optimizer)

gpr = fit_gpr_silent(gpr, X, y)
raster_maker = raster_from_model(
    gpr,
    "world",
    res,
    show_uncertainties=True,
    verbose=True,
)

with open("one.pkl", "wb") as file:
    pickle.dump(raster_maker, file)