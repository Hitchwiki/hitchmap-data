import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
import rasterio.plot
from IPython.display import Image, display
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from tqdm import tqdm
from tqdm.auto import tqdm

from models import *

tqdm.pandas()

RESOLUTION = 2


def save_numpy_map(
    map, region="world", method="ordinary", kind_of_map="map", resolution=RESOLUTION
):
    map_path = f"intermediate/{kind_of_map}_{method}_{region}_{resolution}.txt"
    np.savetxt(map_path, map)


def load_numpy_map(
    region="world", method="ordinary", kind_of_map="map", resolution=RESOLUTION
):
    map_path = f"intermediate/{kind_of_map}_{method}_{region}_{resolution}.txt"
    return np.loadtxt(map_path)


def load_raster(region="world", method="ordinary", resolution=RESOLUTION):
    map_path = f"intermediate/map_{method}_{region}_{resolution}.tif"
    return rasterio.open(map_path)


def get_points_in_region(points, region="world"):
    # set lat long boundaries of different scopes of the map

    maps = {
        "germany": [3.0, 48.0, 16.0, 55.0],
        "spain": [-8.0, 36.0, 3.0, 43.0],
        "spain_france": [-8.0, 36.0, 6.0, 50.0],
        "europe": [-12.0, 35.0, 45.0, 71.0],
        "world": [-180.0, -85.0, 180.0, 85.0],  # 85 lat bc of 3857
        "small": [12.0, 52.0, 15.0, 54.0],
        "africa": [-20.0, -35.0, 60.0, 40.0],
        "asia": [40.0, 0.0, 180.0, 85.0],
        "north_america": [-180.0, 0.0, -20.0, 85.0],
        "south_america": [-90.0, -60.0, -30.0, 15.0],
        "australia": [100.0, -50.0, 180.0, 0.0],
        "middle_africa": [-10.0, -35.0, 60.0, 20.0],
        "artificial": [8.0, -1.0, 30.0, 1.0],
        "greenland": [-80.0, 60.0, -10.0, 85.0],
    }
    map_boundary = maps[region]

    # create boundary polygon
    # nodes stars bottom left & clockwise
    polygon = Polygon(
        [
            (map_boundary[0], map_boundary[1]),
            (map_boundary[0], map_boundary[3]),
            (map_boundary[2], map_boundary[3]),
            (map_boundary[2], map_boundary[1]),
            (map_boundary[0], map_boundary[1]),
        ]
    )
    polygon = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon])
    polygon = polygon.to_crs(epsg=3857)

    # extract points within polygon
    points = points[points.geometry.within(polygon.geometry[0])]

    return points, polygon, map_boundary


def raster_from_model(
    model, region, resolution=RESOLUTION, show_uncertainties=False, verbose=False
):
    model_name = type(model).__name__

    raster_maker = MapBasedModel(
        method=model_name, region=region, resolution=resolution, verbose=verbose
    )

    raster_maker.get_map_grid()
    X, Y = raster_maker.X, raster_maker.Y
    grid = np.array((X, Y))
    height = X.shape[0]
    map = np.empty((0, height))
    if show_uncertainties:
        uncertainty_map = np.empty((0, height))

    # transposing the grid enables us to iterate over it vertically
    # and single elements become lon-lat pairs that can be fed into the model
    print("Compute rows of pixels...")
    for vertical_line in tqdm(grid.transpose(), disable=not verbose):
        if show_uncertainties:
            pred, stdv = model.predict(vertical_line, return_std=True)
            uncertainty_map = np.vstack((uncertainty_map, stdv))
        else:
            pred = model.predict(vertical_line)
        map = np.vstack((map, pred))

    # because we vstacked above
    map = map.T
    if show_uncertainties:
        uncertainty_map = uncertainty_map.T

    save_numpy_map(map, region=region, method=model_name, resolution=resolution)
    if show_uncertainties:
        save_numpy_map(
            uncertainty_map,
            region=region,
            method=model_name,
            kind_of_map="uncertainty",
            resolution=resolution,
        )

    raster_maker.raw_raster = map
    if show_uncertainties:
        raster_maker.raw_uncertainties = uncertainty_map

    raster_maker.save_as_raster()

    return raster_maker


def map_from_model(
    model,
    region,
    resolution=RESOLUTION,
    show_uncertainties=False,
    verbose=False,
    discrete_uncertainties=False,
    return_raster=False,
    final=False,
):
    raster_maker = raster_from_model(
        model,
        region,
        resolution,
        show_uncertainties=show_uncertainties,
        verbose=verbose,
    )
    raster_maker.build_map(
        show_uncertainties=show_uncertainties,
        discrete_uncertainties=discrete_uncertainties,
        final=final,
        show_cities=True,
        show_roads=True,
    )

    if return_raster:
        return raster_maker


def show_map(path="maps/map.png"):
    display(Image(filename=path))


# draw a map from pre-computed raster
def generate_highres_map(
    region,
    resolution=RESOLUTION,
    show_uncertainties=False,
    verbose=False,
    discrete_uncertainties=False,
    return_raster=False,
    final=False,
    figsize=10,
    show=True,
):
    res = 10

    m = MapBasedModel(
        method="TransformedTargetRegressorWithUncertainty",
        region=region,
        resolution=res,
    )

    if show_uncertainties:
        m.raw_uncertainties = load_numpy_map(
            region=region,
            method="TransformedTargetRegressorWithUncertainty",
            kind_of_map="uncertainty",
            resolution=res,
        )

    m.build_map(
        final=final,
        show_states=True,
        show_cities=show,
        show_roads=show,
        show_uncertainties=show_uncertainties,
        discrete_uncertainties=discrete_uncertainties,
        figsize=figsize
    )


def map(region="world"):
    show_map(path=f"final_maps/{region}.png")