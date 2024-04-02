import rasterio
import rasterio.plot
from rasterio.crs import CRS
from matplotlib import pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import rasterio.mask
from shapely.geometry import Point
from geopandas import GeoDataFrame
import matplotlib.colors as colors
from matplotlib import cm
from shapely.geometry import Polygon
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
import osmnx
from pathlib import Path

from tqdm.auto import tqdm

tqdm.pandas()


RESOLUTION = 20  # pixel per degree


def define_raster(polygon, map, res=RESOLUTION):
    xx, yy = polygon.geometry[0].exterior.coords.xy

    # Note above return values are of type `array.array`
    xx = xx.tolist()
    yy = yy.tolist()

    degree_width = int(map[2] - map[0])
    degree_height = int(map[3] - map[1])
    pixel_width = degree_width * res
    pixel_height = degree_height * res

    return xx, yy, pixel_width, pixel_height


def save_raster(Z, polygon, map, map_path, res=RESOLUTION):

    polygon_vertices_x, polygon_vertices_y, pixel_width, pixel_height = define_raster(
        polygon, map, res
    )
    # https://gis.stackexchange.com/questions/425903/getting-rasterio-transform-affine-from-lat-and-long-array

    # lower/upper - left/right
    ll = (polygon_vertices_x[0], polygon_vertices_y[0])
    ul = (polygon_vertices_x[1], polygon_vertices_y[1])  # in lon, lat / x, y order
    ur = (polygon_vertices_x[2], polygon_vertices_y[2])
    lr = (polygon_vertices_x[3], polygon_vertices_y[3])
    cols, rows = pixel_width, pixel_height

    # ground control points
    gcps = [
        GCP(0, 0, *ul),
        GCP(0, cols, *ur),
        GCP(rows, 0, *ll),
        GCP(rows, cols, *lr),
    ]

    # seems to need the vertices of the map polygon
    transform = from_gcps(gcps)

    # cannot use np.longdouble to write to tif
    Z = np.double(Z)
    Z = np.round(Z, 0)

    # save the colored raster using the above transform
    # TODO find out why raster is getting smaller in x direction when stored as tif (e.g. 393x700 -> 425x700)
    with rasterio.open(
        map_path,
        "w",
        driver="GTiff",
        height=Z.shape[0],
        width=Z.shape[1],
        count=1,
        crs=CRS.from_epsg(3857),
        transform=transform,
        dtype=Z.dtype,
    ) as destination:
        destination.write(Z, 1)


def get_map_grid(polygon, map, res=RESOLUTION):
    xx, yy, pixel_width, pixel_height = define_raster(polygon, map, res)
    x = np.linspace(xx[0], xx[2], pixel_width)
    # mind starting with upper value of y axis here
    y = np.linspace(yy[2], yy[0], pixel_height)
    X, Y = np.meshgrid(x, y)
    # higher precision prevents pixels far away from the points to be 0/ nan
    X = np.longdouble(X)
    Y = np.longdouble(Y)
    return X, Y


def get_points_in_region(points, region="world"):
    # set lat long boundaries of different scopes of the map

    maps = {
        "germany": [5.0, 48.0, 15.0, 55.0],
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


def build_map(
    map_path,
    method="ORDINARY",
    points=None,
    all_points=None,
    region="world",
    polygon=None,
    show_cities=True,
    show_roads=True,
    show_spots=True,
):
    print("Loading information about states...")
    states = gpd.read_file("map_features/states/ne_10m_admin_1_states_provinces.shp")
    states = states.to_crs(epsg=3857)

    # use smaller units for Russia
    # country level except for Canada, Russia, USA, Australia, China, Brazil, India, Indonesia
    states = states[states.admin != "Antarctica"]

    # a state is hitchhikable if there are hitchhiking spots in it
    def check_hitchhikability(state):
        points_in_state = points[points.geometry.within(state.geometry)]
        return len(points_in_state) > 0

    print("Checking hitchhikability for each state...")
    states["hh"] = states.progress_apply(check_hitchhikability, axis=1)
    # define the heatmap color scale

    # TODO smoother spectrum instead of buckets
    buckets = [
        "grey",  # not enough data
        "#008200",  # dark green
        "#00c800",  # light green
        "green",  # green
        "#c8ff00",  # light yellow
        "#ffff00",  # yellow
        "#ffc800",  # light orange
        "#ff8200",  # dark orange
        "red",  # red
        "#c80000",  # dark red
        "#820000",  # wine red
        "blue",  # not necessary to color (eg sea)
    ]

    cmap = colors.ListedColormap(buckets)

    max_wait = (
        all_points.wait.max() + 0.1
    )  # to get at least this value as maximum for the colored buckets
    num_scale_colors = len(buckets) - 2  # because of upper and lower bucket
    # build log scale starting at 0 and ending at max wait
    base = (max_wait + 1) ** (1 / num_scale_colors)

    def log_scale(x):
        return base**x - 1

    # how to prevent numerical instabilities resulting in some areas having a checkerboard pattern
    # round pixel values to ints
    # set the boundaries of the buckets to not be ints
    # should happen automatically through the log scale
    # -0.1 should be 0.0 actually
    # boundary of last bucket does not matter - values outside of the range are colored in the last bucket
    boundaries = (
        [-1, -0.1]
        + [log_scale(i) for i in range(1, num_scale_colors + 1)]
        + [max_wait + 1]
    )

    # prepare the plot
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    fig, ax = plt.subplots(figsize=(100, 100))

    # get borders of all countries
    # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip
    # from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
    print("Loading country shapes...")
    countries = gpd.datasets.get_path("naturalearth_lowres")
    countries = gpd.read_file(countries)
    countries = countries.to_crs(epsg=3857)
    countries = countries[countries.name != "Antarctica"]
    # TODO so far does not work as in final map the raster is not applied to the whole region
    # countries = countries[countries.geometry.within(polygon.geometry[0])]
    country_shapes = countries.geometry
    countries.plot(ax=ax, facecolor="none", edgecolor="black")

    # TODO takes more time than expected
    # use a pre-compiled list of important cities
    # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_populated_places.zip
    # from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
    # cities = gpd.read_file("cities/ne_10m_populated_places.shp", bbox=polygon.geometry[0]) should work but does not
    if show_cities:
        print("Loading cities...")
        cities = gpd.read_file(
            "map_features/cities/ne_10m_populated_places.shp"
        )  # takes most time
        cities = cities.to_crs(epsg=3857)
        cities = cities[cities.geometry.within(polygon.geometry[0])]
        cities.plot(ax=ax, markersize=1, color="black")

    # use a pre-compiles list of important roads
    # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_roads.zip
    # from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
    if show_roads:
        print("Loading roads...")
        roads = gpd.read_file("map_features/roads/ne_10m_roads.shp")
        roads = roads.to_crs(epsg=3857)
        roads = roads[roads.geometry.within(polygon.geometry[0])]
        roads.plot(ax=ax, markersize=1, color="black")

    if show_spots:
        all_points.plot(ax=ax, markersize=10, color="red")

    # limit heatmap to landmass by asigning inf/ high value to sea
    print("Transforming heatmap...")
    with rasterio.open(map_path) as heatmap:
        out_image, out_transform = rasterio.mask.mask(
            heatmap, country_shapes, crop=True, filled=False
        )
        out_meta = heatmap.meta

    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )

    with rasterio.open(map_path, "w", **out_meta) as destination:
        destination.write(out_image)

    # plot the heatmap
    print("Plotting heatmap...")
    raster = rasterio.open(map_path)
    rasterio.plot.show(raster, ax=ax, cmap=cmap, norm=norm)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    if method == "ITERATIVE":
        file_name = f"maps/map_{region}_iter_{ITERATIONS}.png"
    elif method == "DYNAMIC":
        file_name = f"maps/map_{region}_{K}.png"
    elif method == "GP":
        file_name = f"maps/map_gp_{region}.png"
    else:
        file_name = f"maps/map_{region}.png"
    plt.savefig(file_name, bbox_inches="tight")
