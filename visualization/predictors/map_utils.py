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


def define_raster(polygon, map, resolution=RESOLUTION):
    xx, yy = polygon.geometry[0].exterior.coords.xy

    # Note above return values are of type `array.array`
    xx = xx.tolist()
    yy = yy.tolist()

    degree_width = int(map[2] - map[0])
    degree_height = int(map[3] - map[1])
    pixel_width = degree_width * resolution
    pixel_height = degree_height * resolution

    return xx, yy, pixel_width, pixel_height


def save_numpy_map(map, region="world", method="ordinary", kind_of_map='map', resolution=RESOLUTION):
    map_path = f"intermediate/{kind_of_map}_{method}_{region}_{resolution}.txt"
    np.savetxt(map_path, map)

def load_numpy_map(region="world", method="ordinary", kind_of_map='map', resolution=RESOLUTION):
    map_path = f"intermediate/{kind_of_map}_{method}_{region}_{resolution}.txt"
    return np.loadtxt(map_path)

def save_as_raster(map, polygon, map_boundary, region="world", method="ordinary", resolution=RESOLUTION):
    map_path = f"intermediate/map_{method}_{region}_{resolution}.tif"

    polygon_vertices_x, polygon_vertices_y, pixel_width, pixel_height = define_raster(
        polygon, map_boundary, resolution
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
    map = np.double(map)
    # TODO: need this?
    # map = np.round(map, 0)

    # save the colored raster using the above transform
    # TODO find out why raster is getting smaller in x direction when stored as tif (e.g. 393x700 -> 425x700)
    with rasterio.open(
        map_path,
        "w",
        driver="GTiff",
        height=map.shape[0],
        width=map.shape[1],
        count=1,
        crs=CRS.from_epsg(3857),
        transform=transform,
        dtype=map.dtype,
    ) as destination:
        destination.write(map, 1)

    return map


def load_raster(region="world", method="ordinary", resolution=RESOLUTION):
    map_path = f"intermediate/map_{method}_{region}_{resolution}.tif"
    return rasterio.open(map_path)


def get_map_grid(polygon, map_boundary, resolution=RESOLUTION):
    xx, yy, pixel_width, pixel_height = define_raster(polygon, map_boundary, resolution)
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


def make_raster_map(
    points,
    region,
    polygon,
    map,
    iteration=0,
    method="ordinary",
    recompute=True,
    circle_size=50000,
    no_data_threshold=0.00000001,
    resolution=RESOLUTION,
):

    # https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
    def makeGaussian(stdv, x0, y0):
        """Make a square gaussian kernel.
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        # TODO why fwhm used here?
        fwhm = 2.355 * stdv

        # gives the distribution in the whole raster space as X and Y are used here
        # TODO only calculate for pixels that will be colored (landmass) in the end
        # TODO only calculate for pixels that are significantly close to the point (e.g. 500 km around the point)
        return np.exp(-4 * np.log(2) * ((X - x0) ** 2 + (Y - y0) ** 2) / fwhm**2)

    def get_distribution(lat, lon):
        # standard deviation in meters -> 50 km around each spot; quite arbitrary
        if method == "DYNAMIC":
            STDV_M = max(
                circle_size, calc_radius(Point(lon, lat), points.geometry, k=K)
            )
        else:
            STDV_M = circle_size + 10000 * iteration
        return makeGaussian(STDV_M, lon, lat)

    # create pixel grid for map
    X, Y = get_map_grid(polygon, map, resolution=resolution)

    # sum of distributions
    Zn = None
    # weighted sum of distributions
    Zn_weighted = None

    try:
        if recompute:
            raise Exception("recompute")
        else:
            Z = np.loadtxt(f"intermediate/map_{region}.txt", dtype=float)

    except:
        # create a raster map - resulution is defined above
        # https://stackoverflow.com/questions/56677267/tqdm-extract-time-passed-time-remaining
        print("Weighting gaussians for all points...")
        with tqdm(
            zip(points.geometry.y, points.geometry.x, points.wait), total=len(points)
        ) as t:
            # TODO find out how to speed up and parallelize this
            for lat, lon, wait in t:
                # distribution inserted by a single point
                Zi = get_distribution(lat, lon)
                # add the new distribution to the sum of existing distributions
                # write them to Zn_weighted and wait every single point/ distribution by the waiting time
                # => it matters where a distribiton is inserted (areas with more distributions have a higher certainty)
                # and which waiting time weight is associated with it
                if Zn is None:
                    Zn = Zi
                    Zn_weighted = Zi * wait
                else:
                    Zn = np.sum([Zn, Zi], axis=0)
                    Zn_weighted = np.sum([Zn_weighted, Zi * wait], axis=0)

            elapsed = t.format_dict["elapsed"]
            elapsed_str = t.format_interval(elapsed)
            df = pd.DataFrame({"region": region, "elapsed time": [elapsed_str]})

            tracker_name = "logging/time_tracker.csv"
            try:
                full_df = pd.read_csv(tracker_name, index_col=0)
                full_df = pd.concat([full_df, df])
                full_df.to_csv(tracker_name, sep=",")
            except:
                df.to_csv(tracker_name)

        # normalize the weighted sum by the sum of all distributions -> so we see the actual waiting times in the raster
        Z = np.divide(Zn_weighted, Zn, out=np.zeros_like(Zn_weighted), where=Zn != 0)

        # grey out pixels with no hitchhiking spots near them
        undefined = -1.0
        Z = np.where(Zn < no_data_threshold, undefined, Z)

        # save the underlying raster data of the heatmap for later use
        np.savetxt(f"intermediate/map_{region}.txt", Z)

    return X, Y, Z, Zn, Zn_weighted


# from https://rasterio.readthedocs.io/en/stable/quickstart.html#spatial-indexing
# in epsg 3857
def map_predict(lon, lat, raster):
    # transform the lat/lon to the raster's coordinate system
    x, y = raster.index(lon, lat)
    # read the raster at the given coordinates
    return raster.read(1)[x, y]


def make_map_from_gp(gp, average, region, polygon, map_boundary, resolution=10):
    X, Y = get_map_grid(polygon, map_boundary, resolution)
    grid = np.array((Y, X)).T
    map = np.empty((0, X.shape[0]))
    certainty_map = np.empty((0, X.shape[0]))

    for vertical_line in tqdm(grid):
        pred, stdv = gp.predict(vertical_line, return_std=True)
        pred = np.exp(pred + average)
        map = np.vstack((map, pred))
        certainty_map = np.vstack((certainty_map, stdv))

    map = map.T
    certainty_map = certainty_map.T

    save_numpy_map(map, region=region, method="gp", resolution=resolution)
    save_numpy_map(certainty_map, region=region, method="gp", kind_of_map='certainty', resolution=resolution)
    map = save_as_raster(map, polygon, map_boundary, region=region, method='gp', resolution=resolution)
    plt.contourf(X, Y, map)
    plt.colorbar()
    plt.savefig(f"maps/contourf_map_gp_{region}_{resolution}.png")

def build_map(
    method="ORDINARY",
    resolution=RESOLUTION,
    points=None,
    all_points=None,
    region="world",
    polygon=None,
    show_states=False,
    show_cities=False,
    show_roads=False,
    show_spots=False,
    certainty=1.0,
):
    map_path = f"intermediate/map_{method}_{region}_{resolution}.tif"
    
    # print("Loading information about states...")
    # states = gpd.read_file("map_features/states/ne_10m_admin_1_states_provinces.shp")
    # states = states.to_crs(epsg=3857)

    # # use smaller units for Russia
    # # country level except for Canada, Russia, USA, Australia, China, Brazil, India, Indonesia
    # states = states[states.admin != "Antarctica"]

    # # a state is hitchhikable if there are hitchhiking spots in it
    # def check_hitchhikability(state):
    #     points_in_state = points[points.geometry.within(state.geometry)]
    #     return len(points_in_state) > 0

    # print("Checking hitchhikability for each state...")
    # states["hh"] = states.progress_apply(check_hitchhikability, axis=1)

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
    if show_states:
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
    nodata = np.nan
    with rasterio.open(map_path) as heatmap:
        max_map_wait = heatmap.read().max()
        min_map_wait = heatmap.read().min()
        print("max map waiting time:", max_map_wait)
        print("min map waiting time:", min_map_wait)

        out_image, out_transform = rasterio.mask.mask(
            heatmap, country_shapes, nodata=nodata
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

    new_map_path = f"intermediate/map_{method}_{region}_{resolution}_processed.tif"
    with rasterio.open(new_map_path, "w", **out_meta) as destination:
        destination.write(out_image)

    # plot the heatmap
    print("Plotting heatmap...")
    raster = rasterio.open(new_map_path)

    # TODO smoother spectrum instead of buckets
    buckets = [
        "#008200",  # dark green
        "#00c800",  # light green
        "#c8ff00",  # light yellow
        "#ffff00",  # yellow
        "#ffc800",  # light orange
        "#ff8200",  # dark orange
        "red",  # red
        "#c80000",  # dark red
        "#820000",  # wine red
        "#820000",  # drop?
    ]

    cmap = colors.ListedColormap(buckets)

    max_wait = (
        all_points.wait.max() + 0.1
    )  # to get at least this value as maximum for the colored buckets
    print("max waiting time:", max_wait)
    num_scale_colors = len(buckets) - 2  # because of upper and lower bucket
    # build log scale starting at 0 and ending at max wait
    base = (max_map_wait - min_map_wait + 1) ** (1 / num_scale_colors)

    def log_scale(x):
        return base**x - 1

    # define the heatmap color scale

    # how to prevent numerical instabilities resulting in some areas having a checkerboard pattern
    # round pixel values to ints
    # set the boundaries of the buckets to not be ints
    # should happen automatically through the log scale
    # -0.1 should be 0.0 actually
    # boundary of last bucket does not matter - values outside of the range are colored in the last bucket
    boundaries = (
        [-1, min_map_wait - 0.1]
        + [min_map_wait + log_scale(i) for i in range(1, num_scale_colors + 1)]
        + [max_wait + 1]
    )

    # values higher than the upper boundary are colored in the upmost color
    boundaries = (
        [min_map_wait, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    )

    print(boundaries)

    # prepare the plot
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    ax.set_facecolor('0.7') # background color light gray for landmass with uncertainty

    try:
        certainty = load_numpy_map(region=region, method=method, kind_of_map='certainty', resolution=resolution)
        certainty = (certainty - certainty.min()) / (certainty.max() - certainty.min())
        certainty = 1 - certainty
    except:
        certainty = 1.0
    # let certainty have no influence on sea color
    certainty = np.where(np.isnan(raster.read()[0]), 1, certainty)
    # shift (for rational quadratic kernel)
    # certainty = certainty + 0.4
    # certainty = np.clip(certainty, 0, 1)

    # set color for nan (nodata... sea) values
    # from https://stackoverflow.com/questions/2578752/how-can-i-plot-nan-values-as-a-special-color-with-imshow
    cmap.set_bad(color='blue')
    rasterio.plot.show(raster, ax=ax, cmap=cmap, norm=norm, alpha=certainty)

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.ax.tick_params(labelsize=50)
    if method == "ITERATIVE":
        file_name = f"maps/map_{region}_iter_{ITERATIONS}.png"
    elif method == "DYNAMIC":
        file_name = f"maps/map_{region}_{K}.png"
    elif method == "gp":
        file_name = f"maps/map_gp_{region}_{resolution}.png"
    else:
        file_name = f"maps/map_{region}.png"
    plt.savefig(file_name, bbox_inches="tight")
