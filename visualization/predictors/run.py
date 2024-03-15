# notebook

# print full np arrays
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

# import libraries

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

plt.rcParams.update({'font.size': 20})
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# set vars here! to define which map is created in which way

# plot the heatmap
fig, ax = plt.subplots(figsize=(100, 100))

regions = ["africa", "asia", "australia", "europe", "north_america", "south_america"]

here = "/home/till/projects"

ROADS = False
CITIES = False
SPOTS = False
CIRCLE = 50000 # circle around spot in m for which we assume the waiting time at the spot is definitely relevant
DAY = 24 * 60
WAIT_MAX = DAY
# resolution 40 for world -> 6800x14400 image -> lets calculating distribution crash -> too many values in np array to calculate?
RESOLUTION = 10 # pixel per degree
THRESHOLD = 0.0 # sum of distributions for at pixel at which we assume we have no data there; quite arbitrary
RECOMPUTE = True # otherwise load from cache if possible

# color the whole map smoothly & fine grained where possible

# approach 1
# adjust stdv of single points
DYNAMIC = False
K = 1
FACTOR_STDV = 1 # the xth stdv

# approach 2
# color not defined areas iteratively by using larger stdvs
ITERATIVE = False
ITERATIONS = 10
for region in regions:
    map_path = f"{here}/heatchmap/map/map_{region}.tif"

    def get_points(path):
        points = gpd.read_file(path)
        points.wait = points.wait.astype(float)
        points.lat = points.lat.astype(float)
        points.lon = points.lon.astype(float)
        # threshold - assuming that values above that are skewed due to angriness of the hiker
        points = points[points['wait'] <= WAIT_MAX]

        # use epsg 3857 as default as it gives coordinates in meters
        points.geometry = gpd.points_from_xy(points.lon, points.lat)
        points.crs = CRS.from_epsg(4326)
        points = points.to_crs(epsg=3857)

        return points
        
    # read from the hitchmap dump
    points = get_points(f"{here}/heatchmap/waiting_time_per_point.csv")
    all_points = points

    # to get the same example as here https://abelblogja.wordpress.com/average-waiting-times-in-europe/
    artificial_points = get_points(f"{here}/heatchmap/artificial_points.csv")
    def get_points_in_region(points, region):
            # set lat long boundaries of different scopes of the map

            maps = {
                    "germany": [5.0, 48.0, 15.0, 55.0],
                    "europe": [-12.0, 35.0, 45.0, 71.0],
                    "world": [-180.0, -85.0, 180.0, 85.0], # 85 lat bc of 3857
                    "small": [12.0, 52.0, 15.0, 54.0],
                    "africa": [-20.0, -35.0, 60.0, 40.0],
                    "asia": [40.0, 0.0, 180.0, 85.0],
                    "north_america": [-180.0, 0.0, -20.0, 85.0],
                    "south_america": [-90.0, -60.0, -30.0, 15.0],
                    "australia": [100.0, -50.0, 180.0, 0.0],
                    "middle_africa": [-10.0, -35.0, 60.0, 20.0],
                    "artificial": [8.0, -10.0, 30.0, 10.0],
                    "greenland": [-80.0, 60.0, -10.0, 85.0],
                    }
            map = maps[region]

            # create boundary polygon
            polygon = Polygon([(map[0], map[1]), (map[0], map[3]), (map[2], map[3]), (map[2], map[1]), (map[0], map[1])])
            polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon])  
            polygon = polygon.to_crs(epsg=3857)

            # extract points within polygon
            points = points[points.geometry.within(polygon.geometry[0])]

            return points, polygon, map

    points, polygon, map = get_points_in_region(points, region)
    if region == "artificial": 
            points = artificial_points
    # choose larger radius for not dense points (set radius till x other spots are in the radius)
    # scaling: https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
    def calc_radius(point, other_points, k=5):
        k = k + 1 # +1 as the point itself is also in the array
        distances = point.distance(other_points).to_numpy()
        idx = np.argpartition(distances, k)
        closest_distances = distances[idx[:k]]
        radius = np.ceil(max(closest_distances)) / FACTOR_STDV
        return radius
    def define_raster(polygon, map):
        xx, yy = polygon.geometry[0].exterior.coords.xy

        # Note above return values are of type `array.array` 
        xx = xx.tolist()
        yy = yy.tolist()

        degree_width = int(map[2] - map[0])
        degree_height = int(map[3] - map[1])
        pixel_width = degree_width * RESOLUTION
        pixel_height = degree_height * RESOLUTION

        return xx, yy, pixel_width, pixel_height

    def save_raster(Z, polygon, map):

        polygon_vertices_x, polygon_vertices_y, pixel_width, pixel_height = define_raster(polygon, map)
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


    def make_raster_map(points, polygon, map, iteration=0):
        
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
            if DYNAMIC: STDV_M = max(CIRCLE, calc_radius(Point(lon, lat), points.geometry, k=K))
            else : STDV_M = CIRCLE + 10000 * iteration
            return makeGaussian(STDV_M, lon, lat)

        # create pixel grid for map

        xx, yy, pixel_width, pixel_height = define_raster(polygon, map)
        x = np.linspace(xx[0], xx[2], pixel_width)
        # mind starting with upper value of y axis here
        y = np.linspace(yy[2], yy[0], pixel_height)
        X, Y = np.meshgrid(x, y)
        # higher precision prevents pixels far away from the points to be 0/ nan
        X = np.longdouble(X)
        Y = np.longdouble(Y)

        # sum of distributions
        Zn = None
        # weighted sum of distributions
        Zn_weighted = None

        try:
            if RECOMPUTE:
                raise Exception("recompute")
            else:
                Z = np.loadtxt(f'{here}/heatchmap/map/map_{region}.txt', dtype=float)
                
        except:
            # create a raster map - resulution is defined above
            # https://stackoverflow.com/questions/56677267/tqdm-extract-time-passed-time-remaining
            with tqdm(zip(points.geometry.y, points.geometry.x, points.wait), total=len(points)) as t:
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

                elapsed = t.format_dict['elapsed']
                elapsed_str = t.format_interval(elapsed)
                df = pd.DataFrame({"region": region, 'elapsed time': [elapsed_str]})

                tracker_name = f'{here}/heatchmap/map/time_tracker.csv'
                try:
                    full_df = pd.read_csv(tracker_name, index_col=0)
                    full_df = pd.concat([full_df, df])
                    full_df.to_csv(tracker_name, sep=',')
                except:
                    df.to_csv(tracker_name)

            # normalize the weighted sum by the sum of all distributions -> so we see the actual waiting times in the raster
            Z = np.divide(Zn_weighted, Zn, out=np.zeros_like(Zn_weighted), where=Zn!=0)

            # grey out pixels with no hitchhiking spots near them
            undefined = -1.0
            Z = np.where(Zn < THRESHOLD, undefined, Z)
        
            # save the underlying raster data of the heatmap for later use
            np.savetxt(f'{here}/heatchmap/map/map_{region}.txt', Z) 

        return X, Y, Z, Zn, Zn_weighted

    # set radius of all points +x km and recompute to only color remaining "not-enough-data" pixels
    # -> repeat untill full map is colored
    if ITERATIVE:
        for i in range(0, ITERATIONS):
            X, Y, Z_raw, Zn, Zn_w = make_raster_map(points, polygon, map, i)

            if i == 0:
                Z = Z_raw
            else:
                mask = np.logical_and(Z_raw != -1.0, Z == -1.0)
                if np.any(mask):
                    Z = np.where(mask, Z_raw, Z)
                else:
                    break

        save_raster(Z, polygon, map)
    else:
        X, Y, Z_raw, Zn, Zn_w = make_raster_map(points, polygon, map)
        Z = Z_raw
        save_raster(Z, polygon, map)
        
    # get borders of all countries
    # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip
    # from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
    countries = gpd.datasets.get_path("naturalearth_lowres")
    countries = gpd.read_file(countries)
    countries = countries.to_crs(epsg=3857)
    countries = countries[countries.name != "Antarctica"]
    # TODO so far does not work as in final map the raster is not applied to the whole region
    # countries = countries[countries.geometry.within(polygon.geometry[0])]
    country_shapes = countries.geometry
    # limit heatmap to landmass by asigning inf/ high value to sea
    with rasterio.open(map_path) as heatmap:
        out_image, out_transform = rasterio.mask.mask(heatmap, country_shapes, crop=True, filled=False)
        out_meta = heatmap.meta
        
    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    with rasterio.open(map_path, "w", **out_meta) as destination:
        destination.write(out_image)

    # TODO takes more time than expected

    # use a pre-compiled list of important cities
    # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_populated_places.zip
    # from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
    # cities = gpd.read_file("cities/ne_10m_populated_places.shp", bbox=polygon.geometry[0]) should work but does not
    if CITIES:
        cities = gpd.read_file("cities/ne_10m_populated_places.shp") # takes most time
        cities = cities.to_crs(epsg=3857)
        cities = cities[cities.geometry.within(polygon.geometry[0])]

    # use a pre-compiles list of important roads
    # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_roads.zip
    # from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
    if ROADS:
        roads = gpd.read_file("roads/ne_10m_roads.shp")
        roads = roads.to_crs(epsg=3857)
        roads = roads[roads.geometry.within(polygon.geometry[0])]
    # define the heatmap color scale

    # TODO smoother spectrum instead of buckets
    buckets = [
        'grey', # not enough data
        '#008200', # dark green
        '#00c800', # light green
        'green', # green
        '#c8ff00', # light yellow
        '#ffff00', # yellow
        '#ffc800', # light orange
        '#ff8200', # dark orange
        'red', # red
        '#c80000', # dark red
        '#820000', # wine red
        'blue', # not necessary to color
        ]

    cmap = colors.ListedColormap(buckets)

    max_wait = all_points.wait.max() + 0.1 # to get at least this value as maximum for the colored buckets
    num_scale_colors = len(buckets) - 2 # because of upper and lower bucket
    # build log scale starting at 0 and ending at max wait
    base = (max_wait + 1) ** (1/num_scale_colors)

    def log_scale(x):
        return base ** x - 1

    # how to prevent numerical instabilities resulting in some areas having a checkerboard pattern
    # round pixel values to ints
    # set the boundaries of the buckets to not be ints
        # should happen automatically through the log scale
        # -0.1 should be 0.0 actually
    # boundary of last bucket does not matter - values outside of the range are colored in the last bucket
    boundaries = [-1, -0.1] + [log_scale(i) for i in range(1, num_scale_colors+1)] + [max_wait + 1]

    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    # plot the heatmap
    raster = rasterio.open(map_path)
    rasterio.plot.show(raster, ax=ax, cmap=cmap, norm=norm)

countries.plot(ax=ax, facecolor='none', edgecolor='black')
if CITIES: cities.plot(ax=ax, markersize=1, color='black')
if ROADS: roads.plot(ax=ax, markersize=1, color='black')

if SPOTS: all_points.plot(ax=ax, markersize=10, color='red')

fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
if len(regions) > 1: file_name = f"{here}/heatchmap/map/map_all.png"
elif ITERATIVE: file_name = f"{here}/heatchmap/map/map_{region}_iter_{ITERATIONS}.png"
elif DYNAMIC: file_name = f"{here}/heatchmap/map/map_{region}_{K}.png"
else: file_name = f"{here}/heatchmap/map/map_{region}.png"
plt.savefig(file_name, bbox_inches='tight')

print("Done!")