import rasterio
import rasterio.plot
from rasterio.crs import CRS
from matplotlib import pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio.mask
from shapely.geometry import Point
import matplotlib.colors as colors
from matplotlib import cm
from shapely.geometry import Polygon
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP

from sklearn.base import BaseEstimator, RegressorMixin
from tqdm.auto import tqdm

tqdm.pandas()

RESOLUTION = 2

# 180 degree meridian in epsg 3857
MERIDIAN = 20037508


class MapBasedModel(BaseEstimator, RegressorMixin):
    def __init__(self, method, region="world", resolution=RESOLUTION, verbose=False):
        self.method = method
        self.region = region
        self.resolution = resolution  # pixel per degree
        self.verbose = verbose

    def get_map_boundary(self):
        # [lon_bottom_left, lat_bottom_left, lon_top_right, lat_top_right]
        maps = {
            "germany": [3.0, 48.0, 16.0, 55.0],
            "spain": [-8.0, 36.0, 3.0, 43.0],
            "spain_france": [-8.0, 36.0, 6.0, 50.0],
            "turkey": [26.0, 36.0, 45.0, 42.0],
            "europe": [-25.0, 34.0, 46.0, 72.0],
            "world": [-180.0, -56.0, 180.0, 80.0],
            "africa": [-19.0, -35.0, 52.0, 38.0],
            "nigeria": [2.0, 4.0, 15.0, 14.0],
            "asia": [34.0, 5.0, 180.0, 78.0], # TODO right lon value should be -169.0
            "north_america": [-170.0, 15.0, -50.0, 75.0],
            "south_america": [-83.0, -56.0, -33.0, 15.0],
            "australia": [95.0, -48.0, 180.0, 8.0],
            "artificial": [8.0, -1.0, 30.0, 1.0],
        }

        return maps[self.region]

    def map_to_polygon(self):
        # create boundary polygon
        polygon = Polygon(
            [
                (self.map_boundary[0], self.map_boundary[1]),
                (self.map_boundary[0], self.map_boundary[3]),
                (self.map_boundary[2], self.map_boundary[3]),
                (self.map_boundary[2], self.map_boundary[1]),
                (self.map_boundary[0], self.map_boundary[1]),
            ]
        )
        polygon = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon])
        polygon = polygon.to_crs(epsg=3857)  # transform to metric epsg
        polygon = polygon.geometry[0]

        return polygon

    def save_as_raster(self):
        map_path = self.get_raster_path()

        polygon_vertices_x, polygon_vertices_y, pixel_width, pixel_height = (
            self.define_raster()
        )

        # handling special case when map spans over the 180 degree meridian
        if (polygon_vertices_x[0] > 0 and polygon_vertices_x[2] < 0):
            polygon_vertices_x[2] = 2 * MERIDIAN + polygon_vertices_x[2]
            polygon_vertices_x[3] = 2 * MERIDIAN + polygon_vertices_x[3]


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

        # cannot use np.float128 to write to tif
        self.raw_raster = self.raw_raster.astype(np.float64)

        # save the colored raster using the above transform
        # important: rasterio requires [0,0] of the raster to be in the upper left corner and [rows, cols] in the lower right corner
        # TODO find out why raster is getting smaller in x direction when stored as tif (e.g. 393x700 -> 425x700)
        with rasterio.open(
            map_path,
            "w",
            driver="GTiff",
            height=self.raw_raster.shape[0],
            width=self.raw_raster.shape[1],
            count=1,
            crs=CRS.from_epsg(3857),
            transform=transform,
            dtype=self.raw_raster.dtype,
        ) as destination:
            destination.write(self.raw_raster, 1)

        self.raster = rasterio.open(map_path)

        return map

    # create pixel grid for map
    def get_map_grid(self):
        self.map_boundary = self.get_map_boundary()

        xx, yy, pixel_width, pixel_height = self.define_raster()
        # handling special case when map spans over the 180 degree meridian
        if not (xx[0] > 0 and xx[2] < 0):
            x = np.linspace(xx[0], xx[2], pixel_width)
        else:
            ratio = (MERIDIAN - xx[0]) / ((MERIDIAN - xx[0]) + (xx[2] + MERIDIAN))
            x = np.linspace(xx[0], MERIDIAN, int(pixel_width * ratio))
            x = np.append(x, np.linspace(-MERIDIAN+1, xx[2], int(pixel_width * (1 - ratio))))

        # mind starting with upper value of y axis here
        y = np.linspace(yy[2], yy[0], pixel_height)
        self.X, self.Y = np.meshgrid(x, y)
        # higher precision prevents pixels with high uncertainties (no data) in the WAG model to become 0/ nan
        self.X = np.longdouble(self.X)
        self.Y = np.longdouble(self.Y)

    def define_raster(self):
        xx, yy = self.map_to_polygon().exterior.coords.xy

        # Note above return values are of type `array.array`
        xx = xx.tolist()
        yy = yy.tolist()

        degree_height = int(self.map_boundary[3] - self.map_boundary[1])
        # handling special case when map spans over the 180 degree meridian
        degree_width = (
            int(self.map_boundary[2] - self.map_boundary[0])
            if not (self.map_boundary[2] < 0 and self.map_boundary[0] > 0)
            else int(360 - self.map_boundary[0] + self.map_boundary[2])
        )
        pixel_width = degree_width * self.resolution
        pixel_height = degree_height * self.resolution

        return xx, yy, pixel_width, pixel_height

    def get_raster_path(self):
        return f"intermediate/map_{self.method}_{self.region}_{self.resolution}.tif"

    def build_map(
        self,
        points=None,
        all_points=None,
        show_states=True,
        show_cities=False,
        show_roads=False,
        show_points=False,
        show_axis=False,
        show_uncertainties=False,
        discrete_uncertainties=False,
        final=False,
        figsize=10,
    ):
        # setup
        if not hasattr(self, "raw_uncertainties"):
            uncertainties = 1.0
        else:
            uncertainties = self.raw_uncertainties
        self.map_boundary = self.get_map_boundary()
        map_path = self.get_raster_path()

        fig, ax = plt.subplots(figsize=(figsize, figsize))

        # get borders of all countries
        # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip
        # from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/

        if self.verbose:
            print("Loading country shapes...")
        countries = gpd.read_file(
            "map_features/countries/ne_110m_admin_0_countries.shp"
        )
        countries = countries.to_crs(epsg=3857)
        countries = countries[countries.NAME != "Antarctica"]
        # TODO so far does not work as in final map the raster is not applied to the whole region
        # countries = countries[countries.geometry.within(polygon.geometry[0])]
        country_shapes = countries.geometry
        if show_states:
            countries.plot(ax=ax, linewidth=0.5 * (0.1 * figsize), facecolor="none", edgecolor="black")

        # use a pre-compiles list of important roads
        # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_roads.zip
        # from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
        if show_roads:
            if self.verbose:
                print("Loading roads...")
            roads = gpd.read_file("map_features/roads/ne_10m_roads.shp")
            roads = roads.to_crs(epsg=3857)
            roads = roads[roads.geometry.within(self.map_to_polygon())]
            roads.plot(ax=ax, markersize=0.2 * (0.1 * figsize), linewidth=0.5 * (0.1 * figsize), color="gray", zorder=2)

        # takes a lot of time
        # use a pre-compiled list of important cities
        # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_populated_places.zip
        # from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
        if show_cities:
            if self.verbose:
                print("Loading cities...")
            cities = gpd.read_file("map_features/cities/ne_10m_populated_places.shp")
            cities = cities.to_crs(epsg=3857)
            cities = cities[cities.geometry.within(self.map_to_polygon())]
            cities.plot(ax=ax, markersize=1.0 * figsize, color="navy", marker="o", zorder=10)

        if show_points:
            all_points.plot(ax=ax, markersize=10, color="red")

        # limit heatmap to landmass by asigning no data value to sea
        if self.verbose:
            print("Transforming heatmap...")
        nodata = np.nan
        with rasterio.open(map_path) as heatmap:
            max_map_wait = heatmap.read().max()
            min_map_wait = heatmap.read().min()
            if self.verbose:
                print("max map waiting time:", max_map_wait)
            if self.verbose:
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

        new_map_path = f"intermediate/map_{self.method}_{self.region}_{self.resolution}_processed.tif"
        with rasterio.open(new_map_path, "w", **out_meta) as destination:
            destination.write(out_image)

        # plot the heatmap
        if self.verbose:
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
            "#330101",  # drop?
        ]

        cmap = colors.ListedColormap(buckets)

        # max_wait = max(
        #     all_points.wait.max() + 0.1, 100
        # )  # to get at least this value as maximum for the colored buckets
        max_wait = 675
        if self.verbose:
            print("max waiting time:", max_wait)

        # define the heatmap color scale
        # values higher than the upper boundary are colored in the upmost color
        boundaries = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # prepare the plot
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

        background_color = "0.7"
        ax.set_facecolor(
            background_color
        )  # background color light gray for landmass with uncertainties

        if show_uncertainties:
            # let certainty have no influence on sea color
            uncertainties = np.where(
                np.isnan(raster.read()[0]), uncertainties.min(), uncertainties
            )
            if (uncertainties.max() - uncertainties.min()) != 0:
                uncertainties = (uncertainties - uncertainties.min()) / (
                    uncertainties.max() - uncertainties.min()
                )
                uncertainties = 1 - uncertainties
                if discrete_uncertainties:
                    # threshold for uncertainty decided by experiment
                    uncertainties = np.where(uncertainties < 0.25, 0.0, 1.0)
            else:
                uncertainties = 1.0
            # let certainty have no influence on sea color
            uncertainties = np.where(np.isnan(raster.read()[0]), 1, uncertainties)
            uncertainties = uncertainties.astype(
                np.float64
            )  # matplotlib cannot handle float128
            self.uncertainties = uncertainties
        else:
            uncertainties = 1.0

        # set color for nan (nodata... sea) values
        # from https://stackoverflow.com/questions/2578752/how-can-i-plot-nan-values-as-a-special-color-with-imshow
        cmap.set_bad(color="blue")
        # TODO: not able to plot across 180 meridian
        rasterio.plot.show(raster, ax=ax, cmap=cmap, norm=norm, alpha=uncertainties)
        if show_axis:
            ax.set_xlabel("Longitude", fontsize=figsize)
            ax.set_ylabel("Latitude", fontsize=figsize)
            ax.tick_params(axis="both", which="major", labelsize=figsize)
        else:
            # do not show axis labels
            ax.set_xticks([])
            ax.set_yticks([])

        if show_uncertainties and not discrete_uncertainties:
            norm_uncertainties = plt.Normalize(0, 1)
            cmap_uncertainties = colors.LinearSegmentedColormap.from_list(
                "", ["#00c800", background_color]
            )

            cbar_uncertainty = fig.colorbar(
                cm.ScalarMappable(norm=norm_uncertainties, cmap=cmap_uncertainties),
                ax=ax,
                shrink=0.3,
                pad=0.0,
            )
            cbar_uncertainty.ax.tick_params(labelsize=figsize)
            cbar_uncertainty.ax.set_ylabel(
                "Uncertainty in waiting time estimation", rotation=90, fontsize=figsize
            )

        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])

        cbar = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax
        )
        boundary_labels = [
            "0 min",
            "10",
            "20",
            "30",
            "40",
            "50",
            "60",
            "70",
            "80",
            "90",
            "> 100",
        ]
        cbar.ax.set_yticks(ticks=boundaries, labels=boundary_labels)
        cbar.ax.tick_params(labelsize=figsize)

        if self.method == "ITERATIVE":
            file_name = f"maps/map_{region}_iter_{ITERATIONS}.png"
        elif self.method == "DYNAMIC":
            file_name = f"maps/map_{region}_{K}.png"
        else:
            if final:
                file_name = f"final_maps/{self.region}.png"
            else:
                file_name = f"maps/{self.method}_{self.region}_{self.resolution}.png"
        plt.savefig(file_name, bbox_inches="tight")
        plt.show()
        


class Average(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.mean = np.mean(y)

        return self

    def predict(self, X):
        return np.ones(X.shape[0]) * self.mean


class Tiles(MapBasedModel):
    def __init__(self, region="world", tile_size=300000):
        self.region = region
        self.tile_size = tile_size  # in meters

    def get_tile_intervals(self, min, max):
        intervals = [min]
        while (max - min) > self.tile_size:
            new_interval_bound = min + self.tile_size
            intervals.append(new_interval_bound)
            min = new_interval_bound

        intervals.append(max)

        return intervals

    def create_tiles(self):
        xx, yy = self.map_polygon.exterior.coords.xy
        lon_min = xx[0]
        lon_max = xx[3]
        lat_min = yy[0]
        lat_max = yy[1]

        self.lon_intervals = self.get_tile_intervals(lon_min, lon_max)
        self.lat_intervals = self.get_tile_intervals(lat_min, lat_max)

        tiles = np.zeros((len(self.lon_intervals) - 1, len(self.lat_intervals) - 1))

        return tiles

    def get_interval_num(self, intervals, value):
        for i in range(len(intervals) - 1):
            if value >= intervals[i] and value <= intervals[i + 1]:
                return i

    def fit(self, X, y):
        self.map_boundary = self.get_map_boundary()
        self.map_polygon = self.map_to_polygon()
        self.tiles = self.create_tiles()

        points_per_tile = np.zeros(self.tiles.shape)

        for x, single_y in zip(X, y):
            lon, lat = x
            lon_num = self.get_interval_num(self.lon_intervals, lon)
            lat_num = self.get_interval_num(self.lat_intervals, lat)

            self.tiles[lon_num][lat_num] += single_y
            points_per_tile[lon_num][lat_num] += 1

        # average
        points_per_tile = np.where(points_per_tile == 0, 1, points_per_tile)
        self.tiles = self.tiles / points_per_tile

        return self

    def predict(self, X):
        predictions = []

        for x in X:
            lon, lat = x
            lon_num = self.get_interval_num(self.lon_intervals, lon)
            lat_num = self.get_interval_num(self.lat_intervals, lat)
            predictions.append(self.tiles[lon_num][lat_num])

        return np.array(predictions)


class WeightedAveragedGaussian(MapBasedModel):
    def __init__(
        self, region="world", method="ordinary", resolution=RESOLUTION, verbose=False
    ):
        self.region = region
        self.method = method
        self.resolution = resolution  # pixel per degree
        self.verbose = verbose

    # https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
    def makeGaussian(self, stdv, x0, y0):
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
        return np.exp(
            -4 * np.log(2) * ((self.X - x0) ** 2 + (self.Y - y0) ** 2) / fwhm**2
        )

    # choose larger radius for not dense points (set radius till k other spots are in the radius)
    # scaling: https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
    def calc_radius(self, point, other_points, k=5):
        k = k + 1  # +1 as the point itself is also in the array
        distances = point.distance(other_points).to_numpy()
        idx = np.argpartition(distances, k)
        closest_distances = distances[idx[:k]]
        radius = np.ceil(max(closest_distances)) / FACTOR_STDV
        return radius

    def get_distribution(self, lon, lat):
        # standard deviation in meters -> 50 km around each spot; quite arbitrary choice
        if self.method == "DYNAMIC":
            stdv_m = max(
                self.circle_size,
                calc_radius(Point(lon, lat), self.points.geometry, k=K),
            )
        else:
            stdv_m = self.circle_size + 10000 * self.iteration
        return self.makeGaussian(stdv_m, lon, lat)

    # make_raster_map
    def fit(
        self,
        X,
        y,
        iteration=0,
        recompute=True,
        no_data_threshold=0.00000001,
    ):
        self.raster = None
        self.raw_raster = None

        self.points = X
        self.circle_size = 50000
        self.iteration = iteration
        self.get_map_grid()

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
            if self.verbose:
                print("Weighting gaussians for all points...")
            with tqdm(
                zip(X[:, 0], X[:, 1], y), total=X.shape[0], disable=not self.verbose
            ) as t:
                # TODO find out how to speed up and parallelize this
                for lon, lat, wait in t:
                    # distribution inserted by a single point
                    Zi = self.get_distribution(lon, lat)
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
                df = pd.DataFrame(
                    {"region": self.region, "elapsed time": [elapsed_str]}
                )

                tracker_name = "logging/time_tracker.csv"
                try:
                    full_df = pd.read_csv(tracker_name, index_col=0)
                    full_df = pd.concat([full_df, df])
                    full_df.to_csv(tracker_name, sep=",")
                except:
                    df.to_csv(tracker_name)

            # normalize the weighted sum by the sum of all distributions -> so we see the actual waiting times in the raster
            Z = np.divide(
                Zn_weighted, Zn, out=np.zeros_like(Zn_weighted), where=Zn != 0
            )

            # grey out pixels with no hitchhiking spots near them
            undefined = -1.0
            Z = np.where(Zn < no_data_threshold, undefined, Z)

            # save the underlying raster data of the heatmap for later use
            np.savetxt(f"intermediate/map_{self.region}.txt", Z)

        self.raw_raster = Z
        self.save_as_raster()

        return self

    def predict(self, X):
        # from https://rasterio.readthedocs.io/en/stable/quickstart.html#spatial-indexing
        # in epsg 3857

        predictions = []

        for lon, lat in X:
            # transform the lat/lon to the raster's coordinate system
            x, y = self.raster.index(lon, lat)
            # read the raster at the given coordinates
            predictions.append(self.raster.read(1)[x, y])

        return np.array(predictions)
