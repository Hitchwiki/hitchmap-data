import geopandas as gpd
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.plot
from matplotlib import cm
from matplotlib import pyplot as plt
from rasterio.control import GroundControlPoint as GCP
from rasterio.crs import CRS
from rasterio.transform import from_gcps
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm.auto import tqdm
import time

tqdm.pandas()

RESOLUTION = 2

# 180 degree meridian in epsg 3857
MERIDIAN = 20037508

BUCKETS = [
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

# define the heatmap color scale
# values higher than the upper boundary are colored in the upmost color
BOUNDARIES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


class MapBasedModel(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        method: str,
        region: str = "world",
        resolution: int = RESOLUTION,
        version: str = "",
        verbose: bool = False,
    ):
        self.method = method
        self.region = region
        self.resolution = resolution  # pixel per degree
        self.version = version
        self.verbose = verbose

        self.map_boundary = self.get_map_boundary()
        self.rasterio_path = f"intermediate/map_{self.method}_{self.region}_{self.resolution}_{self.version}.tif"
        self.map_path = f"intermediate/map_{method}_{region}_{resolution}_{version}.txt"
    

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
            "asia": [34.0, 5.0, 180.0, 78.0],  # TODO right lon value should be -169.0
            "north_america": [-170.0, 15.0, -50.0, 75.0],
            "south_america": [-83.0, -56.0, -33.0, 15.0],
            "australia": [95.0, -48.0, 180.0, 8.0],
            "artificial": [8.0, -1.0, 30.0, 1.0],
        }

        return maps[self.region]

    def get_text_anchor(self):
        """Define where the descriptive text should be placed on the map."""
        anchors = {
            "europe": Point(-10, 70),
            "world": Point(-170, -20),
            "africa": Point(-14, -10),
            "asia": Point(130, 28),
            "north_america": Point(-165, 40),
            "south_america": Point(-60, -45),
            "australia": Point(98, -40),
        }

        # top left corner of text
        anchor = anchors[self.region]
        anchor = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[anchor])
        anchor = anchor.to_crs(epsg=3857)  # transform to metric epsg
        anchor = anchor.geometry[0]

        return anchor

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
        """Saves as .tif raster for rasterio."""

        polygon_vertices_x, polygon_vertices_y, pixel_width, pixel_height = (
            self.define_raster()
        )

        # handling special case when map spans over the 180 degree meridian
        if polygon_vertices_x[0] > 0 and polygon_vertices_x[2] < 0:
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
            self.rasterio_path,
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

        self.raster = rasterio.open(self.rasterio_path)

        return map

    def get_landmass_raster(self):
        countries = gpd.read_file(
            "map_features/countries/ne_110m_admin_0_countries.shp"
        )
        countries = countries.to_crs(epsg=3857)
        countries = countries[countries.NAME != "Antarctica"]
        country_shapes = countries.geometry
        country_shapes = country_shapes.apply(lambda x: make_valid(x))
        self.landmass_raster = np.zeros(map.grid.shape[1:])
        for x, vertical_line in tqdm(enumerate(map.grid.transpose()), total=len(map.grid.transpose())):
            for y, coords in enumerate(vertical_line):
                this_point = Point(float(coords[0]), float(coords[1]))
                self.landmass_raster[y][x] = 1 if any([country_shape.contains(this_point) for country_shape in country_shapes]) else 0

    def get_recalc_raster(self):

        def pixel_from_point(point) -> tuple[int, int]:
            lats = map.Y.transpose()[0]
            lat_index = None
            for i, lat in enumerate(lats):
                if lat >= point["lat"] and point["lat"] >= lats[i+1]:
                    lat_index = i
                    break

            lons = map.X[0]
            lon_index = None
            for i, lon in enumerate(lons):
                if lon <= point["lon"] and point["lon"] <= lons[i+1]:
                    lon_index = i
                    break

            return (lat_index, lon_index)

        recalc_radius = 800000 # TODO: determine from model largest influence radius
        recalc_radius_pixels = int(np.ceil(abs(recalc_radius / (map.grid[0][0][0] - map.grid[0][0][1]))))

        self.recalc_raster = np.zeros(map.grid.shape[1:])
        self.recalc_raster.shape
        for i, point in points.iterrows():
            lat_pixel, lon_pixel = pixel_from_point(point)

            for i in range(lat_pixel - recalc_radius_pixels, lat_pixel + recalc_radius_pixels):
                for j in range(lon_pixel - recalc_radius_pixels, lon_pixel + recalc_radius_pixels):
                    if i < 0 or j < 0 or i >= self.recalc_raster.shape[0] or j >= self.recalc_raster.shape[1]:
                        continue
                    self.recalc_raster[i, j] = 1
        
        print("Report reduction of rasters.")
        print(self.recalc_raster.sum(), self.recalc_raster.shape[0] * self.recalc_raster.shape[1], self.recalc_raster.sum() / (self.recalc_raster.shape[0] * self.recalc_raster.shape[1]))
        self.get_landmass_raster()
        self.recalc_raster = self.recalc_raster * self.landmass_raster
        print(self.landmass_raster.sum(), self.landmass_raster.shape[0] * self.landmass_raster.shape[1], self.landmass_raster.sum() / (self.landmass_raster.shape[0] * self.landmass_raster.shape[1]))
        print(self.recalc_raster.sum(), self.recalc_raster.shape[0] * self.recalc_raster.shape[1], self.recalc_raster.sum() / (self.recalc_raster.shape[0] * self.recalc_raster.shape[1]))


    def get_map_grid(self) -> np.array:
        """Create pixel grid for map."""

        xx, yy, pixel_width, pixel_height = self.define_raster()
        # handling special case when map spans over the 180 degree meridian
        if not (xx[0] > 0 and xx[2] < 0):
            x = np.linspace(xx[0], xx[2], pixel_width)
        else:
            ratio = (MERIDIAN - xx[0]) / ((MERIDIAN - xx[0]) + (xx[2] + MERIDIAN))
            x = np.linspace(xx[0], MERIDIAN, int(pixel_width * ratio))
            x = np.append(
                x, np.linspace(-MERIDIAN + 1, xx[2], int(pixel_width * (1 - ratio)))
            )

        # mind starting with upper value of y axis here
        y = np.linspace(yy[2], yy[0], pixel_height)
        self.X, self.Y = np.meshgrid(x, y)
        # higher precision prevents pixels with high uncertainties (no data) in the WAG model to become 0/ nan
        self.X = np.longdouble(self.X)
        self.Y = np.longdouble(self.Y)

        grid = np.array((self.X, self.Y))
        self.grid = grid
        return grid

    def define_raster(self):
        """Defines the raster in mercator projection not in usual degrees."""
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

    def build_map(
        self,
        points=None,
        show_states: bool = True,
        show_cities: bool = False,
        show_roads: bool = False,
        show_points: bool = False,
        show_axis: bool = False,
        show_uncertainties: bool = False,
        discrete_uncertainties: bool = False,
        final: bool = False,
        figsize: int = 10,
    ):
        """Creates the map from rasterio .tif raster.

        Args:
            points: GeoDataFrame with hitchhiking spots
            show_states: bool, default=True
                Whether to show country borders
            show_cities: bool, default=False
                Whether to show cities
            show_roads: bool, default=False
                Whether to show roads
            show_points: bool, default=False
                Whether to show hitchhiking spots
            show_axis: bool, default=False
                Whether to show axis labels
            show_uncertainties: bool, default=False
                Whether to show uncertainties in waiting time estimation
            discrete_uncertainties: bool, default=False
                Whether to show discrete uncertainties
            final: bool, default=False
                Whether to show the final map
            figsize: int, default=10
                Size of the figure
        """
        # setup
        if not hasattr(self, "raw_uncertainties"):
            uncertainties = 1.0
        else:
            uncertainties = self.raw_uncertainties
        self.map_boundary = self.get_map_boundary()

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
        country_shapes = countries.geometry
        country_shapes = country_shapes.apply(lambda x: make_valid(x))
        if show_states:
            start = time.time()
            countries.plot(
                ax=ax,
                linewidth=0.5 * (0.1 * figsize),
                facecolor="none",
                edgecolor="black",
            )
            print(f"Time elapsed to load countries: {time.time() - start}")

        # use a pre-compiles list of important roads
        # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_roads.zip
        # from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
        if show_roads:
            start = time.time()
            if self.verbose:
                print("Loading roads...")
            roads = gpd.read_file("map_features/roads/ne_10m_roads.shp")
            roads = roads.to_crs(epsg=3857)
            roads = roads[roads.geometry.within(self.map_to_polygon())]
            roads.plot(
                ax=ax,
                markersize=0.2 * (0.1 * figsize),
                linewidth=0.5 * (0.1 * figsize),
                color="gray",
                zorder=2,
            )
            print(f"Time elapsed to load roads: {time.time() - start}")

        # takes a lot of time
        # use a pre-compiled list of important cities
        # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_populated_places.zip
        # from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
        if show_cities:
            start = time.time()
            if self.verbose:
                print("Loading cities...")
            cities = gpd.read_file("map_features/cities/ne_10m_populated_places.shp")
            cities = cities.to_crs(epsg=3857)
            cities = cities[cities.geometry.within(self.map_to_polygon())]
            cities = cities[cities.geometry.within(unary_union(country_shapes))]
            cities.plot(
                ax=ax, markersize=1.0 * figsize, color="navy", marker="o", zorder=10
            )
            print(f"Time elapsed to load cities: {time.time() - start}")

        if show_points:
            start = time.time()
            points.plot(ax=ax, markersize=10, color="red")
            print(f"Time elapsed to load points: {time.time() - start}")

        # limit heatmap to landmass by asigning no data value to sea
        if self.verbose:
            print("Transforming heatmap...")
        nodata = np.nan
        with rasterio.open(self.rasterio_path) as heatmap:
            start = time.time()
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
            print(f"Time elapsed to transform heatmap: {time.time() - start}")

        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        new_map_path = f"intermediate/map_{self.method}_{self.region}_{self.resolution}_{self.version}_processed.tif"
        with rasterio.open(new_map_path, "w", **out_meta) as destination:
            destination.write(out_image)

        # plot the heatmap
        print("Plotting heatmap...") if self.verbose else None
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

        # define the heatmap color scale
        # values higher than the upper boundary are colored in the upmost color
        boundaries = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        cmap = colors.ListedColormap(buckets)

        # prepare the plot
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

        background_color = "0.7"
        ax.set_facecolor(
            background_color
        )  # background color light gray for landmass with uncertainties

        if show_uncertainties:
            start = time.time()
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
            print(f"Time elapsed to load uncertainties: {time.time() - start}")
        else:
            uncertainties = 1.0

        # set color for nan (nodata... sea) values
        # from https://stackoverflow.com/questions/2578752/how-can-i-plot-nan-values-as-a-special-color-with-imshow
        cmap.set_bad(color="blue")
        # TODO: not able to plot across 180 meridian
        rasterio.plot.show(raster, ax=ax, cmap=cmap, norm=norm, alpha=uncertainties)

        if final:
            anchor = self.get_text_anchor()

            fontsize_factor = 1.0 if self.region == "world" else 1.2
            region_text = self.region.replace("_", " ").title()
            if self.region == "world":
                text = f"""Hitchhiking waiting times
worldwide"""
            else:
                text = f"""Hitchhiking waiting times
in {region_text}"""

            ax.text(
                anchor.x,
                anchor.y,
                text,
                fontsize=figsize * fontsize_factor,
                fontfamily="serif",
                verticalalignment="top",
                fontweight="bold",
            )
            text = f"""



- insufficient data regions grayed out -
based on hitchmap.com and hitchwiki.org
https://github.com/Hitchwiki/hitchmap-data
made by Till Wenke (April 2024)
wenke.till@gmail.com"""

            ax.text(
                anchor.x,
                anchor.y,
                text,
                fontsize=figsize * fontsize_factor / 2,
                fontfamily="serif",
                verticalalignment="top",
            )

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

            # from https://stackoverflow.com/a/56900830
            cax = fig.add_axes(
                [
                    ax.get_position().x1 + 0.01,
                    ax.get_position().y0
                    + (ax.get_position().height * 2 / 3)
                    - (ax.get_position().height / 20),
                    0.02,
                    (ax.get_position().height / 3),
                ]
            )

            cbar_uncertainty = fig.colorbar(
                cm.ScalarMappable(norm=norm_uncertainties, cmap=cmap_uncertainties),
                cax=cax,
            )
            cbar_uncertainty.ax.tick_params(labelsize=figsize)
            cbar_uncertainty.ax.set_ylabel(
                "Uncertainty in waiting time estimation",
                rotation=90,
                fontsize=figsize * 2 / 3,
            )

        # from https://stackoverflow.com/a/56900830
        cax = fig.add_axes(
            [
                ax.get_position().x1 + 0.01,
                ax.get_position().y0 + (ax.get_position().height / 20),
                0.02,
                (ax.get_position().height / 3),
            ]
        )
        cbar = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
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
        self.raw_raster: np.array = None

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
