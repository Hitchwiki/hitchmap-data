import itertools
import json
import os
import pickle
import sqlite3
import sys
import time
import warnings
from itertools import chain
from typing import List

import anytree
import folium
import folium.plugins
import geopandas as gpd
import numpy as np
import osmnx
import pandas as pd
import pyproj
import scipy
import sklearn
from anytree import Node, RenderTree
from matplotlib import pyplot as plt
from osmnx._errors import InsufficientResponseError
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import (LineString, MultiPoint, MultiPolygon, Point,
                              Polygon)
from shapely.ops import nearest_points, unary_union
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances

pd.options.display.max_rows = 4000
np.set_printoptions(threshold=sys.maxsize)

osmnx.settings.use_cache = True
osmnx.settings.log_console = False

import logging

from stats import *
from utils import *

logging.basicConfig(filename='logging.log', encoding='utf-8', level=logging.INFO)
# progress bars
from tqdm import tqdm

tqdm.pandas()
metric_crs = 'EPSG:3857'

# used for gps - geographic crs
standard_crs = 'EPSG:4326'
# fetch data
def points_from_country(region="Germany", level="country"):
    file = "./points.sqlite"
    points = pd.read_sql("select * from points where not banned", sqlite3.connect(file))
    region_osm_data = osmnx.geocode_to_gdf({level: region})
    polygon = region_osm_data.iloc[0].geometry
    # TODO repeated pattern?
    points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.lon, points.lat))
    points = points[points.progress_apply(lambda point: point["geometry"].within(polygon), axis=1)]

    return points
# TODO why working on places and points at the same time - places are sufficient - modify points at the end only
# TODO delete places in water
# TODO write test for the single components?

# select the country here
# contains all entries for the same place as single samples
region = "Saxony"
level = "state"
points = points_from_country(region, level)
# points = points_from_country("Germany", "country")

# entries for the same spot are grouped together
places = places_from_points(points)
# to later plot the difference between orgiginal and modified places
places["original_lat"] = places["lat"]
places["original_lon"] = places["lon"]
# possible cleaning reasons
# feature specifies the type of feature as str 
# other reasons are binary
places["feature"] = None
places["proximity"] = False
places["road_delete"] = False
places["road_distance"] = False
places["road_segment"] = False

original_places = places

places = gpd.GeoDataFrame(places, geometry=gpd.points_from_xy(places.lon, places.lat), crs=standard_crs)
# FULL PIPELINE

# "geometry" is the feature that gets modified in our correction process
# aligning the other features as to be done separately
def update_places(places):
    places["lon"] = places.geometry.x
    places["lat"] = places.geometry.y
    return places
def cluster_places(places, n_clusters=100, linkage="average", distance_threshold=None):
    places = update_places(places)
    X = np.array(places[["lon", "lat"]])
    # with haversine distance we can use non metric crs
    # and get distances in km
    D = pairwise_distances(X, X, metric=haversine_lists, n_jobs=-1)

    # hierarchical clustering technique
    # clustering does not have to be perfect
    # we want to cut the whole dataset of places into n_clusters chunks
    # to handle each chunk separately -> this will save us compute
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        compute_distances=True,
        metric="precomputed",
        distance_threshold=distance_threshold
    )
    clustering.fit(D)

    return clustering
# TODO to check internals try:

# plt.title("Hierarchical Clustering Dendrogram")
# # plot the top three levels of the dendrogram
# plot_dendrogram(clustering, truncate_mode="level", p=3)
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.show()

# with open("germany_100cluster_average_from_german_points.pkl", "wb") as f:
#     pickle.dump(clustering, f)
## 1-3 Map Featues
# choose the cluster count so that there are all clusters with at maximum x
# this depends on the region chosen
N_CLUSTERS = 100
clustering = cluster_places(places, n_clusters=N_CLUSTERS)

# assign a cluster to each place
places["feature_cluster"] = clustering.labels_

# collect places that have already been changed (because they belong to a map feature)
# to be modified in later pipeline steps
PLACES_AT_FEATURES: List[Point] = []
# distance to a feature (potentially an area) in meters
# for which a place should belong to the mentioned feature
services_threshold = 30
fuel_threshold = 30
parking_threshold = 30
# there are often streets close to ports; place should only be matched to port
# if the hh used the port to get on a ship
port_threshold = 10
# inspired by https://github.com/gboeing/osmnx/blob/3822ed659f1cc9f426a990c02dd8ca6b3f4d56d7/osmnx/utils_geo.py#L420
# blows up a place to be a circle
def circle(place, radius):
    point = place.geometry
    earth_radius = 6_371_009  # meters
    delta_lat = (radius / earth_radius) * (180 / np.pi)
    return point.buffer(delta_lat)

# let the places be circles and store this geometry as a separate feature
places['services_geometry'] = places.progress_apply(lambda row: circle(row, services_threshold), axis=1)
places['fuel_geometry'] = places.progress_apply(lambda row: circle(row, fuel_threshold), axis=1)
places['parking_geometry'] = places.progress_apply(lambda row: circle(row, parking_threshold), axis=1)
places['port_geometry'] = places.progress_apply(lambda row: circle(row, port_threshold), axis=1)

# features to store the new (corrected) point geometry of a place
# TODO necessary to have multiple places within one service area?
places['service_corrected'] = None
places['fuel_corrected'] = None
places['parking_corrected'] = None
# places in a port area should be merged as well
# TODO think about if there are different places within a port area
places['port_corrected'] = None
### run ~10 min
warnings.simplefilter(action="ignore", category=ShapelyDeprecationWarning)


def get_close_feature(geometry, features):
    relevant = features[features.intersects(geometry)]
    if not relevant.empty:
        # potentially there are multiple features close to a place
        # we choose the first one
        # and return get to corrected position of the place
        single_feature = relevant.iloc[0]
        return single_feature.geometry.centroid
    else:
        return None


def correct_place(place):
    new_location = None
    feature = None
    # reflects the precedence of features we want to match the places to
    if place.port_corrected is not None:
        new_location = place.port_corrected
        feature = "port"
    elif place.service_corrected is not None:
        new_location = place.service_corrected
        feature = "service"
    elif place.fuel_corrected is not None:
        new_location = place.fuel_corrected
        feature = "fuel"
    elif place.parking_corrected is not None:
        new_location = place.parking_corrected
        feature = "parking"

    if new_location is not None:
        # correct the place
        return new_location, feature
    else:
        return place.geometry, None


def feature_from_region(region: Polygon, tags: dict):
    try:
        features = osmnx.features.features_from_polygon(region, tags=tags)
        return features
    except InsufficientResponseError:
        return gpd.GeoDataFrame()


for i in range(0, N_CLUSTERS):
    places_cluster = places.loc[places.feature_cluster == i]

    # retrieving the OSM features that are relavant for the current cluster
    places_in_cluster = MultiPolygon(places_cluster.services_geometry.to_list())
    current_region = places_in_cluster.convex_hull

    start = time.time()

    # TODO query everything once and cache might result in a speedup here?!
    # e.g. area="Germany"; fuel = osmnx.features.features_from_place(area, tags={'amenity': 'fuel'})

    # TODO try this
    # # USE THIS
    # # speedup
    # cf = '["amenity"~"fuel|parking"]'
    # G = osmnx.graph_from_place(area, custom_filter=cf)
    # cf = '["highway"~"services"]'
    # G = osmnx.graph_from_place(area, custom_filter=cf)
    # fig, ax = osmnx.plot_graph(G)
    # #"natural"~"water"

    services = feature_from_region(current_region, tags={"highway": "services"})
    # query both features at once is faster
    fuel_parking = feature_from_region(
        current_region, tags={"amenity": ["parking", "fuel"]}
    )
    port = feature_from_region(current_region, tags={"industrial": "port"})

    logging.info(f"OSM query time for features: {time.time() - start}")
    start = time.time()

    if not services.empty:
        try:
            # polygons of service stations can be found via the "way" key
            services = services.loc["way"]
            places.loc[
                places.feature_cluster == i, "service_corrected"
            ] = places_cluster.apply(
                lambda place: get_close_feature(place.services_geometry, services),
                axis=1,
            )
        except:
            # there are no services in the region
            pass

    if not fuel_parking.empty:
        fuel = fuel_parking.loc[fuel_parking.amenity == "fuel"]
        parking = fuel_parking.loc[fuel_parking.amenity == "parking"]

        places.loc[places.feature_cluster == i, "fuel_corrected"] = places_cluster.apply(
            lambda place: get_close_feature(place.fuel_geometry, fuel), axis=1
        )
        places.loc[places.feature_cluster == i, "parking_corrected"] = places_cluster.apply(
            lambda place: get_close_feature(place.parking_geometry, parking), axis=1
        )

    if not port.empty:
        places.loc[places.feature_cluster == i, "port_corrected"] = places_cluster.apply(
            lambda place: get_close_feature(place.port_geometry, port), axis=1
        )

    logging.info(f"Time to match places to features: {time.time() - start}")

(
    places["geometry"],
    places["feature"],
) = zip(*places.progress_apply(correct_place, axis=1))


# we assume that you either hitchhike from a map feature or a road
# save for later - if a place is attached to a feature it is protected from further changes
places_at_feature = places.loc[places.feature.notnull()]
# to treat places that are not a feature
places = places[places.feature.isnull()]

# mind that lat, lon of places are not changed yet - "geometry" contains the up-to-date location
## 4 Proximity
# distance between spots in meter for which spots are seen as the same - transitive
merge_threshold = 50
merge_threshold = merge_threshold / 1000

# calculate again because places are changed now
# modification to clustering before: in one cluster there will not be places transitively more than merge_threshold apart
clustering = cluster_places(places, n_clusters=None, linkage="single", distance_threshold=merge_threshold)

n_clusters = len(np.unique(clustering.labels_))
places["proximity_cluster"] = clustering.labels_

# merge places to the centroid of their group
for cluster in range(0, n_clusters):
    if len(places[places.proximity_cluster == cluster]) > 1:
        merged_location = places[places.proximity_cluster == cluster].geometry.unary_union.centroid
        places.loc[places.proximity_cluster == cluster, ["geometry", "proximity"]] = merged_location, True
## 5-9 Road
places = update_places(places)

# using a metric coordinate system here for distance calculation
places.to_crs(metric_crs, inplace=True)

# new features - identifier of the nearest road segement and new location closer to a road
places[['nearest_road','road_correction']] = None, None
# handles a "line" composed of one or more line segments.
def cut(line: LineString, distance: float) -> List[LineString]:
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    # iterate through the line segments
    for i, point in enumerate(coords):
        # calculate the distance from the starting point to the point of the current line segment
        point_distance = line.project(Point(point))
        if point_distance == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if point_distance > distance:
            # get the point on the line that is distance away from the starting point
            # this has to lie on the current line segment
            # thus we have to make the cut in this segment
            cutting_point = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cutting_point.x, cutting_point.y)]),
                LineString([(cutting_point.x, cutting_point.y)] + coords[i:])]
### run
# above seems like the place was not meant to be on a road or the creator did not take any effort to place
# it accurately on the road - in either case the place can be deleted as it introduces noise
road_upper_threshold = 100
# below places are considered accurate enough - the user will know which road the spot is related to
road_lower_threshold = 30
# distance to use when searching for road points around a place
# road points should be frequent but are definitely further away than the direct distance to the road
# at important marks of a road there are road (ancor) points
# thus one can assume that when the place is close to a road but not close enough to a road point that it is placed
# at a location where you can/ should not hitchhike
road_search_radius = 300


def get_nearest_road(place):
    # OSM query could fail
    try:
        # query a graph of all roads around the place
        # using lat, lon in geometric coordinates here - make sure they are up-to-date
        # TODO bound by api queries - can this be done more efficient?
        roads = osmnx.graph.graph_from_point(
            (place.lat, place.lon),
            dist=road_search_radius,
            network_type="drive",
            retain_all=True,
            truncate_by_edge=True,
        )
        roads = osmnx.projection.project_graph(roads, to_crs=metric_crs)
        # get the nearest road to the place
        # returns nearest edges as (u, v, key)
        nearest = osmnx.distance.nearest_edges(
            roads, place.geometry.x, place.geometry.y, return_dist=True
        )
        road = nearest[0]
        dist = nearest[1]
        # identify road by its nodes
        road_name = f"{road[0]}-{road[1]}"

        if dist > road_upper_threshold:
            # for this case delete the place
            return None, None
        elif road_lower_threshold < dist <= road_upper_threshold:
            # get the coordinates of new spot that is moved closer to the road than original spot
            # translate the road into a geometry object
            _, gdf_roads = osmnx.utils_graph.graph_to_gdfs(roads)
            road_geom = gdf_roads.loc[road].geometry
            # get the point on the road that is closest to the place to project the place on the road
            # first point would be the place itself
            nearest_point = nearest_points(place.geometry, road_geom)[1]
            # move the place to the road but not directly on it
            # using LineString here as the generalization of a line
            # osm graph and gdf use LineString as well to describe road segments
            tangent_line = LineString([nearest_point, place.geometry])
            cut_line = cut(tangent_line, road_lower_threshold)
            return road_name, Point(cut_line[0].coords[-1])
        else:
            # if the place is already placed accurately enough on the road we want to know which road it belongs to
            # but do not need to adjust the place
            return road_name, None
    except:
        return None, None


# introduce new features
# id of the nearest road
# and corrected location
(
    places["nearest_road"],
    places["road_correction"],
) = zip(*places.progress_apply(get_nearest_road, axis=1))
# have to retransform because in the previous step we used a metric coordinate system
crs_transformer = pyproj.Transformer.from_crs(metric_crs, standard_crs)
places.to_crs(standard_crs, inplace=True)
# delete points that are not related to a road nor to a map feature
places.loc[places.nearest_road.isna(), "road_delete"] = True
# save for later - if a place is deleted it is not treated further
places_deleted = places[places.road_delete]
# to treat places that are not a feature
places = places[~places.road_delete]


# TODO in human tool - give chance to move the spot to a suitable location
# correct spots to their new location at the road


# give places their new location closer to their road
def attach_place_to_road(place):
    if place.road_correction is not None:
        lat, lon = crs_transformer.transform(
            place.road_correction.x, place.road_correction.y
        )
        return Point(lon, lat), True
    else:
        return place.geometry, False


(
    places["geometry"],
    places["road_distance"],
) = zip(*places.progress_apply(attach_place_to_road, axis=1))
# TODO drop? - most examples where this was applied came with informatin loss
# only do it for road segements with length < X meters

# find groups of places that are at the same road segment so they can be merged
places_by_road_segment = places.groupby("nearest_road")

for group_id, places_group_at_road_segment in places_by_road_segment:
    if len(places_group_at_road_segment) > 1:
        place_indices_at_road_segment = places_group_at_road_segment.index
        # TODO better solution than centroid here?
        # use latest comment or place with most comments instead
        merged_location = places.loc[place_indices_at_road_segment].geometry.unary_union.centroid
        places.loc[place_indices_at_road_segment, "geometry"] = merged_location
        places.loc[place_indices_at_road_segment, "road_segment"] = True
# recombine all places that were treated separately
cleaned_places = pd.concat([places, places_at_feature, places_deleted])
cleaned_places = update_places(cleaned_places)
stats = get_stats(cleaned_places)
with open(f"./stats/stats_{region}.txt", "w") as f:
    f.write(str(stats))
# map
# difference map
# green points are from the new cleaned map vs red points from the original map

map = folium.Map(prefer_canvas=True, control_scale=True)

def dot(lat, lon, color):
    folium.CircleMarker([lat, lon], opacity=0.0, radius=5, fillOpacity=1.0, fillColor=color).add_to(map)

def line(lat1, lon1, lat2, lon2, color):
    folium.PolyLine([(lat1, lon1), (lat2, lon2)], color=color).add_to(map)

for country, group in cleaned_places.groupby("country_group"):
    for index, place in group.iterrows():
        if place.road_delete:
            dot(place.lat, place.lon, "black")
        else:
            dot(place.lat, place.lon, "lightgreen")
            # if the place did not change only the red dot is visible
            dot(place.original_lat, place.original_lon, "red")

            correction_line_color = None
            if place.feature != None:
                correction_line_color = "orange"
            # after proximity merge places can still get moved/ merged because of there relation to a road
            elif place.road_segment:
                correction_line_color = "purple"
            elif place.road_distance:
                correction_line_color = "green"
            elif place.proximity:
                correction_line_color = "blue"
            
            
            if correction_line_color is not None:
                line(place.lat, place.lon, place.original_lat, place.original_lon, correction_line_color)

# show
map.save("map.html")
map
# TODO still have to perform the actual merge on both places and underlying points

# eventually store results
cleaned_places.to_csv(f'data/cleaned_places_{region}.csv', index=True)