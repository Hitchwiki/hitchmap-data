# TODO merge this to "hitch" repo

import numpy as np
from scipy.cluster.hierarchy import dendrogram
import pandas as pd

# returns distance in km!
def haversine(lon1, lat1, lon2, lat2, earth_radius=6367):
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = earth_radius * c
    return km

# can handle np arrays
# from https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.
    """

    # angeles degree to radian
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    return haversine(lon1, lat1, lon2, lat2)


def haversine_lists(x, y):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.
    """

    lon1, lat1, lon2, lat2 = map(np.radians, [x[0], x[1], y[0], y[1]])
    return haversine(lon1, lat1, lon2, lat2)
    


def haversine_points(A, B):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.
    """

    lon1, lat1, lon2, lat2 = map(np.radians, [A.y, A.x, B.y, B.x])
    return haversine(lon1, lat1, lon2, lat2)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# refetching :)
def places_from_points(points):
    # unify encodings (some points are in cp1252, should be utf-8)
    points.loc[points.id.isin(range(1000000, 1040000)), "comment"] = (
        points.loc[points.id.isin(range(1000000, 1040000)), "comment"]
        .str.encode("cp1252", errors="ignore")
        .str.decode("utf-8", errors="ignore")
    )

    points.datetime = pd.to_datetime(points.datetime)

    # aggregate features into "text" feature
    points["text"] = (
        points["comment"]
        + "\n\n―"
        + points["name"].fillna("Anonymous")
        + points.datetime.dt.strftime(", %B %Y").fillna("")
    )

    coords_in_degrees = points[["lon", "lat", "dest_lon", "dest_lat"]].values.T

    # new feature - the distance covered by the ride caught at the spot
    points["distance_from_spot"] = haversine_np(*coords_in_degrees)

    # put comments/ entries for same spot together
    groups = points.groupby(["lat", "lon"])

    # create the new places dataframe
    # we introduce it by only taking the feature we are not aggregating from the groups "country"
    # taking the first value of each group is arbitrary
    places = groups[["country"]].first()
    places["rating"] = groups.rating.mean().round()
    places["wait"] = points[~points.wait.isnull()].groupby(["lat", "lon"]).wait.mean()
    # eventually "distance_from_spot" become an average value too
    places["distance_from_spot"] = (
        points[~points.distance_from_spot.isnull()]
        .groupby(["lat", "lon"])
        .distance_from_spot.mean()
    )
    places["text"] = groups.text.apply(lambda t: "\n\n".join(t.dropna()))

    # group small countries together
    # this is for displaying them in one cluster on the map later
    places["country_group"] = places.country.replace(["BE", "NL", "LU"], "BNL")
    places.country_group = places.country_group.replace(["CH", "AT", "LI"], "ALP")
    places.country_group = places.country_group.replace(
        ["SI", "HR", "BA", "ME", "MK", "AL", "RS", "TR"], "BAL"
    )
    places.country_group = places.country_group.replace(["SK", "HU"], "SKHU")
    places.country_group = places.country_group.replace("MC", "FR")

    places.reset_index(inplace=True)
    # make sure high-rated are on top
    places.sort_values("rating", inplace=True, ascending=False)

    return places