import numpy as np
from scipy.cluster.hierarchy import dendrogram
import pandas as pd



def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def haversine(x, y):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """

    lon1, lat1, lon2, lat2 = map(np.radians, [x[0], x[1], y[0], y[1]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def haversine_points(A, B):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """

    lon1, lat1, lon2, lat2 = map(np.radians, [A.y, A.x, B.y, B.x])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km



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
    points.loc[points.id.isin(range(1000000,1040000)), 'comment'] = points.loc[points.id.isin(range(1000000,1040000)), 'comment'].str.encode("cp1252",errors='ignore').str.decode('utf-8', errors='ignore')

    points.datetime = pd.to_datetime(points.datetime)
    points['text'] = points['comment'] + '\n\n―' + points['name'].fillna('Anonymous') + points.datetime.dt.strftime(', %B %Y').fillna('')

    rads = points[['lon', 'lat', 'dest_lon', 'dest_lat']].values.T

    points['distance_from_spot'] = haversine_np(*rads)

    groups = points.groupby(['lat', 'lon'])

    places = groups[['country']].first()
    places['rating'] = groups.rating.mean().round()
    places['wait'] = points[~points.wait.isnull()].groupby(['lat', 'lon']).wait.mean()
    places['distance_from_spot'] = points[~points.distance_from_spot.isnull()].groupby(['lat', 'lon']).distance_from_spot.mean()
    places['text'] = groups.text.apply(lambda t: '\n\n'.join(t.dropna()))


    places['country_group'] = places.country.replace(['BE', 'NL', 'LU'], 'BNL')
    places.country_group = places.country_group.replace(['CH', 'AT', 'LI'], 'ALP')
    places.country_group = places.country_group.replace(['SI', 'HR', 'BA', 'ME', 'MK', 'AL', 'RS', 'TR'], 'BAL')
    places.country_group = places.country_group.replace(['SK', 'HU'], 'SKHU')
    places.country_group = places.country_group.replace('MC', 'FR')

    places.reset_index(inplace=True)
    # make sure high-rated are on top
    places.sort_values('rating', inplace=True, ascending=False)

    return places

callback = """\
function (row) {
    var marker;
    var color = {1: 'red', 2: 'orange', 3: 'yellow', 4: 'lightgreen', 5: 'lightgreen'}[row[2]];
    var opacity = {1: 0.3, 2: 0.4, 3: 0.6, 4: 0.8, 5: 0.8}[row[2]];
    var point = new L.LatLng(row[0], row[1])
    marker = L.circleMarker(point, {radius: 5, weight: 1 + 1 * (row[2] == 5), fillOpacity: opacity, color: 'black', fillColor: color});

    marker.on('click', function(e) {
        if ($$('.topbar.visible')) return

        points = [point]

        setTimeout(() => {
            bar('.sidebar.show-spot')
            $$('#spot-header').innerText = `${row[0].toFixed(5)}, ${row[1].toFixed(5)}`
            $$('#spot-summary').innerText = `Rating: ${row[2].toFixed(0)}/5
Waiting time in minutes: ${Number.isNaN(row[4]) ? '-' : row[4].toFixed(0)}
Ride distance in km: ${Number.isNaN(row[5]) ? '-' : row[5].toFixed(0)}`

            $$('#spot-text').innerText = row[3];
            if (!row[3] && Number.isNaN(row[5])) $$('#extra-text').innerHTML = 'No comments/ride info. To hide points like this, check out the <a href=/light.html>lightweight map</a>.'
            else $$('#extra-text').innerHTML = ''
        },100)

        L.DomEvent.stopPropagation(e)
    })

    // if(row[2] >= 4) marker.bringToFront()

    return marker;
};
"""