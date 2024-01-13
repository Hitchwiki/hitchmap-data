import pandas as pd

# assume that places with same geometry are the same place
def get_stats(cleaned_places):
    stats = {}
    # places in original dataset
    stats["before"] = len(cleaned_places)

    stats["unchanged"] = len(
        cleaned_places[
            (cleaned_places.lat == cleaned_places.original_lat)
            & (cleaned_places.lon == cleaned_places.original_lon)
        ]
    )
    stats["deleted"] = len(cleaned_places[cleaned_places.road_delete])

    ### FEATURE ###
    ### places at a feature ...

    stats["places_service"] = len(cleaned_places[cleaned_places.feature == "service"])
    stats["places_fuel"] = len(cleaned_places[cleaned_places.feature == "fuel"])
    stats["places_parking"] = len(cleaned_places[cleaned_places.feature == "parking"])
    stats["places_port"] = len(cleaned_places[cleaned_places.feature == "port"])

    stats["places_feature"] = (
        stats["places_service"]
        + stats["places_fuel"]
        + stats["places_parking"]
        + stats["places_port"]
    )

    ### single places that got moved to a more accurate position in feature ...

    stats["single_service"] = (
        cleaned_places[cleaned_places.feature == "service"]
        .geometry.value_counts()[
            cleaned_places[cleaned_places.feature == "service"].geometry.value_counts()
            == 1
        ]
        .sum()
    )

    stats["single_fuel"] = (
        cleaned_places[cleaned_places.feature == "fuel"]
        .geometry.value_counts()[
            cleaned_places[cleaned_places.feature == "fuel"].geometry.value_counts()
            == 1
        ]
        .sum()
    )

    stats["single_parking"] = (
        cleaned_places[cleaned_places.feature == "parking"]
        .geometry.value_counts()[
            cleaned_places[cleaned_places.feature == "parking"].geometry.value_counts()
            == 1
        ]
        .sum()
    )

    stats["single_port"] = (
        cleaned_places[cleaned_places.feature == "port"]
        .geometry.value_counts()[
            cleaned_places[cleaned_places.feature == "port"].geometry.value_counts()
            == 1
        ]
        .sum()
    )

    stats["single_feature"] = (
        stats["single_service"]
        + stats["single_fuel"]
        + stats["single_parking"]
        + stats["single_port"]
    )

    ### places that got merged together at feature ...

    stats["merged_service"] = (
        cleaned_places[cleaned_places.feature == "service"]
        .geometry.value_counts()[
            cleaned_places[cleaned_places.feature == "service"].geometry.value_counts()
            > 1
        ]
        .sum()
    )

    stats["merged_fuel"] = (
        cleaned_places[cleaned_places.feature == "fuel"]
        .geometry.value_counts()[
            cleaned_places[cleaned_places.feature == "fuel"].geometry.value_counts() > 1
        ]
        .sum()
    )

    stats["merged_parking"] = (
        cleaned_places[cleaned_places.feature == "parking"]
        .geometry.value_counts()[
            cleaned_places[cleaned_places.feature == "parking"].geometry.value_counts()
            > 1
        ]
        .sum()
    )

    stats["merged_port"] = (
        cleaned_places[cleaned_places.feature == "port"]
        .geometry.value_counts()[
            cleaned_places[cleaned_places.feature == "port"].geometry.value_counts() > 1
        ]
        .sum()
    )

    stats["merged_feature"] = (
        stats["merged_service"]
        + stats["merged_fuel"]
        + stats["merged_parking"]
        + stats["merged_port"]
    )

    ### how many sites of each featue are affected by merges (to estimate how many places are merged into one)...
    ### = number of places at a feature that will be there after merges

    stats["service"] = (
        cleaned_places[cleaned_places.feature == "service"].geometry.value_counts() > 1
    ).sum()

    stats["fuel"] = (
        cleaned_places[cleaned_places.feature == "fuel"].geometry.value_counts() > 1
    ).sum()

    stats["parking"] = (
        cleaned_places[cleaned_places.feature == "parking"].geometry.value_counts() > 1
    ).sum()

    stats["port"] = (
        cleaned_places[cleaned_places.feature == "port"].geometry.value_counts() > 1
    ).sum()

    stats["feature"] = (
        stats["service"] + stats["fuel"] + stats["parking"] + stats["port"]
    )

    ### PROXIMITY ###

    stats["merged_proximity"] = len(cleaned_places[cleaned_places.proximity])

    # how many places will there be on the map after merging because of proximity merging
    stats["proximity"] = (
        cleaned_places[cleaned_places.proximity].geometry.value_counts() > 1
    ).sum()

    ### ROAD ###

    # those two could overlap
    # places moved closer to road
    stats["moved_road"] = len(cleaned_places[cleaned_places.road_distance])
    # places moved closer to road and did not get merged afterwards
    stats["single_moved_road"] = len(
        cleaned_places[cleaned_places.road_distance & ~cleaned_places.road_segment]
    )
    # places getting merged because they belong to the same road segment
    stats["merged_road"] = len(cleaned_places[cleaned_places.road_segment])

    # how many places will there be on the map after merging because of road merging
    stats["road"] = (
        cleaned_places[cleaned_places.road_segment].geometry.value_counts() > 1
    ).sum()

    ### OVERALL ###

    # the modifications a human had to verify
    # definitely
    stats["check_hard"] = (
        stats["feature"] + stats["proximity"] + stats["road"] + stats["deleted"]
    )
    # could say we do not have to check these (if they got moved less than X meters)
    stats["check_soft"] = stats["single_feature"] + stats["single_moved_road"]

    stats["after"] = (
        stats["unchanged"]
        + stats["check_hard"]
        + stats["check_soft"]
        - stats["deleted"]
    )

    return stats
