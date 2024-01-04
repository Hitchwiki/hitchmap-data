# hitchmap_data_cleaning

Relaunching the Hitchwiki map via https://github.com/bopjesvla/hitch with more meaningful hitchhiking spots.

### clean_map.ipynb

for executing the cleaning of map spots and getting a summary.

---

Using the spots dump from 7th March 2023 for testing here.

25000 spots overall - 2500 in Germany.

Merging = combine places that should be the same into one.

Moving = give a place a new location that is more meaningful.

If features from a map are needed for a task those are obtained from OSM.


Cleaning procedure ideas in the following sequence:

**Feature-based merging (orange):**

1. Merge all spots within 30 meters of a motorway service station or rest area.
2. Merge all spots within 30 meters of a gas station - if not part of a service station.
3. Merge all spots within 30 meters of a parking lot - if not part of a service station or gas station.
4. Merge all spots within 30 meters of a port.

-> move the merged spot to the center of the map feature or the node (for some gas stations)

-> exclude those spots from being treated further

[sequence is thought of as where do people tend to hh: service > gas > parking]

**Proximity-based merging (blue):**

1. Merge spots that are super close to each other (transitively located less than 50 meters away from each other) to the centroid of all those spots. e.g. 5 spots in a line each 50 m away from each other would be merged to the 3rd spot.

**Road-based merging (purple) and moving (green):**

1. Identifiy closest road segment (the road you are probably waiting at) for a place. For this the road must have a node less than 300 m away from the place.
2. Delete spots that have the closest road more than 100 meters away.
3. Move places that are between 30 and 100 m was from a road closer to the road (to 30 m distance).
4. Merge spots that are related to the same road segment to their centriod.
