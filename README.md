# hitchmap_data_cleaning

Relaunching the Hitchwiki map via https://github.com/bopjesvla/hitch with more meaningful hitchhiking spots.

### cleaning.ipynb

for executing the cleaning of map spots and getting a summary.

### map.ipynb

for observing what changed through cleaning visually on a map.

---

Using the spots dump from 7th March 2023 for testing here.

25000 spots overall - 2500 in Germany.

Cleaning procedure ideas in the following sequence:

**Feature-based merging:**

Merge all spots within 30 meters of a motorway service station.

1. Merge all spots within 30 meters of a gas station - if not part of a service station.
3. Merge all spots within 30 meters of a parking lot - if not part of a service station or gas station.

-> move the merged spot to the center of the map feature or the node (for some gas stations)

[sequence is though of as where do people tend to hh: service > gas > lot]

**Proximity-based merging:**

4. Merge spots that are super close to each other (consecutively located less than 50 meters away from each other) to the centroid of all those spots. e.g. 5 spots in a line each 50 m away from each other would be merged to the 3rd spot.

**Road-based merging:**

5. Identifiy spots in areas where one does not need a road (see feature-based merging areas & port areas) and exclude them from these steps.
6. Identifiy closest road segment (the road you are probably waiting at) for all spots but 1-3 & 5.
7. Delete spots that have the closest road more than 100 meters away.
8. Merge spots that are related to the same road segment. There are road segments in OSM that are x meters long at max, never cross another road and each direction of a road has its own segment. So one can safely assume that there are never multiple spots within the same segment with different hitchability.
9. For spots within 30 and 100 meters from their road segment - move them to 30 meters away from the segment.
