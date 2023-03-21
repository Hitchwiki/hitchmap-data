# hitchmap_data_cleaning
Relaunching the Hitchwiki map via https://github.com/bopjesvla/hitch with more meaningful hitchhiking spots.

### cleaning.ipynb
for executing the cleaning of map spots and getting a summary.

### map.ipynb
for observing the changes visually on a map.

---

Using the spots dump from 7th March 2023 for testing here. 

25000 spots overall - 2500 in Germany.

Cleaning procedure ideas:


Feature-based merging:

1. Merge all spots within 30 meters of motorway service stations.
2. Merge all spots within 30 meters of gas stations - if not part of a service station.
3. Merge all spots within 30 meters of parking lot - if not part of a service station or gas station.

-> move the merged spot to the center of the map feature or the node (for some gas stations)

Proximity-based merging:

4. Merge spots that are super close to each other (consecutively located less than 50 meters away from each other) to their centroid.

Road-based merging:

5. Identifiy spots in areas where one doesnt need a road (see feature-based merging & port areas).
6. Identifiy closest road segment (the road you are probably waiting at) for all spots but 1-3 & 5.
7. Delete spots that have the closest road more than 100 meters away.
8. Merge spots that are related to the same road segment.
9. For spots within 30 and 100 meters from their road segment - move them to 30 meters away from the segment.






