# hitchmap_data_cleaning
Relaunching the Hitchwiki map via https://github.com/bopjesvla/hitch with more meaningful hitchhiking spots.

Using the spots dump from 7th March 2023 for testing here. 25000 spots overall - 2500 in Germany.

Cleaning procedure ideas:

0. Merge spots that are consecutively located less than x meters away from each other.

1. Merge all spots within x meters of motorway service stations.
2. Merge all spots within x meters of gas stations - if not part of a service station.
3. Merge all spots within x meters of parking lot - if not part of a service station or gas station.
-> move the merged spot to the center of the map feature or the node (for some gas stations)



