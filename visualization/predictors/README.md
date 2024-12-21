# Building predictive models and drawing hitchhiking maps

`final_maps`/ ... high resolution maps

`maps`/ ... more maps

`intermediate/` ... maps in raw format

`models/` ... pickled models to draw the maps

`draw_map.py & draw_maps.ipynb` ... creating the final maps that are shown in `summary_maps.ipynb`

`experiments_*.ipynb` ... our own experiments with the models from `models.py`.

`models.py` ... models we applied and built so far include:

- Gaussian Process
- weighted Gaussians
- average of all waiting times (baseline)
- average of same-size rectangles

`<ins>summary_maps.ipynb` ... short writeup for the community`</ins>`

`<ins>writeup_heatchmap.ipynb` ... long ML heavy writeup`</ins>`

## Framework to build your own models

We encourage you to contribute to this repository by filing a PR once you start building new models and maps.

How you build models, evaluate them, and draw new maps from them is showcased in `writeup_heatchmap.ipynb`.

One way to build your model is to inherit from `MapBasedModel` in `models.py` - this class is responsible for drawing maps as well.

## We wrote about our work

- https://tillwenke.github.io/2024/05/06/hitchhiking-worldwide.html
- https://tillwenke.github.io/2024/04/21/hitchmap-gp.html
