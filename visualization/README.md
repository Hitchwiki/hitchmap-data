# heatchmap

Important: if you are willing to have a look or contribute reach out to me first. This repo needs some cleaning :)

### Data

2024-03-15 dump (dump.sqlite) from https://hitchmap.com/

### Modelling

- do not want to color where we have no certainty/ data
- in europe normal distribution with 50 km stdv is a good asumption

### Problems

**Weighted Gaussian Convolution**

For low density of points far away averaging them works poorly because calculation are on really small numbers (tail of normal distribution). We get hard edges:

![1703636597008](image/README/1703636597008.png)

Instabilities between color scale sections:

![1703679266533](image/README/1703679266533.png)

**Gaussian Process**
