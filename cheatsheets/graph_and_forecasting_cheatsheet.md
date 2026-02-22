# Graph and Forecasting Cheatsheet

## NetworkX basics

- `nx.DiGraph()` for directed route networks
- `G.add_edge(dep, arr, **attrs)` to add routes
- `nx.betweenness_centrality(G)` to find routing bottlenecks
- `nx.pagerank(G)` to estimate influence/connectivity importance

## Centrality metrics meaning

- Degree: how many direct connections a node has
- Betweenness: how often a node sits on shortest paths (bottleneck risk)
- PageRank: importance from connected important nodes
- Clustering: how tightly a node's neighbors are connected

## Prophet / ARIMA basics

- Prophet: fast trend/seasonality forecasting with easy daily-date interface
- ARIMA: classical time-series model and a good fallback when Prophet is unavailable
- Forecast on aggregated daily/weekly series, not raw event rows

## Common forecasting pitfalls

- Using too little history for stable forecasts
- Ignoring structural changes (schedule shifts, disruptions)
- Treating forecast point estimates as guarantees
- Evaluating only fit, not operational usefulness

