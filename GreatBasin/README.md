# GeoThermalCloud.jl: Machine Learning framework for Geothermal Exploration

<div style="text-align: left; padding-bottom: 30px;">
	<a href="https://github.com/SmartTensors/GeoThermalCloud.jl">
    	<img src="../logos/geothermalcloud-small.png" alt="geothermalcloud" width=25%  max-width=125px;/>
	</a>
</div>

# Great Basin, USA

Great Basin includes multiple geothermal reservoirs ranging from low- to high-temperature.

Great Basin has huge potential geothermal potential.

Further explorations require a better understanding of the local/regional patterns in various geothermal-related attributes observed in the region.

Here, we applied SmartTensors machine learning methods to analyze the available geothermal/geochemical data and to better understand/predict the spatial distribution of the geothermal resources

# Great Basin Machine Learning Repository

**Great Basin** repository contains all the data and codes required to demonstrate applications of machine learning methods for geothermal exploration at this site.

**Great Basin** repository includes:
- site data
- simulation scripts
- jupyter notebooks
- intermediate results
- code outputs, and
- summary figures

Reports, research papers, and presentations summarizing these machine learning analyses are also available and will be posted soon.

**Great Basin** repository directories store:

- `data`: site data
- `maps-*`: site maps
- `results-*`: machine learning outputs
- `figures-*`: summary figures
- `notebook`: jupyter notebook to execute the machine learning analyses

# Great Basin Machine Learning Analyses

## SmartTensors

Machine Learning analyses are performed using the [**SmartTensors**](https://github.com/SmartTensors) machine learning framework.

<div style="text-align: left; padding-bottom: 30px;">
	<a href="https://github.com/SmartTensors">
		<img src="../logos/SmartTensorsNewSmaller.png" alt="SmartTensors" width=25%  max-width=125px;/>
	</a>
</div>

[**SmartTensors**](https://github.com/SmartTensors) provides tools for Unsupervised and Physics-Informed Machine Learning.

More information about [**SmartTensors**](https://github.com/SmartTensors) can be found at [smarttensors.github.io](https://smarttensors.github.io) and [tensors.lanl.gov](http://tensors.lanl.gov).


[**SmartTensors**](https://github.com/SmartTensors) includes a series of modules. Key modules are:

- [**NMFk**](https://github.com/SmartTensors/NMFk.jl): Nonnegative Matrix Factorization + k-means clustering
- [**NTFk**](https://github.com/SmartTensors/NTFk.jl): Nonnegative Tensor Factorization + k-means clustering

<div style="text-align: left; padding-bottom: 30px;">
	<a href="https://github.com/SmartTensors/NMFk.jl">
		<img src="../logos/nmfk-logo.png" alt="nmfk" width=25%  max-width=125px;/>
	</a>
</div>

<div style="text-align: left; padding-bottom: 30px;">
	<a href="https://github.com/SmartTensors/NTFk.jl">
		<img src="../logos/ntfk-logo.png" alt="ntfk" width=40%  max-width=125px;/>
	</a>
</div>

## GeoThermalCloud installation

To install all required GeoThermalCloud and SmartTensors modules, execute in the Julia REPL:

```julia
import Pkg
Pkg.add("GeoThermalCloud")
```
## GeoThermalCloud Machine Learning Analyses

### Julia script

GeoThermalCloud machine learning analyses can be executed as follows:

```julia
import GeoThermalCloud
GeoThermalCloud.GreatBasin() # performs analyses of the Great Basin region
```
### Jupyter notebook

GeoThermalCloud machine learning analyses can be also executed as Jupyter notebooks as well

```julia
import GeoThermalCloud
GeoThermalCloud.GreatBasin(notebook=true) # opens Jupyter notebook for analyses of the Great Basin region
```

To execute the machine learning analyses, you can also open the jupyter notebook `notebook/GreatBasin.ipynb`.

The jupyter notebook is also saved in `html`, `latex`, `txt`, `pdf` and `markdown` formats.

The content of the Jupyter notebook is also available as a [Readme.md](https://github.com/SmartTensors/GeoThermalCloud.jl/tree/master/GreatBasin/notebook) file on GitHub.