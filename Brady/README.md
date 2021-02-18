# GTcloud.jl: GeoThermal Cloud for Machine Learning

<div style="text-align: left; padding-bottom: 30px;">
	<a href="https://github.com/SmartTensors/GTcloud.jl">
    	<img src="../logos/geothermalcloud-small.png" alt="geothermalcloud" width=25%  max-width=125px;/>
	</a>
</div>

# Brady Geothermal Site, Nevada

Brady geothermal field is located in northwestern, Nevada, USA. It is located in the Basin and Range physiographic province.

It has seen geothermal electricity production since 1992 and research or exploration since at least 1959.

The existing hydrothermal system supplies hot fluid to two power stations and a direct-use vegetable drying facility.

Electricity production capacity at Brady is 26.1 MWe, and ~7 MWth is supplied to the drying facility.

Temperatures of produced fluid have been ~130-185°C, though temperatures as high as 219°C have been measured as well.

These relatively high temperatures at relatively shallow levels (300-600 depth for some production wells) occur as a result of either convective upwelling driven by temperature control differences in fluid density, or hydraulic head driven circulation through the hot rock.

In either case, relatively high heat flow at the site is associated with crustal thinning provides the heat.

# Brady Machine Learning Repository

**Brady** repository contains all the data and codes required to demonstrate applications of machine learning methods for geothermal exploration at this site.

**Brady** repository includes:
- site data
- simulation scripts
- jupyter notebooks
- intermediate results
- code outputs, and
- summary figures

Reports, research papers, and presentations summarizing these machine learning analyses are also available and will be posted soon.

**Brady** repository directories store:

- `data`: site data
- `map`: site maps
- `results-*`: machine learning outputs
- `figures-*`: summary figures
- `notebook`: jupyter notebook to execute the machine learning analyses
- `scripts`: Julia scripts to execute the machine learning analyses

# Brady Machine Learning Analyses

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

## SmartTensors Installation

To install required [**SmartTensors**](https://github.com/SmartTensors) modules, execute in the Julia REPL:

```julia
import Pkg
Pkg.add("NMFk")
Pkg.add("DelimitedFiles")
Pkg.add("JLD")
Pkg.add("Gadfly")
Pkg.add("Cairo")
Pkg.add("Fontconfig")
Pkg.add("Mads")
```

## SmartTensors Machine Learning Analyses

### Julia script

To execute the machine learning analyses, run the following command:

```julia
include("scripts/execute.jl")
```

Note that the **Brady** repository should be the current working directory.

### Jupyter notebook

To execute the machine learning analyses, open the jupyter notebook `notebook/Brady.ipynb`

The jupyter notebook is also saved in `html`, `pdf` and `markdown` formats.