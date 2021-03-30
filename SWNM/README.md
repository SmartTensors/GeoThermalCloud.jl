# GeoThermalCloud.jl: Machine Learning framework for Geothermal Exploration

<div style="text-align: left; padding-bottom: 30px;">
	<a href="https://github.com/SmartTensors/GeoThermalCloud.jl">
    	<img src="../logos/geothermalcloud-small.png" alt="geothermalcloud" width=25%  max-width=125px;/>
	</a>
</div>

# Southwest New Mexico (SWNM)

Southwest New Mexico (SWNM) consists of low-, medium-, and high-temperature hydrogeothermal systems.

However, most of these systems are poorly characterized because of a lack of understanding of interrelationships between geological, hydrogeological, geophysical, and geothermal attributes.

SWNM is broadly divided into four physiographic provinces:
 - Colorado Plateau
 - Mogollon-Datil Volcanic Field (MDVF)
 - Basin and Range
 - Rio Grande rift

Our machine learning analyses presented here aim to identify patterns of hydrogeothermal systems in SWNM.

We target to find geothermal processes and attributes associated with the analyzed hydrogeothermal systems, in specific, and the analyzed physiographic provinces, in general.

# SWNM Machine Learning Repository

**SWNM** repository contains all the data and codes required to demonstrate applications of machine learning methods for geothermal exploration at this site.

**SWNM** repository includes:
- site data
- simulation scripts
- jupyter notebooks
- intermediate results
- code outputs, and
- summary figures

Reports, research papers, and presentations summarizing these machine learning analyses are also available and will be posted soon.

**SWNM** repository directories store:

- `data`: site data
- `results-*`: machine learning outputs
- `figures-*`: summary figures
- `notebook`: jupyter notebook to execute the machine learning analyses
- `scripts`: Julia scripts to execute the machine learning analyses

# SWNM Machine Learning Analyses

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
Pkg.add("JLD2")
Pkg.add("Gadfly")
Pkg.add("Cairo")
Pkg.add("Fontconfig")
Pkg.add("Mads")
```

## SmartTensors Machine Learning Analyses

### Julia script

To execute the machine learning analyses, run the following command:

```julia
include("notebook/SWNM.jl")
```

Note that the **SWNM** repository should be the current working directory.

### Jupyter notebook

To execute the machine learning analyses, open the jupyter notebook `notebook/SWNM.ipynb`.

The jupyter notebook is also saved in `html`, `latex`, `txt`, `pdf` and `markdown` formats.

The content of the Jupyter notebook is also available as a [Readme.md](https://github.com/SmartTensors/GeoThermalCloud.jl/tree/master/SWNM/notebook) file on GitHub.