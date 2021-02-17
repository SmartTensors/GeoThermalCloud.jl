# GTcloud.jl: GeoThermal Cloud for Machine Learning

<div style="text-align: left; padding-bottom: 30px;">
	<a href="https://github.com/SmartTensors/GTcloud.jl">
    	<img src="logos/geothermalcloud-small.png" alt="geothermalcloud" width=25%  max-width=125px;/>
	</a>
</div>

**GTcloud.jl** is a repository containing all the data and codes required to demonstrate applications of machine learning methods for geothermal exploration.

**GTcloud.jl** includes:
- site data
- simulation scripts
- jupyter notebooks
- intermediate results
- code outputs, and
- summary figures

**GTcloud.jl** showcases the machine learning analyses performed for 2 geothermal sites:

- **Brady**: geothermal exploration of the Brady geothermal site, Nevada
- **SWNM**: geothermal exploration of Southwest New Mexico (SWNM) region

Reports, research papers, and presentations summarizing these machine learning analyses are also available and will be posted soon.

## Julia installation

Machine Learning analyses are performed using Julia.

To install the most recent version of Julia, please follow the instructions here:

- Julia: https://julialang.org/downloads/

## SmartTensors

Machine Learning analyses are performed using our [**SmartTensors**](https://github.com/SmartTensors) machine learning framework.

More information about [**SmartTensors**](https://github.com/SmartTensors) can be found at [smarttensors.github.io](https://smarttensors.github.io) and [tensors.lanl.gov](http://tensors.lanl.gov).

<div style="text-align: left; padding-bottom: 30px;">
	<a href="https://github.com/SmartTensors">
		<img src="logos/SmartTensorsNewSmaller.png" alt="SmartTensors" width=25%  max-width=125px;/>
	</a>
</div>

[**SmartTensors**](https://github.com/SmartTensors) provides tools for Unsupervised and Physics-Informed Machine Learning.

[**SmartTensors**](https://github.com/SmartTensors) includes a series of modules. Key modules are:

- [**NMFk**](https://github.com/SmartTensors/NMFk.jl): Nonnegative Matrix Factorization + k-means clustering
- [**NTFk**](https://github.com/SmartTensors/NTFk.jl): Nonnegative Tensor Factorization + k-means clustering

<div style="text-align: left; padding-bottom: 30px;">
	<a href="https://github.com/SmartTensors/NMFk.jl">
		<img src="logos/nmfk-logo.png" alt="nmfk" width=25%  max-width=125px;/>
	</a>
</div>

<div style="text-align: left; padding-bottom: 30px;">
	<a href="https://github.com/SmartTensors/NTFk.jl">
		<img src="logos/ntfk-logo.png" alt="ntfk" width=40%  max-width=125px;/>
	</a>
</div>

## SmartTensors installation

To install required [**SmartTensors**](https://github.com/SmartTensors) modules, execute in the Julia REPL:

```julia
import Pkg
Pkg.add("NMFk")
Pkg.add("NTFk")
Pkg.add("DelimitedFiles")
Pkg.add("Gadfly")
Pkg.add("Mads")
```