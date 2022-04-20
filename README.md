# GeoThermalCloud.jl: Machine Learning framework for Geothermal Exploration

<div style="text-align: left; padding-bottom: 30px;">
	<a href="https://github.com/SmartTensors/GeoThermalCloud.jl">
    	<img src="logos/geothermalcloud-small.png" alt="geothermalcloud" width=25%  max-width=125px;/>
	</a>
</div>

**GeoThermalCloud.jl** is a repository containing all the data and codes required to demonstrate applications of machine learning methods for geothermal exploration.

**GeoThermalCloud.jl** includes:
- site data
- simulation scripts
- jupyter notebooks
- intermediate results
- code outputs
- summary figures
- readme markdown files

**GeoThermalCloud.jl** showcases the machine learning analyses performed for the following geothermal sites:

- **Brady**: geothermal exploration of the Brady geothermal site, Nevada
- **SWNM**: geothermal exploration of the Southwest New Mexico (SWNM) region
- **GreatBasin**: geothermal exploration of the Great Basin region

Reports, research papers, and presentations summarizing these machine learning analyses are also available and will be posted soon.

## Julia installation

GeoThermalCloud Machine Learning analyses are performed using Julia.

To install the most recent version of Julia, follow the instructions at https://julialang.org/downloads/

## GeoThermalCloud installation

To install all required the modules, execute in the Julia REPL:

```julia
import Pkg
Pkg.add("GeoThermalCloud")
```
## GeoThermalCloud examples

GeoThermalCloud machine learning analyses can be executed as follows:

```julia
import Pkg
Pkg.add("GeoThermalCloud")
import GeoThermalCloud

GeoThermalCloud.SWNM() # performs analyses of the Sounthwest New Mexico region
GeoThermalCloud.GreatBasin() # performs analyses of the Great Basin region
GeoThermalCloud.Brady() # performs analyses of the Brady site, Nevada
```

GeoThermalCloud machine learning analyses can be also executed as Jupyter notebooks as well

```julia
GeoThermalCloud.notebooks() # open Jupyter notebook to acccess all GeoThermalCloud notebooks
GeoThermalCloud.SWNM(notebook=true) # opens Jupyter notebook for analyses of the Sounthwest New Mexico region
GeoThermalCloud.GreatBasin(notebook=true) # opens Jupyter notebook for analyses of the Great Basin region
GeoThermalCloud.Brady(notebook=true) # opens Jupyter notebook for analyses of the Brady site, Nevada
```
## SmartTensors

GeoThermalCloud analyses are performed using the [**SmartTensors**](https://github.com/SmartTensors) machine learning framework.

<div style="text-align: left; padding-bottom: 30px;">
	<a href="https://github.com/SmartTensors">
		<img src="logos/SmartTensorsNewSmaller.png" alt="SmartTensors" width=25%  max-width=125px;/>
	</a>
</div>

[**SmartTensors**](https://github.com/SmartTensors) provides tools for Unsupervised and Physics-Informed Machine Learning.

More information about [**SmartTensors**](https://github.com/SmartTensors) can be found at [smarttensors.github.io](https://smarttensors.github.io) and [tensors.lanl.gov](http://tensors.lanl.gov).

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
## Publications
### Peer reviewed
- Rau, E., Ahmmed, B., Vesselinov, V.V, Mudunuru, M.K., and Karra, S. (in preparation): Geothermal play development using machine learning, geophysics, and reservoir simulation, Geothermics.
- Ahmmed, B. and Vesselinov, V.V. (in review): Machine learning and shallow groundwater chemistry to identify geothermal resources, to be submitted to Renewable Energy, http://dx.doi.org/10.2139/ssrn.4072512. 
- Vesselinov, V.V., Ahmmed, B., Mudunuru, M.K., Pepin, J. D., Burns, E.R., Siler, D.L., Karra, S., and Middleton, R.S. (in review): Discovering Hidden Geothermal Signatures using Unsupervised Machine Learning, Geothermics.
- Siler, D.L., Pepin, J.D., Vesselinov, V.V., Mudunuru, M.K., and Ahmmed, B. (2021): Machine learning to identify geologic factors associated with production in geothermal  fields: A case-study using 3D geologic data, Brady geothermal field, Nevada, Geothermal Energy, https://doi.org/10.1186/s40517-021-00199-8.
