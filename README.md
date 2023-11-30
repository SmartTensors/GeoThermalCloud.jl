# GeoThermalCloud: A Machine Learning Framework for Geothermal Resources Exploration

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

### Book chapter

- Vesselinov, V.V., Mudunuru, M.K. Ahmmed, B., Karra, S., and O’Malley, D., (accepted): Machine Learning to Discover, Characterize, and Produce Geothermal Energy, CRS Press, Boca Raton, FL.

### Peer reviewed

- Rau, E., Ahmmed, B., Vesselinov, V.V, Mudunuru, M.K., and Karra, S. (in review): Geothermal play development using machine learning, geophysics, and reservoir simulation, Renewable Energy.
- Mudunuru, M.K., Ahmmed, B., Rau, E., Vesselinov, V.V., and Karra, S. (2023): Machine Learning for Geothermal Resource Exploration in the Tularosa Basin, New Mexico. Energies, 16(7), 3098
- Mudunuru, M.K., Vesselinov, V.V. and Ahmmed, B., 2022. GeoThermalCloud: Machine Learning for Geothermal Resource Exploration. Journal of Machine Learning for Modeling and Computing.
- Ahmmed, B. and Vesselinov, V.V., 2022. Machine learning and shallow groundwater chemistry to identify geothermal prospects in the Great Basin, USA. Renewable Energy, 197, pp.1034-1048.
- Vesselinov, V.V., Ahmmed, B., Mudunuru, M.K., Pepin, J.D., Burns, E.R., Siler, D.L., Karra, S. and Middleton, R.S., 2022. Discovering hidden geothermal signatures using non-negative matrix factorization with customized k-means clustering. Geothermics, 106, p.102576.
- Siler, D.L., Pepin, J.D., Vesselinov, V.V., Mudunuru, M.K., and Ahmmed, B. (2021): Machine learning to identify geologic factors associated with production in geothermal fields: A case-study using 3D geologic data, Brady geothermal field, Nevada, Geothermal Energy.


### Conference papers

- Mudunuru, M.K., Ahmmed, B., and Frash, L.: GeoThermalCloud for EGS -- An Open-source, User-friendly, Scalable AI Workflow for Modeling Enhanced Geothermal Systems, Geothermal Rising Conference, Reno, NV, October 1-5, 2023. 
- Mudunuru, M.K., Ahmmed, B., and Frash, L.: Deep Learning for Modeling Enhanced Geothermal Systems, 48th Annual Stanford Geothermal Workshop, Stanford, CA, February 6-8, 2023.  
- Frash, L. and Ahmmed, B.: A FORGE Datathon Case Study to Optimize Well Spacing and Flow Rate for Power Generation, 48th Annual Stanford Geothermal Workshop, Stanford, CA, February 6-8, 2023. 
- Frash, L., Carey, J.W., Ahmmed, B., and others: A Proposal for Safe and Profitable Enhanced Geothermal Systems in Hot Dry Rock, 48th Annual Stanford Geothermal Workshop}, Stanford, CA, February 6-8, 2023.  
- Ahmmed, B., Vesselinov, V.V., Mudunuru, M.K., and Frash, L.: A Progress Report on GeoThermalCloud Framework: An Open-source Machine Learning Based Tool for Discovery, Exploration, and Development of Hidden Geothermal Resources, 48th Annual Stanford Geothermal Workshop, Stanford, CA, February 6-8, 2023. 
- Ahmmed, B., Vesselinov, V.V., Rau, E., and Mudunuru, M.K., and Karra, S.: Machine Learning and a Process Model to Better Characterize Hidden Geothermal Resources, GRC Transactions, v. 46, Reno, NV, August 28-31, 2022. 
- Vesselinov, V.V., Ahmmed, B., Frash, L., and Mudunuru, M.K.: GeoThermalCloud: Machine Learning for Discovery, Exploration, and Development of Hidden Geothermal Resources, 47th Annual Stanford Geothermal Workshop, Stanford, CA, February 7-9, 2022. 
- Vesselinov, V.V., Frash, L., Ahmmed, B., and Mudunuru, M.K.: Machine Learning to Characterize the State of Stress and its Influence on Geothermal Production, Geothermal Rising Conference, San Diego, CA, October 3-6, 2021. 
- Ahmmed, B., Vesselinov, V.V.: Prospectivity Analyses of the Utah FORGE Site using Unsupervised Machine Learning, Geothermal Rising Conference, San Diego, CA, October 3-6, 2021. 
- Ahmmed, B., Vesselinov, V.V., Mudunuru, M.K., Middleton, R., and Karra, S.: Geochemical characteristics of Low-, Medium-, and Hot-temperature Geothermal Resources of the Great Basin, USA, World Geothermal Congress, Reykjavik, Iceland, May 21-26, 2021. 
- Vesselinov, V.V., Ahmmed, B., Mudunuru, M.K., Karra, S., and Middleton, R.: Hidden Geothermal Signatures of the Southwest New Mexico, World Geothermal Congress, Reykjavik, Iceland, May 21-26, 2021. 
- Mudunuru, M.K., Ahmmed, B., Vesselinov, V.V., Burns, E., Livingston, D.R., Karra, S., Middleton, R.S.: Machine Learning for Geothermal Resource Analysis and Exploration, XXIII International Conference on Computational Methods in Water Resources (CMWR), Stanford, CA, December 13-15, 2020, no. 81. 
- Mudunuru, M.K., Ahmmed, B., Karra S., Vesselinov, V.V., Livingston D.R., and Middleton R.S.: Site-scale and Regional-scale Modeling for Geothermal Resource Analysis and Exploration, 45th Annual Stanford Geothermal Workshop, Stanford, CA, February 10-12, 2020. 
- Vesselinov, V.V., Mudunuru, M.K., Ahmmed, B., Karra, S. and Middleton, R.S.: Discovering Signatures of Hidden Geothermal Resources Based on Unsupervised Learning, 45th Annual Stanford Geothermal Workshop, Stanford, CA, February 10-12, 2020. 
 

### Presentations

- Siler, D., Pepin, J., Vesselinov, V.V., Ahmmed, B., and Mudunuru, M.K.: A tale of two unsupervised machine learning techniques: What PCA and NMFk tell us about the geologic controls of hydrothermal processes, American Geophysical Union, New Orleans, LA,, December 13–17, 2021.
- Siler, D., Pepin, J., Vesselinov, V.V., Ahmmed, B., and Mudunuru, M.K.: A tale of two unsupervised machine learning techniques: What PCA and NMFk tell us about the geologic controls of hydrothermal processes, Geothermal Rising Conference, San Diego, CA, October 3-6, 2021.
- Ahmmed, B. Vesselinov, V. and Mudunuru, M.K., Integration of Data, Numerical Inversion,  and  Unsupervised Machine Learning to Identify Hidden Geothermal Resources in Southwest New Mexico, American Geophysical Union Fall Conference, San Francisco, CA, December 1-17, 2020.
- Ahmmed, B., Vesselinov, V.V., and Mudunuru, M.K., Machine learning to characterize regional geothermal reservoirs in the western USA, Abstract T185-358249, Geological Society of America, October 26-29, 2020.
- Ahmmed, B., Lautze, N., Vesselinov, V.V., Dores, D., and Mudunuru, M.K., Unsupervised Machine Learn- ing to Extract Dominant Geothermal Attributes in Hawaii Island Play Fairway Data, Geothermal Resources Council, Reno, NV, October 18-23, 2020.
- Vesselinov, V.V., Ahmmed, B., and Mudunuru, M.K., Unsupervised Machine Learning to discover attributes that characterize low, moderate, and high-temperature geothermal resources, Geothermal Resources Council, Reno, NV, October 18-23, 2020.
- Ahmmed, B., Vesselinov, V., and Mudunuru, M.K., Non-negative Matrix Factorization to Discover Dominant Attributes in Utah FORGE Data, Geothermal Resources Council, Reno, NV, October 18-23, 2020.
- Ahmmed, B., Vesselinov, V.V., and Mudunuru, M.K., Unsupervised machine learning to discover dominant attributes of mineral precipitation due to CO2 sequestration, LA-UR-20-20989, 3rd Machine Learning in Solid Earth Science Conference, Santa Fe, NM, March 16-20, 2020.


