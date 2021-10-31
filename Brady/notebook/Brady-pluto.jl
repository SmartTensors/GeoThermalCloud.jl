### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 0db3b1d1-0f52-479b-883a-5d3f7e782875
md"""
Geothermal machine learning analysis: Brady site, Nevada
---

This notebook is a part of the GeoThermalCloud.jl: Machine Learning framework for Geothermal Exploration.

<div style="text-align: left; padding-bottom: 30px;">
    <img src="../../logos/geothermalcloud-small.png" alt="geothermalcloud" width=25%  max-width=125px;/>
</div>

Machine learning analyses are performed using the **SmartTensors** machine learning framework.

<div style="text-align: left; padding-bottom: 30px;">
	<img src="../../logos/SmartTensorsNewSmaller.png" alt="SmartTensors" width=25%  max-width=125px;/>
</div>

This notebook demonstrates how the **NMFk** module of **SmartTensors** can be applied to perform unsupervised geothermal machine-learning analyses.

<div style="text-align: left; padding-bottom: 30px;">
	<img src="../../logos/nmfk-logo.png" alt="nmfk" width=25%  max-width=125px;/>
</div>

More information on how the ML results are interpreted to provide geothermal insights is discussed in our research paper.
"""

# ╔═╡ edb15a6a-8e4d-4f17-a0cf-e1cc6b162aa7
md"""
## Import required Julia modules

If **NMFk** is not installed, first execute in the Julia REPL `import Pkg; Pkg.add("NMFk"); Pkg.add("DelimitedFiles"); Pkg.add("JLD"); Pkg.add("Gadfly"); Pkg.add("Cairo"); Pkg.add("Fontconfig"); Pkg.add("Mads")`.

"""

# ╔═╡ d5ec17cf-7b8f-40a5-a742-096c042ad38d
begin
	import NMFk
	import DelimitedFiles
	import JLD
	import Gadfly
	import Cairo
	import Fontconfig
	import Mads
end

# ╔═╡ 0ef2bd1b-c3b5-472c-8686-9fc099a5e73c
md"""
## Load and pre-process the dataset
"""

# ╔═╡ fdcb37ee-a8d4-4b34-af07-b8bb1b839d84
md"""
### Setup the working directory containing the Brady site data
"""

# ╔═╡ 018e6301-5ef2-4337-87ba-bce8395ec085
cd(joinpath(GeoThermalCloud.dir, "Brady");

# ╔═╡ 09575485-0e2b-4aac-9c01-0dfbf89f135d
md"""
### Load the data file
"""

# ╔═╡ 9dda8b4c-7a3c-4e91-ab7a-82700f0f0fb0
d, h = DelimitedFiles.readdlm("data/AllBradyWells_LANL_ML_9.txt", ','; header=true);

# ╔═╡ 4d67ff42-9679-4a91-a21e-09aa62785bc9
md"""
### Populate the missing well names
"""

# ╔═╡ 98b47fb8-6fe4-4800-9f5c-7bb34927bde4
begin
	global wellname = ""
	for i = 1:size(d, 1)
		if d[i, 1] != ""
			global wellname = d[i, 1]
		else
			d[i, 1] = wellname
		end
	end
end

# ╔═╡ 95896d0b-ea74-4cdb-9175-f7af1398bfe2
md"""
### Set up missing entries to be equal to zero
"""

# ╔═╡ 5fb84ce6-e6bb-4fb4-96bf-1e64e1245034
d[d[:, 24] .== "", 24] .= 0;

# ╔═╡ be7ef806-c398-47fb-a050-66f21af5c4ef
md"""
### Define names of the data attributes (matrix columns)
"""

# ╔═╡ edbdd788-a389-47ad-9791-356670c530c6
begin
	attributes_short = ["ID", "D", "azimuth", "incline", "x", "y", "z", "casing", "fluids", "use", "production", "use2", "lt750mstatus", "normal", "coulomb", "dilation", "faults", "td", "ts", "curve", "modeltemp", "faultdense", "faultintdense", "inv_distcontacts", "inv_distfaults", "unitthickness", "goodlith", "confidence"]
	attributes_long = ["ID", "Depth", "Azimuth", "Inclination", "X", "Y", "Z", "Casing", "Fluids", "use", "Production", "use2", "Status", "Normal stress", "Coulomb shear stress", "Dilation", "Faulting", "Fault dilation tendency", "Fault slip tendency", "Fault curvature", "Temperature", "Fault density", "Fault intersection density", "Inverse distance from contacts", "Inverse distance from faults", "Unit thickness", "Lithology", "Confidence"];
end

# ╔═╡ 32849a21-da8b-46be-9245-630cd14a266d
md"""
Short attribute names are used for coding.

Long attribute names are used for plotting and visualization.
"""

# ╔═╡ 451c10de-13e0-4d0d-9e0c-09a453bef419
md"""
### Define the attributes that will be processed
"""

# ╔═╡ b43fcf18-6335-4796-8e00-5e53c2c6b0b2
attributes_process = ["normal", "coulomb", "dilation", "faults", "td", "ts", "curve", "modeltemp", "faultdense", "faultintdense", "inv_distcontacts", "inv_distfaults", "unitthickness", "goodlith"];

# ╔═╡ 3122328e-22da-4ebb-94bb-d6fb3a2ad1b6
md"""
### Index the attributes that will be processed
"""

# ╔═╡ 7b0fbafc-62d6-482c-b786-ca504d529b54
begin
	ai = indexin(attributes_process, attributes_short)
	pr = indexin(["production"], attributes_short)
	attributes_process_long = attributes_long[ai]

	attributes_col = vec(permutedims(h))
	attributes = attributes_col[ai];
end

# ╔═╡ 8fe795fa-238e-43d2-8857-963fa6903025
md"""
### Display information about the processed data (min, max, count):
"""

# ╔═╡ 58fe35b3-5c73-4a46-b57d-57e52aaaf615
begin
	for i=1:length(attributes_col); println("$(attributes_col[i]): Column $i Min $(minimum(d[:,i])) Max $(maximum(d[:,i]))"); end
	for i=1:length(attributes_col); println("$(attributes_col[i]): Column $i Unique entries:"); display(unique(sort(d[:,i]))); end
end

# ╔═╡ e99c2d3b-796b-4f68-991e-891a8a46384d
md"""
### Get well locations and production
"""

# ╔═╡ b9f66361-c2ab-4373-9b64-c4c7d0637110
begin
	locations = unique(sort(d[:,1]))
	ii = convert.(Int64, round.(d[:,2]))
	zi = unique(sort(ii))

	xcoord = Vector{Float64}(undef, length(locations))
	ycoord = Vector{Float64}(undef, length(locations))
	production = Vector{String}(undef, length(locations))
	for (j, w) in enumerate(locations)
		iw = d[:, 1] .== w
		i = findmin(d[iw, 2])[2]
		xcoord[j] = d[iw, 5][i]
		ycoord[j] = d[iw, 6][i]
		production[j] = unique(d[iw, pr])[end]
	end
end

# ╔═╡ 9e057dfb-d69d-48da-b3d8-c7c37155250a
md"""
### Define well types
"""

# ╔═╡ 541ef1ac-56fa-434e-a396-fb25c500dbe1
begin
	welltype = Vector{Symbol}(undef, length(locations))
	for (j, w) in enumerate(locations)
		iw = d[:, 1] .== w
		welltype[j] = Symbol(unique(d[iw, indexin(["lt750mstatus"], attributes_short)])[1])
	end
end

# ╔═╡ 98b25ccb-a9ea-45c6-b8d8-030bc3666699
md"""
### Display information about processed well attributes
"""

# ╔═╡ 745790b3-ac35-4bc4-8b17-dfc11f398445
for i = ai
	println("$(attributes_col[i]): $i")
	display(unique(sort(convert.(Float64, d[:,i]))))
end

# ╔═╡ 6d1a9e89-e8ae-4269-98dd-bf891524fc57
md"""
### Collect the well data into a 3D tensor

Tensor indices (dimensions) define depths, attributes, and wells.
"""

# ╔═╡ 3cf50cee-5230-43a3-b644-9b2ce648d626
begin
	T = Array{Float64}(undef, length(zi), length(ai), length(locations))
	T .= NaN

	for w = 1:length(locations)
		iw = d[:, 1] .== locations[w]
		m = d[iw, ai]
		zw = ii[iw]
		for z = 1:length(zw)
			a = vec(m[z, :])
			s = length(a)
			if s == 0
				continue
			end
			T[zw[z] + 1, 1:s, w] .= a
		end
	end
end

# ╔═╡ 63c661d2-672e-4511-a74b-75c9d095ba4a
md"""
### Define the maximum depth

The maximum depth limits the depth of the data included in the analyses.

The maximum depth is set to 750 m.
"""

# ╔═╡ ff035352-09a8-48c7-a698-4476d54e31dd
depth = 750;

# ╔═╡ b428aa71-8a6d-47d3-acb3-285ba7f9ca01
md"""
### Normalize tensor slices associated with each attribute
"""

# ╔═╡ 19587614-bec0-4db8-ad54-7543e887c233
begin
	Tn = deepcopy(T[1:depth,:,:])
	for a = 1:length(ai)
		Tn[:,a,:], _, _ = NMFk.normalize(Tn[:,a,:])
	end
end

# ╔═╡ 8e3e5e39-f640-4d99-bb04-55061dccf82c
md"""
### Define problem setup variables
"""

# ╔═╡ 68d35398-b1a9-40dd-8f2e-851ea30d96ee
begin
	nruns = 1000 # number of random NMF runs
	nkrange = 2:8 # range of k values explored by the NMFk algorithm

	casename = "set00-v9-inv" # casename of the performed ML analyses
	figuredir = "figures-$(casename)-$(depth)" # directory to store figures associated with the performed ML analyses
	resultdir = "results-$(casename)-$(depth)"; # directory to store obtained results associated with the performed ML analyses
end

# ╔═╡ fa6a843c-102f-459f-b4d8-ca73d8ea5aa9
md"""
### Plot well data
"""

# ╔═╡ e400d796-a8c7-46f4-8b54-843778fc1a62
begin
	nlocations = length(locations)
	hovertext = Vector{String}(undef, nlocations)
	for i = 1:nlocations
		hovertext[i] = join(map(j->("$(attributes_process_long[j]): $(round(float.(NMFk.meannan(T[:,j,i])); sigdigits=3))<br>"), 1:length(attributes_process_long)))
	end

	NMFk.plot_wells("map/dataset-$(casename).html", xcoord, ycoord, String.(welltype); hover=locations .* "<br>" .* String.(welltype) .* "<br>" .* production .* "<br>" .* hovertext, title="Brady site: Data")
end

# ╔═╡ 3358ad10-d9fe-47f3-b4f8-06b05d0eeb41
md"""
A HTML file named [../map/dataset-set00-v9-inv.html](../map/dataset-set00-v9-inv.html) is generated mapping the site data.
The map provides interactive visualization of the site data (it can also be opened with any browser).

The map below shows the location of the `Dry`, `Injection` and `Production` wells.

![dataset-set01-v9-inv](../map/dataset-set01-v9-inv.png)
"""

# ╔═╡ 29456432-b54b-40b2-aa02-94df5a5bc806
md"""
## Perform ML analyses

For the ML analyses, the data tensor can be flattened into a data matrix by using two different approaches:

- Type 1: Merge depth and attribute tensor dimensions; in this way, the focus of the ML analysis is on finding the features associated with the well locations
- Type 2: Merge depth and location tensor dimensions; in this way, the focus of the ML analysis is on finding the features associated with the well attributes

After that the **NMFk** algorithm will factorize the data matrix `X` into `W` and `H` matrices. For more information, check out the [**NMFk** website](https://github.com/SmartTensors/NMFk.jl)
"""

# ╔═╡ 271043f3-a183-4680-96d2-aff5bf0259a9
md"""
### Type 1 flattening: Focus on well locations

#### Flatten the tensor into a matrix
"""

# ╔═╡ 42744bd4-1e79-4b2f-a313-56991a2b3563
Xdaln = reshape(Tn, (depth * length(attributes_process)), length(locations));

# ╔═╡ e64df2f4-2503-4fc7-8070-5b59813463f3
md"""
Matrix rows merge the depth and attribute dimensions.

Matrix columns represent the well locations.
"""

# ╔═╡ f0ef0ba5-42f4-47c8-8d30-58f1c55d3cb4
md"""
#### Perform NMFk analyses
"""

# ╔═╡ 66662a04-377c-4bde-b591-eb075edbb973
md"""
Here, the **NMFk** results are loaded from a prior ML run.

As seen from the output above, the **NMFk** analyses identified that the optimal number of geothermal signatures in the dataset **6**.

Solutions with a number of signatures less than **6** are underfitting.

Solutions with a number of signatures greater than **6** are overfitting and unacceptable.

The set of acceptable solutions are defined by the **NMFk** algorithm as follows:
"""

# ╔═╡ 4c8e3e57-d04a-47a6-b257-901748e70b01
md"""
The acceptable solutions contain 2, 5 and 6 signatures.
"""

# ╔═╡ 368755f9-a828-4ae4-a831-1d2b7f92b48a
md"""
#### Post-process NMFk results

#### Number of signatures

Below is a plot representing solution quality (fit) and silhouette width (robustness) for different numbers of signatures `k`:
"""

# ╔═╡ 66a333fa-f19c-4925-8725-c067fb492e23
md"""
The plot above also demonstrates that the acceptable solutions contain 2, 5 and 6 signatures.

#### Analysis of all the acceptable solutions

The ML solutions containing an acceptable number of signatures are further analyzed as follows:
"""

# ╔═╡ 9484a968-b844-4a0b-9693-6aa2277be1bd
md"""
The results for a solution with **6** signatures presented above will be further discussed here.

The well attributes are clustered into **6** groups:
"""

# ╔═╡ b8cb4713-d81e-4a9a-9e7f-daf8aa734f4c
Mads.display("results-set00-v9-inv-750-1000-daln/attributes-6-groups.txt")

# ╔═╡ 012ceac3-6260-485d-999d-5ed6d9171277
md"""


<div style="background-color: gray;">
  <p>
    <iframe src="../results-set00-v9-inv-750-1000-daln/attributes-6-groups.txt" frameborder="0" height="400"
      width="95%"></iframe>
  </p>
</div>

This grouping is based on analyses of the attribute matrix `W`:

![attributes-6-labeled-sorted](../figures-set00-v9-inv-750-1000-daln/attributes-6-labeled-sorted.png)

Note that the attribute matrix `W` is automatically modified to account that a range of vertical depths is applied in characterizing the site wells.

The well locations are also clustered into **6** groups:

<div style="background-color: gray;">
  <p>
    <iframe src="../results-set00-v9-inv-750-1000-daln/locations-6-groups.txt" frameborder="0" height="400"
      width="95%"></iframe>
  </p>
</div>

This grouping is based on analyses of the location matrix `H`:

![locations-6-labeled-sorted](../figures-set00-v9-inv-750-1000-daln/locations-6-labeled-sorted.png)

The map [../figures-set00-v9-inv-750-1000-daln/locations-6-map.html](../figures-set00-v9-inv-750-1000-daln/locations-6-map.html) provides interactive visualization of the extracted well location groups (the html file can also be opened with any browser).

<div>
    <iframe src="../figures-set00-v9-inv-750-1000-daln/locations-6-map.html" frameborder="0" height="400" width="50%"></iframe>
</div>

More information on how the ML results are interpreted to provide geothermal insights is discussed in our research paper.
"""

# ╔═╡ 616b52bc-aba7-4fea-be6b-2a81a8824898
md"""
### Type 2 flattening: Focus on well attributes

#### Flatten the tensor into a matrix
"""

# ╔═╡ 3d4fbbdc-1d30-4ca7-9a3c-fa0f8f3ef8f2
Xdlan = reshape(permutedims(Tn, (1,3,2)), (depth * length(locations)), length(attributes_process));

# ╔═╡ f5328a14-8340-4b99-a282-cb32f5b83350
md"""
Matrix rows merge the depth and well locations dimensions.

Matrix columns represent the well attributes.
"""

# ╔═╡ 1b74cd58-8724-4d5d-95fb-c73f3ccbff98
md"""
#### Perform NMFk analyses
"""

# ╔═╡ c4f06da6-b5dc-4aaa-8c66-42f553bb5aa3
NMFk.getks(nkrange, robustness[nkrange])

# ╔═╡ 7057ece6-010f-4eb2-be74-10417169964f
NMFk.plot_signal_selecton(nkrange, fitquality, robustness; figuredir="$figuredir-$(nruns)-daln", xtitle="Number of signatures")



# ╔═╡ 9c43cb87-7687-4a97-a3e6-01e4be43e725
NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange]), W, H, attributes_process_long, locations; loadassignements=true, lon=xcoord, lat=ycoord, Wsize=depth, Wcasefilename="attributes", Hcasefilename="locations", resultdir=resultdir * "-$(nruns)-daln", figuredir=figuredir * "-$(nruns)-daln", hover="Well: " .* locations .* "<br>" .* "WellType: " .* String.(welltype) .* "<br>" .* production, Wmatrix_font_size=4Gadfly.pt, biplotcolor=:WH, biplotlabel=:WH)

# ╔═╡ 52324ce4-a350-4eea-88b0-544492c216a0
md"""
Here the **NMFk** results are loaded from a prior ML run.

As seen from the output above, the **NMFk** analyses identified that the optimal number of geothermal signatures in the dataset **3**.

Solutions with a number of signatures less than **3** are underfitting.

Solutions with a number of signatures greater than **3** are overfitting and unacceptable.

The set of acceptable solutions are defined by the **NMFk** algorithm as follows:
"""

# ╔═╡ 6eb7f0be-bf2f-48d5-8579-985ed357b844
NMFk.getks(nkrange, robustness[nkrange])

# ╔═╡ 32921f33-3dad-487d-ada9-4c686489b4ea
md"""
The acceptable solutions contain 2 and 3 signatures.
"""

# ╔═╡ a5f8e4af-1b19-41dd-aa9d-c9e16351f358
md"""
#### Post-process NMFk results

#### Number of signatures

Below is a plot representing solution quality (fit) and silhouette width (robustness) for different numbers of signatures `k`:
"""

# ╔═╡ 6eef4ff9-c52a-4d55-9a65-16cb46d56159
NMFk.plot_signal_selecton(nkrange, fitquality, robustness; figuredir="$figuredir-$(nruns)-dlan", xtitle="Number of signatures")

# ╔═╡ b0b39853-e392-438a-951e-075959ba05f9
md"""
The plot above also demonstrates that the acceptable solutions contain 2 and 3 signatures.

#### Analysis of all the acceptable solutions

The ML solutions containing an acceptable number of signatures are further analyzed as follows:
"""

# ╔═╡ 4fa33fb9-f201-431d-b69d-1c6e3c75a77f
NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange]), W, H, locations, attributes_process_long; loadassignements=true, lon=xcoord, lat=ycoord, Wsize=depth, Wcasefilename="locations", Hcasefilename="attributes", resultdir=resultdir * "-$(nruns)-dlan", figuredir=figuredir * "-$(nruns)-dlan", hover="Well: " .* locations .* "<br>" .* "WellType: " .* String.(welltype) .* "<br>" .* production, Wmatrix_font_size=4Gadfly.pt, biplotcolor=:WH, biplotlabel=:WH)

# ╔═╡ 49ceb8a8-61fb-441d-bd61-a3404444addf
md"""
#### Analysis of the 3-signature solution

The results for a solution with **3** signatures presented above will be further discussed here.

The well attributes are clustered into **3** groups:

<div style="background-color: gray;">
    <iframe src="../results-set00-v9-inv-750-1000-dlan/attributes-3-groups.txt" frameborder="0" height="400"
      width="95%"></iframe>
</div>

This grouping is based on analyses of the attribute matrix `W`:

![attributes-3-labeled-sorted](../figures-set00-v9-inv-750-1000-dlan/attributes-3-labeled-sorted.png)

Note that the attribute matrix `W` is automatically modified to account that a range of vertical depths is applied in characterizing the site wells.

The well locations are also clustered into **3** groups:

<div style="background-color: gray;">
  <p>
    <iframe src="../results-set00-v9-inv-750-1000-dlan/locations-3-groups.txt" frameborder="0" height="400"
      width="95%"></iframe>
  </p>
</div>

This grouping is based on analyses of the location matrix `H`:

![locations-3-labeled-sorted](../figures-set00-v9-inv-750-1000-dlan/locations-3-labeled-sorted.png)

The map [../figures-set00-v9-inv-750-1000-dlan/locations-3-map.html](../figures-set00-v9-inv-750-1000-dlan/locations-3-map.html) provides interactive visualization of the extracted well location groups (the html file can also be opened with any browser).

<div>
    <iframe src="../figures-set00-v9-inv-750-1000-dlan/locations-3-map.html" frameborder="0" height="400" width="50%"></iframe>
</div>
"""

# ╔═╡ b9e7a872-27a3-4f2a-89cc-6bbd878c3a9d
begin
	W, H, fitquality, robustness, aic = NMFk.execute(Xdlan, nkrange, nruns; resultdir=resultdir, casefilename="nmfk-dlan-$(join(size(Xdlan), '_'))", load=true)
	W, H, fitquality, robustness, aic = NMFk.load(nkrange, nruns; resultdir=resultdir, casefilename="nmfk-dlan-$(join(size(Xdlan), '_'))");
end

# ╔═╡ b462744e-ec46-44a7-b639-89525221ea01
begin
	W, H, fitquality, robustness, aic = NMFk.execute(Xdaln, nkrange, nruns; resultdir=resultdir, casefilename="nmfk-daln-$(join(size(Xdaln), '_'))", load=true)
	W, H, fitquality, robustness, aic = NMFk.load(nkrange, nruns; resultdir=resultdir, casefilename="nmfk-daln-$(join(size(Xdaln), '_'))");
end

# ╔═╡ Cell order:
# ╠═0db3b1d1-0f52-479b-883a-5d3f7e782875
# ╟─edb15a6a-8e4d-4f17-a0cf-e1cc6b162aa7
# ╠═d5ec17cf-7b8f-40a5-a742-096c042ad38d
# ╟─0ef2bd1b-c3b5-472c-8686-9fc099a5e73c
# ╟─fdcb37ee-a8d4-4b34-af07-b8bb1b839d84
# ╠═018e6301-5ef2-4337-87ba-bce8395ec085
# ╟─09575485-0e2b-4aac-9c01-0dfbf89f135d
# ╠═9dda8b4c-7a3c-4e91-ab7a-82700f0f0fb0
# ╟─4d67ff42-9679-4a91-a21e-09aa62785bc9
# ╠═98b47fb8-6fe4-4800-9f5c-7bb34927bde4
# ╟─95896d0b-ea74-4cdb-9175-f7af1398bfe2
# ╠═5fb84ce6-e6bb-4fb4-96bf-1e64e1245034
# ╟─be7ef806-c398-47fb-a050-66f21af5c4ef
# ╠═edbdd788-a389-47ad-9791-356670c530c6
# ╟─32849a21-da8b-46be-9245-630cd14a266d
# ╟─451c10de-13e0-4d0d-9e0c-09a453bef419
# ╠═b43fcf18-6335-4796-8e00-5e53c2c6b0b2
# ╟─3122328e-22da-4ebb-94bb-d6fb3a2ad1b6
# ╠═7b0fbafc-62d6-482c-b786-ca504d529b54
# ╟─8fe795fa-238e-43d2-8857-963fa6903025
# ╠═58fe35b3-5c73-4a46-b57d-57e52aaaf615
# ╟─e99c2d3b-796b-4f68-991e-891a8a46384d
# ╠═b9f66361-c2ab-4373-9b64-c4c7d0637110
# ╟─9e057dfb-d69d-48da-b3d8-c7c37155250a
# ╠═541ef1ac-56fa-434e-a396-fb25c500dbe1
# ╟─98b25ccb-a9ea-45c6-b8d8-030bc3666699
# ╠═745790b3-ac35-4bc4-8b17-dfc11f398445
# ╟─6d1a9e89-e8ae-4269-98dd-bf891524fc57
# ╠═3cf50cee-5230-43a3-b644-9b2ce648d626
# ╟─63c661d2-672e-4511-a74b-75c9d095ba4a
# ╠═ff035352-09a8-48c7-a698-4476d54e31dd
# ╟─b428aa71-8a6d-47d3-acb3-285ba7f9ca01
# ╠═19587614-bec0-4db8-ad54-7543e887c233
# ╟─8e3e5e39-f640-4d99-bb04-55061dccf82c
# ╠═68d35398-b1a9-40dd-8f2e-851ea30d96ee
# ╟─fa6a843c-102f-459f-b4d8-ca73d8ea5aa9
# ╠═e400d796-a8c7-46f4-8b54-843778fc1a62
# ╟─3358ad10-d9fe-47f3-b4f8-06b05d0eeb41
# ╟─29456432-b54b-40b2-aa02-94df5a5bc806
# ╟─271043f3-a183-4680-96d2-aff5bf0259a9
# ╠═42744bd4-1e79-4b2f-a313-56991a2b3563
# ╟─e64df2f4-2503-4fc7-8070-5b59813463f3
# ╟─f0ef0ba5-42f4-47c8-8d30-58f1c55d3cb4
# ╠═b462744e-ec46-44a7-b639-89525221ea01
# ╟─66662a04-377c-4bde-b591-eb075edbb973
# ╠═c4f06da6-b5dc-4aaa-8c66-42f553bb5aa3
# ╟─4c8e3e57-d04a-47a6-b257-901748e70b01
# ╟─368755f9-a828-4ae4-a831-1d2b7f92b48a
# ╠═7057ece6-010f-4eb2-be74-10417169964f
# ╟─66a333fa-f19c-4925-8725-c067fb492e23
# ╠═9c43cb87-7687-4a97-a3e6-01e4be43e725
# ╟─9484a968-b844-4a0b-9693-6aa2277be1bd
# ╠═b8cb4713-d81e-4a9a-9e7f-daf8aa734f4c
# ╟─012ceac3-6260-485d-999d-5ed6d9171277
# ╟─616b52bc-aba7-4fea-be6b-2a81a8824898
# ╠═3d4fbbdc-1d30-4ca7-9a3c-fa0f8f3ef8f2
# ╟─f5328a14-8340-4b99-a282-cb32f5b83350
# ╟─1b74cd58-8724-4d5d-95fb-c73f3ccbff98
# ╠═b9e7a872-27a3-4f2a-89cc-6bbd878c3a9d
# ╟─52324ce4-a350-4eea-88b0-544492c216a0
# ╠═6eb7f0be-bf2f-48d5-8579-985ed357b844
# ╟─32921f33-3dad-487d-ada9-4c686489b4ea
# ╟─a5f8e4af-1b19-41dd-aa9d-c9e16351f358
# ╠═6eef4ff9-c52a-4d55-9a65-16cb46d56159
# ╟─b0b39853-e392-438a-951e-075959ba05f9
# ╠═4fa33fb9-f201-431d-b69d-1c6e3c75a77f
# ╟─49ceb8a8-61fb-441d-bd61-a3404444addf
