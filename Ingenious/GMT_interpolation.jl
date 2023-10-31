import NMFk
import JLD
import DelimitedFiles
import SVR
import Kriging
import Gadfly
import GMT

include("gmtplot_greatbasin.jl")

fauld_pfa_map_points, hs = DelimitedFiles.readdlm("data/jimfauldsPFAdata.csv", ',', header=true)

attributes = ["Temperature", "Quartz", "Chalcedony", "pH", "TDS", "Al", "B", "Ba", "Be", "Br", "Ca", "Cl", "HCO3", "K", "Li", "Mg", "Na", "delO18"]

resultdir = "results"
nkrange = 2:10
nruns = 640

W, H, fitquality, robustness, aic = NMFk.execute(Xnl, nkrange, nruns; resultdir=resultdir, casefilename="nmfk-nl", load=true)

resultdirpost = "results-postprocessing-nl-$(nruns)"
figuredirpost = "figures-postprocessing-nl-$(nruns)"
mapdirpost = "map-postprocessing-nl-$(nruns)"
W, H, fitquality, robustness, aic = NMFk.load(nkrange, nruns; cutoff=0.4, resultdir=resultdir, casefilename="nmfk-nl")
NMFk.plot_feature_selecton(nkrange, fitquality, robustness; figuredir=figuredirpost)
Sorder, Wclusters, Hclusters = NMFk.clusterresults(NMFk.getk(nkrange, robustness[nkrange], 0.4), W, H, string.(collect(1:npoints)), attributes; lon=xcoord, lat=ycoord, resultdir=resultdirpost, figuredir=figuredirpost, ordersignal=:Wcount, Hcasefilename="attributes", Wcasefilename="locations", biplotcolor=:WH, sortmag=false, biplotlabel=:H, point_size_nolabel=2Gadfly.pt, point_size_label=4Gadfly.pt, createplots=false)

coord = permutedims([xcoord ycoord])

xgrid, ygrid = NMFk.griddata(xcoord, ycoord; stepvalue=0.1)

for k in Sorder[end]
	l = 'A' + k - 1
	inversedistancevector = Vector{Float64}(undef, 0)
	xvector = Vector{Float64}(undef, 0)
	yvector = Vector{Float64}(undef, 0)
	v = W[3][:,k]
	v ./= maximum(v)
	for (i, x) in enumerate(xgrid), (j, y) in enumerate(ygrid)
		inversedistance = Kriging.inversedistance(permutedims([x y]), coord, v, 2; cutoff=1000)[1]
		if !isnan(inversedistance)
			push!(xvector, x)
			push!(yvector, y)
			push!(inversedistancevector, inversedistance)
		end
	end
	i5 = inversedistancevector .> 0.5
	inversedistancevector[i5] .= 0.5
	gmtplot_greatbasin([xvector yvector inversedistancevector], "Signature $(l)", "$(mapdirpost)/signatures-3-$(l)")
end

# Plots signature and Jim Faulds' PFA critical points
pfa_labels = ["1","2","3.1","4","5.1","6.1","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25"]
for k in Sorder[end]
	l = 'A' + k - 1
	inversedistancevector = Vector{Float64}(undef, 0)
	xvector = Vector{Float64}(undef, 0)
	yvector = Vector{Float64}(undef, 0)
	v = W[3][:,k]
	v ./= maximum(v)
	for (i, x) in enumerate(xgrid), (j, y) in enumerate(ygrid)
		inversedistance = Kriging.inversedistance(permutedims([x y]), coord, v, 2; cutoff=1000)[1]
		if !isnan(inversedistance)
			push!(xvector, x)
			push!(yvector, y)
			push!(inversedistancevector, inversedistance)
		end
	end
	i5 = inversedistancevector .> 0.5
	inversedistancevector[i5] .= 0.5
	#gmtplot_greatbasin_v2([xvector yvector inversedistancevector], "PFA and Signature $(l)", "$(mapdirpost)/signatures-pfa-comparison-3nolabel-$(l)")
	gmtplot_greatbasin_v3([xvector yvector inversedistancevector], "$(mapdirpost)/signatures-pfa-comparison-3nolabel-$(l)")
end

for k in Sorder[end]
	l = 'A' + k - 1
	inversedistancefield = Array{Float64}(undef, length(xgrid), length(ygrid))
	v = W[3][:,k]
	v ./= maximum(v)
	for (i, x) in enumerate(xgrid), (j, y) in enumerate(ygrid)
		inversedistancefield[i, j] = Kriging.inversedistance(permutedims([x y]), coord, v, 2; cutoff=1000)[1]
	end
	NMFk.plotmatrix(rotl90(inversedistancefield); quiet=false, filename="$(figuredirpost)/Signature_$(l)_map_inversedistance.png", maxvalue=0.5, title="Signature $(l)")
end

attributes_plot_max = Vector{Float32}(undef, nattributes)
attributes_plot_min = Vector{Float32}(undef, nattributes)
for i = 1:nattributes
	v = X[:,i]
	iz = .!isnan.(v)
	icoord = coord[:,iz]
	v = v[iz]
	inversedistancevector = Vector{Float64}(undef, 0)
	xvector = Vector{Float64}(undef, 0)
	yvector = Vector{Float64}(undef, 0)
	for (i, x) in enumerate(xgrid), (j, y) in enumerate(ygrid)
		inversedistance = Kriging.inversedistance(permutedims([x y]), icoord, v, 2; cutoff=1000)[1]
		if !isnan(inversedistance)
			push!(xvector, x)
			push!(yvector, y)
			push!(inversedistancevector, inversedistance)
		end
	end
	if i == 18
		inversedistancevector .-= 20
	end
	attributes_plot_max[i] = NMFk.maximumnan(inversedistancevector)
	attributes_plot_min[i] = NMFk.minimumnan(inversedistancevector)
	maxvalue = attributes_plot_min[i] + (attributes_plot_max[i] - attributes_plot_min[i])/ 2
	minvalue = attributes_plot_min[i]
	stepvalue = (attributes_plot_max[i] - attributes_plot_min[i]) / 5
	i5 = inversedistancevector .> maxvalue
	inversedistancevector[i5] .= maxvalue
	if attributes_plot_min[i] < 0
		i5 = inversedistancevector .< attributes_plot_min[i]
		inversedistancevector[i5] .= attributes_plot_min[i]
	else
		minvalue = attributes_plot_min[i]
	end
	gmtplot_greatbasin([xvector yvector inversedistancevector], attributes[i], "maps-data/Attribute_$(attributes[i])_map_inversedistance", (minvalue,maxvalue,stepvalue))
end

# VERY SLOW
# for k in Sorder[end]
# 	l = 'A' + k - 1
# 	inversedistancefield = Array{Float64}(undef, length(xgrid), length(ygrid))
# 	v = W[3][:,k]
# 	v ./= maximum(v)
# 	krige_est = Array{Float64}(undef, length(xgrid), length(ygrid))
# 	krige_var = Array{Float64}(undef, length(xgrid), length(ygrid))
# 	for (i, x) in enumerate(xgrid), (j, y) in enumerate(ygrid)
# 		krige_est[i, j], krige_var[i, j] = Kriging.krige(permutedims([x y]), coord, v, covfun)
# 	end
# 	NMFk.plotmatrix(rotl90(inversedistancefield); quiet=false, filename="$(figuredirpost)/Signature_$(l)_map_kriging.png", maxvalue=0.5, title="Signature $(l)")
# end

Xe =  W[3] * H[3]
Xedn = NMFk.denormalizematrix_col(Xe, xlmin, xlmax; logv=logv, zflag=zflag)

for i = 1:nattributes
	v = Xedn[:,i]
	iz = .!isnan.(v)
	icoord = coord[:,iz]
	v = v[iz]
	inversedistancevector = Vector{Float64}(undef, 0)
	xvector = Vector{Float64}(undef, 0)
	yvector = Vector{Float64}(undef, 0)
	for (i, x) in enumerate(xgrid), (j, y) in enumerate(ygrid)
		inversedistance = Kriging.inversedistance(permutedims([x y]), icoord, v, 2; cutoff=1000)[1]
		if !isnan(inversedistance)
			push!(xvector, x)
			push!(yvector, y)
			push!(inversedistancevector, inversedistance)
		end
	end
	if i == 18
		inversedistancevector .-= 20
	end
	maxvalue = attributes_plot_min[i] + (attributes_plot_max[i] - attributes_plot_min[i])/ 2
	minvalue = attributes_plot_min[i]
	stepvalue = (attributes_plot_max[i] - attributes_plot_min[i]) / 5
	i5 = inversedistancevector .> maxvalue
	inversedistancevector[i5] .= maxvalue
	if attributes_plot_min[i] < 0
		i5 = inversedistancevector .< attributes_plot_min[i]
		inversedistancevector[i5] .= attributes_plot_min[i]
	else
		minvalue = attributes_plot_min[i]
	end
	imax = NMFk.maximumnan(inversedistancevector)
	imin = NMFk.minimumnan(inversedistancevector)
	@info("$(attributes[i]): Max: $imax $(attributes_plot_max[i]) Min: $imin $(attributes_plot_min[i])")
	gmtplot_greatbasin([xvector yvector inversedistancevector], attributes[i], "maps-postprocessing-nl-$(nruns)/Attribute_$(attributes[i])_map_inversedistance", (minvalue,maxvalue,stepvalue))
end

for i = 1:nattributes
	v = Xvar[:,i]
	iz = .!isnan.(v)
	icoord = coord[:,iz]
	v = v[iz]
	inversedistancevector = Vector{Float64}(undef, 0)
	xvector = Vector{Float64}(undef, 0)
	yvector = Vector{Float64}(undef, 0)
	for (i, x) in enumerate(xgrid), (j, y) in enumerate(ygrid)
		inversedistance = Kriging.inversedistance(permutedims([x y]), icoord, v, 2; cutoff=1000)[1]
		if !isnan(inversedistance)
			push!(xvector, x)
			push!(yvector, y)
			push!(inversedistancevector, inversedistance)
		end
	end
	plot_max = NMFk.maximumnan(inversedistancevector)
	plot_min = NMFk.minimumnan(inversedistancevector)
	maxvalue = plot_min + (plot_max - plot_min)/ 2
	minvalue = plot_min
	stepvalue = (plot_max - plot_min) / 5
	i5 = inversedistancevector .> maxvalue
	inversedistancevector[i5] .= maxvalue
	gmtplot_greatbasin([xvector yvector inversedistancevector], attributes[i], "maps-postprocessing-nl-$(nruns)/Attribute_$(attributes[i])_var_s4_map_inversedistance", (minvalue,maxvalue,stepvalue))
end

for k = 2:4
	Xmin, Xmax, Xvar = NMFk.uncertaintyranges(X, k, nruns; resultdir=resultdir, casefilename="nmfk-nl-uncert")
	i = nattributes
	v = Xvar[:,i]
	iz = .!isnan.(v)
	icoord = coord[:,iz]
	v = v[iz]
	inversedistancevector = Vector{Float64}(undef, 0)
	xvector = Vector{Float64}(undef, 0)
	yvector = Vector{Float64}(undef, 0)
	for (i, x) in enumerate(xgrid), (j, y) in enumerate(ygrid)
		inversedistance = Kriging.inversedistance(permutedims([x y]), icoord, v, 2; cutoff=1000)[1]
		if !isnan(inversedistance)
			push!(xvector, x)
			push!(yvector, y)
			push!(inversedistancevector, inversedistance)
		end
	end
	maxvalue = 0.1
	minvalue = 0
	stepvalue = (maxvalue - minvalue) / 5
	i5 = inversedistancevector .> maxvalue
	inversedistancevector[i5] .= maxvalue
	gmtplot_greatbasin([xvector yvector inversedistancevector], attributes[i], "maps-postprocessing-nl-$(nruns)/Attribute_$(attributes[i])_var_s$(k)_map_inversedistance_range", (minvalue,maxvalue,stepvalue))
end