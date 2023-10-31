import Gadfly, Compose, Cairo, Fontconfig
import Mads
import NMFk
import Kriging
import DelimitedFiles
import JLD
import GMT

include("gmtplot_greatbasin.jl")
fauld_pfa_map_points, hs = DelimitedFiles.readdlm("data/jimfauldsPFAdata.csv", ',', header=true)
hot_springs, hhs = DelimitedFiles.readdlm("data/hot_springs_great_basin.csv", ',', header=true)
hot_springs_coords = convert.(Float32, hot_springs[:,2:3])

@JLD.load "data/Xnl.jld" Xnl xlmin xlmax zflag coords attributes_all
npoints = size(coords, 1)
xcoords = coords[:,1];
ycoords = coords[:,2];
xgrid, ygrid = NMFk.griddata(xcoords, ycoords; stepvalue=0.1)

nkrange = 2:6
nruns = 100
resultdir = "results-nl-$(nruns)"
resultdirpost = "results-postprocessing-nl-$(nruns)"
figuredirpost = "figures-postprocessing-nl-$(nruns)"
mapdirpost = "maps-postprocessing-nl-$(nruns)"

W, H, fitquality, robustness, aic = NMFk.execute(Xnl, nkrange, nruns; resultdir=resultdir, casefilename="nmfk-nl", load=true)
W, H, fitquality, robustness, aic = NMFk.load(nkrange, nruns; resultdir=resultdir, casefilename="nmfk-nl")
NMFk.plot_signal_selecton(nkrange, fitquality, robustness; xtitle="Number of signatures", ytitle="Normalized performance metrics", figuredir=figuredirpost)

Sorder, Wclusters, Hclusters = NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange], 0.5), W, H, string.(collect(1:npoints)), attributes_all; lon=xcoords, lat=ycoords, resultdir=resultdirpost, figuredir=figuredirpost, ordersignal=:Wcount, Hcasefilename="attributes", Wcasefilename="locations", biplotcolor=:WH, sortmag=false, biplotlabel=:H, point_size_nolabel=2Gadfly.pt, point_size_label=4Gadfly.pt)

for k in Sorder[end]
	l = 'A' + k - 1
	inversedistancevector = Vector{Float64}(undef, 0)
	xvector = Vector{Float64}(undef, 0)
	yvector = Vector{Float64}(undef, 0)
	sigselection=size(Sorder[end],1)
	v = W[sigselection][:,k]
	v ./= maximum(v)
	for (i, x) in enumerate(xgrid), (j, y) in enumerate(ygrid)
		inversedistance = Kriging.inversedistance(permutedims([x y]), coord, v, 2; cutoff=500)[1]
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
	sigselection=size(Sorder[end],1)
	v = W[sigselection][:,k]
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
	#gmtplot_greatbasin_label([xvector yvector inversedistancevector], "$(mapdirpost)/signatures-pfa-comparison-6-label-$(l)", hot_springs_coords[:,2], hot_springs_coords[:,1], string.(hot_springs[:,1]))
	gmtplot_greatbasin_nolabel([xvector yvector inversedistancevector], "$(mapdirpost)/signatures-pfa-comparison-6-no-spring-label-$(l)", hot_springs_coords[:,2], hot_springs_coords[:,1])
end

nkrange = 2:10
nruns = 100
resultdir = "results-nl-$(nruns)"
resultdirpost = "results-postprocessing-nl-$(nruns)"
figuredirpost = "figures-postprocessing-nl-$(nruns)"
mapdirpost = "maps-postprocessing-nl-$(nruns)"

W, H, fitquality, robustness, aic = NMFk.execute(Xnl, nkrange, nruns; resultdir=resultdir, casefilename="nmfk-nl-10", load=true)
W, H, fitquality, robustness, aic = NMFk.load(nkrange, nruns; resultdir=resultdir, casefilename="nmfk-nl-10")
NMFk.plot_signal_selecton(nkrange, fitquality, robustness; xtitle="Number of signatures", ytitle="Normalized performance metrics", figuredir=figuredirpost)

Sorder, Wclusters, Hclusters = NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange], 0.5), W, H, string.(collect(1:npoints)), attributes_all; lon=xcoords, lat=ycoords, resultdir=resultdirpost, figuredir=figuredirpost, ordersignal=:Wcount, Hcasefilename="attributes", Wcasefilename="locations", biplotcolor=:WH, sortmag=false, biplotlabel=:H, point_size_nolabel=2Gadfly.pt, point_size_label=4Gadfly.pt)

for k in Sorder[end]
	l = 'A' + k - 1
	inversedistancevector = Vector{Float64}(undef, 0)
	xvector = Vector{Float64}(undef, 0)
	yvector = Vector{Float64}(undef, 0)
	sigselection=size(Sorder[end],1)
	v = W[sigselection][:,k]
	v ./= maximum(v)
	for (i, x) in enumerate(xgrid), (j, y) in enumerate(ygrid)
		inversedistance = Kriging.inversedistance(permutedims([x y]), coord, v, 2; cutoff=500)[1]
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
	sigselection=size(Sorder[end],1)
	v = W[sigselection][:,k]
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
	#gmtplot_greatbasin_label([xvector yvector inversedistancevector], "$(mapdirpost)/signatures-pfa-comparison-6-label-$(l)", hot_springs_coords[:,2], hot_springs_coords[:,1], string.(hot_springs[:,1]))
	gmtplot_greatbasin_nolabel([xvector yvector inversedistancevector], "$(mapdirpost)/signatures-pfa-comparison-6-no-spring-label-$(l)", hot_springs_coords[:,2], hot_springs_coords[:,1])
end

nkrange = 2:6
nruns = 100
resultdir = "results-no-gwt-nl-$(nruns)"
resultdirpost = "results-postprocessing-nl-$(nruns)-no-gwt"
figuredirpost = "figures-postprocessing-nl-$(nruns)-no-gwt"
mapdirpost = "maps-postprocessing-nl-$(nruns)-no-gwt"

W, H, fitquality, robustness, aic = NMFk.execute(Xnl[:,2:end], nkrange, nruns; resultdir=resultdir, casefilename="nmfk-nl", load=true)
W, H, fitquality, robustness, aic = NMFk.load(nkrange, nruns; resultdir=resultdir, casefilename="nmfk-nl")
NMFk.plot_signal_selecton(nkrange, fitquality, robustness; xtitle="Number of signatures", ytitle="Normalized performance metrics", figuredir=figuredirpost)

Sorder, Wclusters, Hclusters = NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange], 0.5), W, H, string.(collect(1:npoints)), attributes_all[2:end]; lon=xcoords, lat=ycoords, resultdir=resultdirpost, figuredir=figuredirpost, ordersignal=:Wcount, Hcasefilename="attributes", Wcasefilename="locations", biplotcolor=:WH, sortmag=false, biplotlabel=:H, point_size_nolabel=2Gadfly.pt, point_size_label=4Gadfly.pt)

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
	v = W[size(Sorder[end],1)][:,k]
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
	gmtplot_greatbasin_nolabel([xvector yvector inversedistancevector], "$(mapdirpost)/signatures-pfa-comparison-3nolabel-$(l)")
end


attributes_geop = ["Temperature [C]", "GTM Quartz [C]", "GTM Chalcedony [C]", "Heat flow [W/m²]", "Two m temperature [C]", "Depth to the basement [m]", "Gravity anomaly [mGal]", "Magnetic anomaly [nT]", "Strain rate [nS⁻¹]", "Dilation rate [nS⁻¹]", "Shear rate [nS⁻¹]"]
att_indcs = indexin(attributes_geop, attributes_all)

nkrange = 2:6
nruns = 100
resultdir = "results-geop-nl-$(nruns)"
resultdirpost = "results-postprocessing-nl-$(nruns)-geop"
figuredirpost = "figures-postprocessing-nl-$(nruns)-geop"
mapdirpost = "maps-postprocessing-nl-$(nruns)-geop"

W, H, fitquality, robustness, aic = NMFk.execute(Xnl[:,att_indcs], nkrange, nruns; resultdir=resultdir, casefilename="nmfk-nl", load=true)
W, H, fitquality, robustness, aic = NMFk.load(nkrange, nruns; resultdir=resultdir, casefilename="nmfk-nl")
NMFk.plot_signal_selecton(nkrange, fitquality, robustness; xtitle="Number of signatures", ytitle="Normalized performance metrics", figuredir=figuredirpost)

Sorder, Wclusters, Hclusters = NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange], 0.5), W, H, string.(collect(1:npoints)), attributes_all[att_indcs]; lon=xcoords, lat=ycoords, resultdir=resultdirpost, figuredir=figuredirpost, ordersignal=:Wcount, Hcasefilename="attributes", Wcasefilename="locations", biplotcolor=:WH, sortmag=false, biplotlabel=:H, point_size_nolabel=2Gadfly.pt, point_size_label=4Gadfly.pt)

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
	v = W[size(Sorder[end],1)][:,k]
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
	gmtplot_greatbasin_nolabel([xvector yvector inversedistancevector], "$(mapdirpost)/signatures-pfa-comparison-3nolabel-$(l)")
end





