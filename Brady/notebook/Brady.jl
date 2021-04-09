
import GeoThermalCloud
import NMFk
import Mads
import DelimitedFiles
import JLD
import Gadfly
import Cairo
import Fontconfig

cd(joinpath(GeoThermalCloud.dir, "Brady"));

d, h = DelimitedFiles.readdlm("data/AllBradyWells_LANL_ML_9.txt", ','; header=true);

global wellname = ""
for i = 1:size(d, 1)
	if d[i, 1] != ""
		global wellname = d[i, 1]
	else
		d[i, 1] = wellname
	end
end

d[d[:, 24] .== "", 24] .= 0;

attributes_short = ["ID", "D", "azimuth", "incline", "x", "y", "z", "casing", "fluids", "use", "production", "use2", "lt750mstatus", "normal", "coulomb", "dilation", "faults", "td", "ts", "curve", "modeltemp", "faultdense", "faultintdense", "inv_distcontacts", "inv_distfaults", "unitthickness", "goodlith", "confidence"]
attributes_long = ["ID", "Depth", "Azimuth", "Inclination", "X", "Y", "Z", "Casing", "Fluids", "use", "Production", "use2", "Status", "Normal stress", "Coulomb shear stress", "Dilation", "Faulting", "Fault dilation tendency", "Fault slip tendency", "Fault curvature", "Temperature", "Fault density", "Fault intersection density", "Inverse distance from contacts", "Inverse distance from faults", "Unit thickness", "Lithology", "Confidence"];

attributes_process = ["normal", "coulomb", "dilation", "faults", "td", "ts", "curve", "modeltemp", "faultdense", "faultintdense", "inv_distcontacts", "inv_distfaults", "unitthickness", "goodlith"];

ai = indexin(attributes_process, attributes_short)
pr = indexin(["production"], attributes_short)
attributes_process_long = attributes_long[ai]

attributes_col = vec(permutedims(h))
attributes = attributes_col[ai];

for i=1:length(attributes_col); println("$(attributes_col[i]): Column $i Min $(minimum(d[:,i])) Max $(maximum(d[:,i]))"); end
for i=1:length(attributes_col); println("$(attributes_col[i]): Column $i Unique entries:"); display(unique(sort(d[:,i]))); end

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

welltype = Vector{Symbol}(undef, length(locations))
for (j, w) in enumerate(locations)
	iw = d[:, 1] .== w
	welltype[j] = Symbol(unique(d[iw, indexin(["lt750mstatus"], attributes_short)])[1])
end

for i = ai
	println("$(attributes_col[i]): $i")
	display(unique(sort(convert.(Float64, d[:,i]))))
end

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

depth = 750;

Tn = deepcopy(T[1:depth,:,:])
for a = 1:length(ai)
	Tn[:,a,:], _, _ = NMFk.normalize!(Tn[:,a,:])
end

nruns = 1000 # number of random NMF runs
nkrange = 2:8 # range of k values explored by the NMFk algorithm

casename = "set00-v9-inv" # casename of the performed ML analyses
figuredir = "figures-$(casename)-$(depth)" # directory to store figures associated with the performed ML analyses
resultdir = "results-$(casename)-$(depth)"; # directory to store obtained results associated with the performed ML analyses

nlocations = length(locations)
hovertext = Vector{String}(undef, nlocations)
for i = 1:nlocations
	hovertext[i] = join(map(j->("$(attributes_process_long[j]): $(round(float.(NMFk.meannan(T[:,j,i])); sigdigits=3))<br>"), 1:length(attributes_process_long)))
end

NMFk.plot_wells("map/dataset-$(casename).html", xcoord, ycoord, String.(welltype); hover=locations .* "<br>" .* String.(welltype) .* "<br>" .* production .* "<br>" .* hovertext, title="Brady site: Data")

Xdaln = reshape(Tn, (depth * length(attributes_process)), length(locations));

W, H, fitquality, robustness, aic = NMFk.execute(Xdaln, nkrange, nruns; resultdir=resultdir, casefilename="nmfk-daln-$(join(size(Xdaln), '_'))", load=true)
W, H, fitquality, robustness, aic = NMFk.load(nkrange, nruns; resultdir=resultdir, casefilename="nmfk-daln-$(join(size(Xdaln), '_'))");

NMFk.getks(nkrange, robustness[nkrange])

NMFk.plot_signal_selecton(nkrange, fitquality, robustness; figuredir="$figuredir-$(nruns)-daln", xtitle="Number of signatures")



NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange]), W, H, attributes_process_long, locations; loadassignements=true, lon=xcoord, lat=ycoord, Wsize=depth, Wcasefilename="attributes", Hcasefilename="locations", resultdir=resultdir * "-$(nruns)-daln", figuredir=figuredir * "-$(nruns)-daln", hover="Well: " .* locations .* "<br>" .* "WellType: " .* String.(welltype) .* "<br>" .* production, Wmatrix_font_size=4Gadfly.pt, biplotcolor=:WH, biplotlabel=:WH)

Mads.display("results-set00-v9-inv-750-1000-daln/attributes-6-groups.txt")

Xdlan = reshape(permutedims(Tn, (1,3,2)), (depth * length(locations)), length(attributes_process));

W, H, fitquality, robustness, aic = NMFk.execute(Xdlan, nkrange, nruns; resultdir=resultdir, casefilename="nmfk-dlan-$(join(size(Xdlan), '_'))", load=true)
W, H, fitquality, robustness, aic = NMFk.load(nkrange, nruns; resultdir=resultdir, casefilename="nmfk-dlan-$(join(size(Xdlan), '_'))");

NMFk.getks(nkrange, robustness[nkrange])

NMFk.plot_signal_selecton(nkrange, fitquality, robustness; figuredir="$figuredir-$(nruns)-dlan", xtitle="Number of signatures")

NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange]), W, H, locations, attributes_process_long; loadassignements=true, lon=xcoord, lat=ycoord, Wsize=depth, Wcasefilename="locations", Hcasefilename="attributes", resultdir=resultdir * "-$(nruns)-dlan", figuredir=figuredir * "-$(nruns)-dlan", hover="Well: " .* locations .* "<br>" .* "WellType: " .* String.(welltype) .* "<br>" .* production, Wmatrix_font_size=4Gadfly.pt, biplotcolor=:WH, biplotlabel=:WH)
