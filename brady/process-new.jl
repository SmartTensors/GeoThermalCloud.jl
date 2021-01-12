import NMFk
import Gadfly

nlocations = length(locations)
hovertext = Vector{String}(undef, nlocations)
for i = 1:nlocations
	hovertext[i] = join(map(j->("$(attributes_process_long[j]): $(round(float.(NMFk.meannan(T[:,j,i])); sigdigits=3))<br>"), 1:length(attributes_process_long)))
end

NMFk.plot_wells("map/dataset-$(casename).html", xcoord, ycoord, String.(welltype); hover=locations .* "<br>" .* String.(welltype) .* "<br>" .* production .* "<br>" .* hovertext, title="Brady site: Data")

nkrange = 2:8
figuredir = "figures-$(casename)-$(depth)"
resultdir = "results-$(casename)-$(depth)"

Xdaln = reshape(Tn, ((depth * length(attributes_process))), length(locations))
W, H, fitquality, robustness, aic = NMFk.execute(Xdaln, nkrange, nruns; resultdir=resultdir, casefilename="nmfk-daln-$(join(size(Xdaln), '_'))", load=true)
W, H, fitquality, robustness, aic = NMFk.load(nkrange, nruns; resultdir=resultdir, casefilename="nmfk-daln-$(join(size(Xdaln), '_'))")
if plotresults
	NMFk.plot_signal_selecton(nkrange, fitquality, robustness; figuredir="$figuredir-$(nruns)-daln")
	NMFk.clusterresults(unique(sort(vcat([4], NMFk.getks(nkrange, robustness[nkrange])))), W, H, attributes_process_long, locations; loadassignements=true, lon=xcoord, lat=ycoord, Wsize=depth, Wcasefilename="attributes", Hcasefilename="locations", resultdir=resultdir * "-$(nruns)-daln", figuredir=figuredir * "-$(nruns)-daln", hover="Well: " .* locations .* "<br>" .* "WellType: " .* String.(welltype) .* "<br>" .* production, Wmatrix_font_size=4Gadfly.pt, biplotcolor=:WH, biplotlabel=:WH)
end

# for k in NMFk.getks(nkrange, robustness[nkrange])[end]
# 	Xe = reshape(W[k] * H[k], (depth, length(attributes_process), length(locations)))
# 	Wa = reshape(NMFk.normalizematrix_col!(W[k])[1], (depth, length(attributes_process), k))
# 	for a = 1:length(attributes_process)
# 		Mads.plotseries(Tn[:,a,:]; title=attributes_process[a])
# 		Mads.plotseries(Wa[:,a,:]; title=attributes_process[a])
# 		Mads.plotseries(Xe[:,a,:]; title=attributes_process[a])
# 	end
# end

Xdlan = reshape(permutedims(Tn, (1,3,2)), ((depth * length(locations))), length(attributes_process))
W, H, fitquality, robustness, aic = NMFk.execute(Xdlan, nkrange, nruns; resultdir=resultdir, casefilename="nmfk-dlan-$(join(size(Xdlan), '_'))", load=true)
W, H, fitquality, robustness, aic = NMFk.load(nkrange, nruns; resultdir=resultdir, casefilename="nmfk-dlan-$(join(size(Xdlan), '_'))")
if plotresults
	NMFk.plot_signal_selecton(nkrange, fitquality, robustness; figuredir="$figuredir-$(nruns)-dlan")
	NMFk.clusterresults(unique(sort(vcat([4], NMFk.getks(nkrange, robustness[nkrange])))), W, H, locations, attributes_process_long; loadassignements=true, lon=xcoord, lat=ycoord, Wsize=depth, Wcasefilename="locations", Hcasefilename="attributes", resultdir=resultdir * "-$(nruns)-dlan", figuredir=figuredir * "-$(nruns)-dlan", hover="Well: " .* locations .* "<br>" .* "WellType: " .* String.(welltype) .* "<br>" .* production, Wmatrix_font_size=4Gadfly.pt, biplotcolor=:WH, biplotlabel=:WH)
end

# for w = 1:length(locations)
# 	Mads.plotseries(Tn[:,:,w]; names=attributes_process, title=locations[w])
# end

# for k in NMFk.getks(nkrange, robustness[nkrange])[end]
# 	Xe = reshape(W[k] * H[k], (depth, length(attributes_process), length(locations)))
# 	Wa = reshape(NMFk.normalizematrix_col!(W[k])[1], (depth, length(locations), k))
# 	for w = 1:length(locations)
# 		Mads.plotseries(Tn[:,:,w]; names=attributes_process, title=locations[w])
# 		Mads.plotseries(Wa[:,w,:]; title=locations[w])
# 		Mads.plotseries(Xe[:,:,w]; names=attributes_process, title=locations[w])
# 	end
# end