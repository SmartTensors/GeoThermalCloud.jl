include("load.jl")

attributes_remove = uppercasefirst.(lowercase.(["Air Temperature"; "Subcrop Permeability"; "Watertable Elevation"; "Groundsurface Elevation"; "Watertable Depth"]))

figuredir = "figures-case01"
resultdir = "results-case01"

include("preprocess.jl")

nkrange=2:15

include("process.jl")

order = collect(indexin(attributes_ordered, uppercasefirst.(lowercase.(attributes))))

so, wl, hl = NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange]), W, H, attributes, string.(collect(1:size(locations, 1))); Worder=order, lat=lat, lon=lon, resultdir=resultdir, figuredir=figuredir, Hcasefilename="locations", Wcasefilename="attributes", loadassignements=false, createbiplots=false)