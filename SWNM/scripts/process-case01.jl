import NMFk

include("load.jl")

attributes_remove = uppercasefirst.(lowercase.(["Air Temperature"; "Subcrop Permeability"; "Watertable Elevation"; "Groundsurface Elevation"; "Watertable Depth"]))

figuredir = "figures-case01"
resultdir = "results-case01"

include("preprocess.jl")

nkrange=2:10

include("process.jl")

order = collect(indexin(attributes_ordered, uppercasefirst.(lowercase.(attributes))))

source_order, wmatrix_labels, hmatrix_labels = NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange], 0.2), W, H, attributes, locations_short[rows]; Worder=order, lat=lat, lon=lon, resultdir=resultdir, figuredir=figuredir, Hcasefilename="locations", Wcasefilename="attributes")

source_order, wmatrix_labels, hmatrix_labels = NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange], 0.2), W, H, attributes, locations_short[rows]; Worder=order, lat=lat, lon=lon, resultdir=resultdir, figuredir=figuredir, Hcasefilename="locations", Wcasefilename="attributes", loadassignements=false, createbiplots=false)