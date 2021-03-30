module GeoThermalCloud

import NMFk
import JLD
import JLD2
import DelimitedFiles
import SVR
import Statistics
import Clustering
import Mads
import Kriging

dir = dirname(Base.source_path())

function Brady()
	cd(joinpath(dir, Brady))
	include(joinpath(notebook, Brady.jl))
end

function SWNM()
	cd(joinpath(dir, SWNM))
	include(joinpath(notebook, SWNM.jl))
end

function GreatBasin()
	cd(joinpath(dir, GreatBasin))
	include(joinpath(notebook, GreatBasin.jl))
end

end