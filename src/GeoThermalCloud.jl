module GeoThermalCloud

import NMFk
import NTFk
import IJulia
import JLD
import JLD2
import DelimitedFiles
import SVR
import Statistics
import Clustering
import Mads
import Kriging
import Cairo
import Fontconfig

dir = dirname(Base.source_path())
dir = dirname(dir)

function analysis(problem::AbstractString; notebook::Bool=false)
	@info("GeoThermalCloud: $problem analysis")
	if notebook
		IJulia.notebook(; dir=joinpath(dir, problem, "notebook"), detached=true)
	else
		cd(joinpath(dir, problem))
		include(joinpath("notebook", "$(problem).jl"))
	end
end

function Brady(; kw...)
	analysis("Brady"; kw...)
end

function SWNM(; notebook::Bool=false)
	analysis("SWNM"; kw...)
end

function GreatBasin(; notebook::Bool=false)
	analysis("GreatBasin"; kw...)
end

end