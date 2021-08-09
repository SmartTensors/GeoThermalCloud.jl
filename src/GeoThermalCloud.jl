module GeoThermalCloud

import NMFk
import NTFk
import IJulia
import JLD
import JLD2
import XLSX
import DelimitedFiles
import SVR
import Statistics
import Clustering
import GMT
import Gadfly
import Mads
import Kriging
import Cairo
import Fontconfig

const dir = splitdir(splitdir(pathof(GeoThermalCloud))[1])[1]

function analysis(problem::AbstractString; notebook::Bool=false)
	@info("GeoThermalCloud: $problem analysis")
	if notebook
		IJulia.notebook(; dir=joinpath(dir, problem), detached=true)
	else
		c = pwd()
		cd(joinpath(dir, problem, "notebook"))
		Mads.runcmd("jupyter-nbconvert --to script $problem.ipynb")
		cd(joinpath(dir, problem))
		Base.include(Main, joinpath("notebook", "$(problem).jl"))
		cd(c)
	end
end

function notebooks()
	analysis("."; notebook=true)
end

function Brady(; kw...)
	analysis("Brady"; kw...)
end

function SWNM(; kw...)
	analysis("SWNM"; kw...)
end

function GreatBasin(; kw...)
	analysis("GreatBasin"; kw...)
end

end