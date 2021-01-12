import DelimitedFiles
import NMFk

attributes_short = ["wellid", "md", "azimuth", "inclination", "x", "y", "z", "casing", "fluids", "use", "production", "use2", "lt700mstatus", "normal", "coulomb", "dilation", "faults", "td", "ts", "curve", "modeltemp", "faultdense", "faultintdense", "distfromcontacts", "distfromfaults", "unitthickness", "goodlith", "confidence"]
ai = indexin(attributes_process, attributes_short)

d, h = DelimitedFiles.readdlm("data/AllBradyWells_LANL_ML_3.txt", ','; header=true)

global wellname = ""
for i = 1:size(d, 1)
	if d[i, 1] != ""
		global wellname = d[i, 1]
	else
		d[i, 1] = wellname
	end
end

d[d[:,24] .== "", 24] .= 0

attributes_col = vec(permutedims(h))
attributes = attributes_col[ai]

for i=1:length(attributes_col); @info attributes_col[i], i; display([minimum(d[:,i]); maximum(d[:,i])]); end
for i=1:length(attributes_col); @info attributes_col[i], i; display(unique(sort(d[:,i]))); end

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
	welltype[j] = Symbol(unique(d[iw, indexin(["lt700mstatus"], attributes_short)])[1])
end

for i=ai; @info attributes_col[i], i; display(unique(sort(convert.(Float64, d[:,i])))); end

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

# for a = 1:length(ai)
# 	display(findlast(sum(.!isnan.(T[:,a,:]); dims=2) .== 51)[1])
# end

# for a = 1:length(ai)
# 	display(findlast(sum(.!isnan.(T[:,a,:]); dims=2) .== 25)[1])
# end

Tn = deepcopy(T[1:depth,:,:])
for a = 1:length(ai)
	# display(NMFk.plotmatrix(T[1:607,a,:]))
	Tn[:,a,:], _, _ = NMFk.normalize!(Tn[:,a,:])
	# display(NMFk.plotmatrix(Tn[:,a,:]))
end