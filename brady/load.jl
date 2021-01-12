import DelimitedFiles

attributes_short = ["wellid", "md", "x", "y", "z", "inclination", "azimuth", "casing", "fluids", "use", "use2", "production", "faults", "distfromfault", "distfromcontact", "td", "ts", "curve", "temp", "ints", "lithgoodbad", "lithname", "liththickness", "goodliththickness", "faultsingoodlith", "confidence", "Dilation", "Coulomb", "Normal"]

d, h = DelimitedFiles.readdlm("data/AllBradyWellsData.txt", ','; header=true)

attributes_col = vec(permutedims(h))

for i=1:29; display([h[:,i]; minimum(d[:,i]); maximum(d[:,i])]); end
for i=1:29; display([h[:,i]; unique(sort(d[:,i]))]); end

locations = unique(sort(d[:,1]))
ii = convert.(Int64, round.(d[:,2]))
zi = unique(sort(ii))
ai = indexin(attributes_process, attributes_short)

xcoord = Vector{Float64}(undef, length(locations))
ycoord = Vector{Float64}(undef, length(locations))
for (j, w) in enumerate(locations)
	iw = d[:, 1] .== w
	i = findmin(d[iw, 2])[2]
	xcoord[j] = d[iw, 4][i]
	ycoord[j] = d[iw, 3][i]
end

for i=ai; display([h[:,i]; unique(sort(d[:,i]))]); end

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