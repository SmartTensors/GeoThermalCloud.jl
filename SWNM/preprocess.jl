index_remove = indexin(attributes_remove, attributes_long)
attributes_cols = trues(length(attributes_long))
attributes_cols[index_remove] .= false
cols = vec(4:26)[attributes_cols]
attributes = attributes_long[cols .- 3]

# mask_locations_pred = indexin(locations, locations_pred) .!= nothing
# mask_attribute_pred = indexin(attributes, ["Heat flow"]) .!= nothing

X = permutedims(d[dindex, cols])

Xu, nmin, nmax = NMFk.normalizematrix_row!(X)

nkrange = 2:10

:preprocessed