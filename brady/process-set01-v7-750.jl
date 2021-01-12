depth = 750
casename = "set01-v7-inv"
nruns = 1000
plotresults = isdefined(Main, :Cairo)

attributes_process = ["normal", "coulomb", "dilation", "faults", "td", "ts", "curve", "modeltemp", "faultdense", "faultintdense", "inv_distfaults"]

include("load-v7.jl")
include("process-new.jl")