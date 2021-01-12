depth = 700
casename = "set01-"

attributes_process = ["normal", "coulomb", "dilation", "faults", "td", "ts", "curve", "modeltemp", "faultdense", "faultintdense", "distfromfaults"]

include("load-v3.jl")
include("process.jl")