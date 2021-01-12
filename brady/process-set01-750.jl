depth = 750
casename = "set01-v5-"

attributes_process = ["normal", "coulomb", "dilation", "faults", "td", "ts", "curve", "modeltemp", "faultdense", "faultintdense", "distfromfaults"]

include("load-v5.jl")
include("process.jl")