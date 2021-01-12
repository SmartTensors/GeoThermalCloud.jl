depth = 750
casename = "set01-v6-inv-"

attributes_process = ["normal", "coulomb", "dilation", "faults", "td", "ts", "curve", "modeltemp", "faultdense", "faultintdense", "inv_distfaults"]

include("load-v6.jl")
include("process-new.jl")