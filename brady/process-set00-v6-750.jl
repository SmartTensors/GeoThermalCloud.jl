depth = 750
casename = "set00-v6-inv-"

attributes_process = ["normal", "coulomb", "dilation", "faults", "td", "ts", "curve", "modeltemp", "faultdense", "faultintdense", "inv_distcontacts", "inv_distfaults", "unitthickness", "goodlith"]

include("load-v6.jl")
include("process-new.jl")