depth = 750
casename = "set00-v5-"

attributes_process = ["normal", "coulomb", "dilation", "faults", "td", "ts", "curve", "modeltemp", "faultdense", "faultintdense", "distfromcontacts", "distfromfaults", "unitthickness", "goodlith"]

include("load-v5.jl")
include("process.jl")