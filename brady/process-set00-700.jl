depth = 700
casename = "set00-"

attributes_process = ["normal", "coulomb", "dilation", "faults", "td", "ts", "curve", "modeltemp", "faultdense", "faultintdense", "distfromcontacts", "distfromfaults", "unitthickness", "goodlith"]

include("load-v3.jl")
include("process.jl")