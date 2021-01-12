depth = 607
casename = "withconfidence-"

attributes_process = ["faults", "distfromfault", "distfromcontact", "td", "ts", "curve", "temp", "ints", "lithgoodbad", "liththickness", "goodliththickness", "faultsingoodlith", "confidence", "Dilation", "Coulomb", "Normal"]

include("load.jl")
include("process.jl")