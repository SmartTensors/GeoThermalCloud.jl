depth = 700
casename = ""

attributes_process = ["faults", "distfromfault", "distfromcontact", "td", "ts", "curve", "temp", "ints", "lithgoodbad", "liththickness", "goodliththickness", "faultsingoodlith", "Dilation", "Coulomb", "Normal"]

include("load.jl")
include("process.jl")