# GTcloud.jl: Geothermal Cloud for Machine Learning

- Brady: geothermal exploration of Brady site, NV; paper under review: Machine learning to identify geologic factors associated with production in geothermal fields: A casestudy using 3D geologic data, Brady geothermal field, Nevada. This paper is under review in the Journal of Geothermal Energy.
- SWNM: geothermal exploration of southwest New Mexico. We have submitted a manuscript to Geothermics based on this work.

# Julia and NMFk installation

- Julia: https://julialang.org/downloads/
- NMFk: https://github.com/TensorDecompositions/NMFk.jl

## NMFk is hosted on Los Alamos National Laboratory github account, so it may require proxies. Please implement the following commands for proxies:

ENV["ftp_proxy"] =  "http://proxyout.lanl.gov:8080"  
ENV["rsync_proxy"] = "http://proxyout.lanl.gov:8080"  
ENV["http_proxy"] = "http://proxyout.lanl.gov:8080"  
ENV["https_proxy"] = "http://proxyout.lanl.gov:8080"  
ENV["no_proxy"] = ".lanl.gov"

