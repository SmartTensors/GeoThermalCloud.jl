import NMFk

X = reshape(Tn, ((depth * length(attributes_process))), length(locations))
Xu, nmin, nmax = NMFk.normalizematrix_row!(X) # this is wrong

figuredir = "figures-$(casename)$(depth)"
resultdir = "results-$(casename)$(depth)"
nkrange = 2:10
W, H, fitquality, robustness, aic = NMFk.execute(Xu, nkrange; resultdir=resultdir, load=true)
W, H, fitquality, robustness, aic = NMFk.load(nkrange, 10; resultdir=resultdir)

NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange]), W, H, attributes_process_long, locations; loadassignements=true, lon=xcoord, lat=ycoord, sizeW=depth, resultdir=resultdir, figuredir=figuredir, Htypes=welltype, hover="Well: " .* locations .* "<br>" .* "WellType: " .* String.(welltype) .* "<br>" .* production)