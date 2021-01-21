import NMFk

W, H, fitquality, robustness, aic = NMFk.execute(Xu, nkrange, 1000; resultdir=resultdir, load=true)
W, H, fitquality, robustness, aic = NMFk.load(nkrange, 1000; resultdir=resultdir)

NMFk.plot_signal_selecton(nkrange, fitquality, robustness; figuredir=figuredir)
NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange]), W, H, attributes, string.(collect(1:size(locations, 1))); lat=lat, lon=lon, resultdir=resultdir, figuredir=figuredir, Hcasefilename="locations", Wcasefilename="attributes")