import NMFk

W, H, fitquality, robustness, aic = NMFk.execute(Xu, nkrange, 1000; resultdir=resultdir, load=true)
W, H, fitquality, robustness, aic = NMFk.load(nkrange, 1000; resultdir=resultdir)

NMFk.plot_signal_selecton(nkrange, fitquality, robustness; figuredir=figuredir)