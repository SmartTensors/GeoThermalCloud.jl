function gmtplot_greatbasin(X, title, filename, range=(0,0.5,0.1))
	GMT.grdimage("data/greatbasin-v3.nc", proj=:Mercator, shade=(azimuth=100, norm="e0.8"), color=GMT.makecpt(color=:grayC, transparency=10, range=(0,5000,500), continuous=true), figsize=8, conf=(MAP_FRAME_TYPE="plain", MAP_GRID_PEN_PRIMARY="thinnest,gray,.", MAP_GRID_CROSS_SIZE_SECONDARY=0.1, MAP_FRAME_PEN=0.5, MAP_TICK_PEN_PRIMARY=0.1, MAP_TICK_LENGTH_PRIMARY=0.01, FORMAT_GEO_MAP="ddd", FONT_ANNOT_PRIMARY=0.1, FONT_ANNOT_SECONDARY=0.1), frame=(axis="lrtb"))
	cgyr = GMT.makecpt(C=(:green,:yellow,:red), range=range, continuous=true, conf=(COLOR_BACKGROUND=:green, COLOR_FOREGROUND=:red))
	GMT.plot!(X; marker=:s, markersize=0.1, color=cgyr, coast=(proj=:Mercator, DCW=(country="US.UT,US.NV,US.CA,US.AZ,US.OR,US.ID", pen=(0.5,:black))))
	GMT.colorbar!(pos=(inside=true, anchor=:BL, length=(1.5,0.2), vertical=true, offset=(0.2,0.3)), box=(pen=0.1, fill=:white), color=cgyr, par=(:FONT_ANNOT_PRIMARY, "6p"), conf=(MAP_FRAME_TYPE="plain", MAP_GRID_PEN_PRIMARY="thinnest,gray,.", MAP_GRID_CROSS_SIZE_SECONDARY=0.1, MAP_FRAME_PEN=0.1, MAP_TICK_PEN_PRIMARY=0.1, MAP_TICK_LENGTH_PRIMARY=0.01,FONT_ANNOT_PRIMARY=4, FONT_ANNOT_SECONDARY=4))
	GMT.legend!(box=(pen=0.1, fill=:white),
	pos=(inside=true, anchor=:TC, width=1.5, justify=:CM, offset=(-0.4, -0.3)),
	GMT.text_record(["S 0.00001i c 0.0001i white 0.05p 0.05i $(title)"]),
	par=(:FONT_ANNOT_PRIMARY, "6p"),
	fmt=:png, savefig=filename, show=true)
end

function gmtplot_greatbasin_label(X, filename, labelx, labely, labels, range=(0,0.5,0.1), grd="data/greatbasin-v3.nc", range_topo=(0,5000,500))
	GMT.grdimage(grd, proj=:Mercator, shade=(azimuth=100, norm="e0.8"), color=GMT.makecpt(color=:grayC, transparency=10, range=range_topo, continuous=true), figsize=8, conf=(MAP_FRAME_TYPE="plain", MAP_GRID_PEN_PRIMARY="thinnest,gray,.", MAP_GRID_CROSS_SIZE_SECONDARY=0.1, MAP_FRAME_PEN=0.5, MAP_TICK_PEN_PRIMARY=0.1, MAP_TICK_LENGTH_PRIMARY=0.01, FORMAT_GEO_MAP="ddd", FONT_ANNOT_PRIMARY=0.1, FONT_ANNOT_SECONDARY=0.1), frame=(axis="lrtb"))
	cgyr = GMT.makecpt(C=(:green,:yellow,:red), range=range, continuous=true, conf=(COLOR_BACKGROUND=:green, COLOR_FOREGROUND=:red))
	GMT.plot!(X; marker=:s, markersize=0.1, color=cgyr, coast=(proj=:Mercator, DCW=(country="US.UT,US.NV,US.CA,US.AZ,US.OR,US.ID", pen=(0.5,:black))))
	
	GMT.plot!(labelx, labely, marker=:c, markersize=0.1)
	label_txt = GMT.text_record([labelx labely], labels);
	GMT.text!(label_txt, font=5, justify=:CT)

	GMT.colorbar!(pos=(inside=true, anchor=:BL, length=(1.5,0.2), vertical=true, offset=(0.2,0.3)), box=(pen=0.1, fill=:white), color=cgyr, par=(:FONT_ANNOT_PRIMARY, "6p"), conf=(MAP_FRAME_TYPE="plain", MAP_GRID_PEN_PRIMARY="thinnest,gray,.", MAP_GRID_CROSS_SIZE_SECONDARY=0.1, MAP_FRAME_PEN=0.1, MAP_TICK_PEN_PRIMARY=0.1, MAP_TICK_LENGTH_PRIMARY=0.01,FONT_ANNOT_PRIMARY=4, FONT_ANNOT_SECONDARY=4), fmt=:png, savefig=filename, show=true)
end

function gmtplot_greatbasin_nolabel(X, filename, labelx, labely, range=(0,0.5,0.1), grd="data/greatbasin-v3.nc", range_topo=(0,5000,500))
	GMT.grdimage(grd, proj=:Mercator, shade=(azimuth=100, norm="e0.8"), color=GMT.makecpt(color=:grayC, transparency=10, range=range_topo, continuous=true), figsize=8, conf=(MAP_FRAME_TYPE="plain", MAP_GRID_PEN_PRIMARY="thinnest,gray,.", MAP_GRID_CROSS_SIZE_SECONDARY=0.1, MAP_FRAME_PEN=0.5, MAP_TICK_PEN_PRIMARY=0.1, MAP_TICK_LENGTH_PRIMARY=0.01, FORMAT_GEO_MAP="ddd", FONT_ANNOT_PRIMARY=0.1, FONT_ANNOT_SECONDARY=0.1), frame=(axis="lrtb"))
	cgyr = GMT.makecpt(C=(:green,:yellow,:red), range=range, continuous=true, conf=(COLOR_BACKGROUND=:green, COLOR_FOREGROUND=:red))
	GMT.plot!(X; marker=:s, markersize=0.1, color=cgyr, coast=(proj=:Mercator, DCW=(country="US.UT,US.NV,US.CA,US.AZ,US.OR,US.ID", pen=(0.5,:black))))
	GMT.plot!(labelx, labely, marker=:c, markersize=0.1)
	GMT.colorbar!(pos=(inside=true, anchor=:BL, length=(1.5,0.2), vertical=true, offset=(0.2,0.3)), box=(pen=0.1, fill=:white), color=cgyr, par=(:FONT_ANNOT_PRIMARY, "6p"), conf=(MAP_FRAME_TYPE="plain", MAP_GRID_PEN_PRIMARY="thinnest,gray,.", MAP_GRID_CROSS_SIZE_SECONDARY=0.1, MAP_FRAME_PEN=0.1, MAP_TICK_PEN_PRIMARY=0.1, MAP_TICK_LENGTH_PRIMARY=0.01,FONT_ANNOT_PRIMARY=4, FONT_ANNOT_SECONDARY=4), fmt=:png, savefig=filename, show=true)
end
