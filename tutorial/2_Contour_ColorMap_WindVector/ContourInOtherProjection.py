import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import oyl

ds = xr.open_dataset("../datas/era5uvz500.nc")
##选择经纬度区域
ds = ds.sel(latitude=slice(60,0), longitude=slice(70, 140))
##除以98把位势变成位势高度（单位为位势什米
z = ds.z[0]/98

##定义一个ds数据集描述的地图对象
##投影方式为兰伯特投影，中心点(110E,45N)
##x轴设置用xticks， 设置海岸线细一点，颜色用灰色
m = oyl.ncmap(ds, coast_color='grey', coast_linewidth=0.8,xticks=range(80,130,20),
              projection=ccrs.EquidistantConic(110, 45))
##加载省份，线条用红色，线细一点，颜色透明一点
m.load_province(edgecolor='red', linewidth=0.8, alpha=0.3)

##填色位势高度,色标放下面
##其余如levels, cmap, extend,zorder, vmin, norm等参数见plt.contourf
m.contourf(z, location='bottom', zorder=0, levels=range(560, 601, 4),
           cmap=plt.cm.BrBG_r, alpha=0.9, extend='both')
##画等值线，除clabel参数(是否打标注)独有，其余同plt.contour
m.contour(z, zorder=0, colors='k', levels=range(560, 601, 4))
m.show()

