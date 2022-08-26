import matplotlib.pyplot as plt
import xarray as xr
import oyl

##读取nc数据
ds = xr.open_dataset("../datas/era5uvz500.nc")
##除以98把位势变成位势高度（单位为位势什米
z = ds.z[0]/98

##定义一个ds数据集描述的地图对象
m = oyl.ncmap(ds, coast_color='grey', coast_linewidth=0.8)
##加载中国, 细的红虚线
m.load_china(linewidth=0.8,edgecolor='red',linestyle='--')
##定义指定层次的位势高度
levels = list(range(460, 581, 20)) + [584, 588]
"""
填色位势高度函数contourf
m.contourf特有参数：
cbar:是否画色标，默认为True
location:色标位置，默认为'right',可选为'bottom'

plt.contourf参数：
zorder:高度位置（表征画图顺序，越小越先画
levels:画指定的等值线
cmap:填色的颜色体系
alpha:透明度（0到1)，越小越透明
extend:填色的延伸方向
其余如vmin, norm等参数见plt.contourf
"""
m.contourf(z, location='bottom', zorder=0, levels=levels,
           cmap=plt.cm.BrBG_r, alpha=0.9, extend='both')
##画等值线，除clabel参数(是否打标注)独有，其余同plt.contour
m.contour(z, zorder=0, colors='k', levels=levels)
m.show()

