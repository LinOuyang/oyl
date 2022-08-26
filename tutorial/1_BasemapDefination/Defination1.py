import cartopy.crs as ccrs
import oyl

##定义一个经纬度为东经70到140，北纬0到60的地图对象
m = oyl.map([70,140], [0,60])
##定义画布，同plt.figure
m.figure(figsize=[10,6])
##创建子图（实际开始画地图）
##若需要划分多个子图，可以：
m.subplot(1,2,1)
##加载中国
##类似的还有加载省界(load_province)、河流(load_river)和青藏高原(load_tp)
##加载的函数里面可以定义其它行为，比如画河流的颜色为蓝色
m.load_china(facecolor='red', alpha=0.8)
m.load_river(color='blue')

##默认的投影方式为PlateCarree，可以更换成其它的
##投影方式用cartopy的crs里面的对象
##比如这里，用投影中心为（120°E，60°N）的等距圆锥投影
m.subplot(1,2,2,projection=ccrs.EquidistantConic(120, 60))
m.load_province()
m.show()

"""
上述投影方式没有网格，可以使用gca获取ax添加
这是与cartopy定义的GeoAxe的接口
可以使用GeoAxe的更多方法对地图做自定义
"""
m.subplot(111,projection=ccrs.EquidistantConic(120, 60))
ax = m.gca()
m.load_province()
ax.gridlines(linestyle='--', color='gray')
m.show()
