from oyl import shpmap

##由于shp文件含有经纬度信息，所以可以使用它的信息来定义地图
##coast=False可以设置不画海岸线
##xticks可以设置指定的坐标轴刻度
m = shpmap("../datas/simplied_china_country.shp", coast=False, xticks=range(70,140,30))
m.show()
