from oyl import map

##定义东亚地图, xticks设置x坐标刻度
m = map([70,150], xticks=range(70,151,20))
##给大图用红线绘制省份
m.load_province(edgecolor='red')
"""
small_map画小地图
第一个参数loc=[0.795, 0.005, 0.2, 0.3]
即前两个为小图相对大图的xy位置
后两个是宽度和长度
第二个参数extent=[105, 125, 0, 25]
前2两个是小图的经度范围，后面是纬度范围
init为是否画海岸线，默认True

使用m.small_map语句之前
m的画图操作视为对大图绘制
当使用了m.small_map语句之后
对m做的画图操作视为对小地图绘制
"""
m.small_map()
##给小图用红线绘制中国
m.load_china(edgecolor='red')
m.show()
