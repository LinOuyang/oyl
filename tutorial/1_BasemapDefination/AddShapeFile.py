import oyl

##定义地图底图且不绘制海岸线
m = oyl.map(y=[15,55], coast=False)
##看一眼地图(空白，什么都没有
m.show()
##随后开始添加shapefile
##添加简化的中国地图
m.add_shape("../Datas/simplied_china_country.shp")
##按照默认参数添加江苏
m.add_shape("../Datas/江苏.shp")
##需要改变轮廓颜色用color
m.add_shape("../Datas/安徽.shp",  color='blue')
##需要改变填充颜色用fececolor,同时，规定color为None
m.add_shape("../Datas/江西.shp",  color=None, facecolor='red')

m.show()
