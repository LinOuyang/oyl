# oyl
oyl自定义的库，最为主要的用途用于画地图，除此之外，还有一些其它的个人常用功能。

## 背景
对于气象专业相关的研究人员，常需要在地图上绘制等值线，比较主流的绘制地图的库有basemap和cartopy两种。
两者在绘制地图的方式上存在一定差异，
cartopy主要是在画图轴ax的基础上加入了投影
basemap则是直接新建一个地图对象,
但随着时间的推移，basemap逐渐被淘汰，cartopy成为主流
oyl通过将加入cartopy投影的ax进行封装，创建了一个地图对象
在一定程度上能够作为逝去的basemap的替代品

## 安装
该库已发布pypi，当cartopy、xarray和sklearn已安装后，使用pip安装：
```
pip install oyl
```

## 快速上手

### 创建底图
```
import oyl
m = oyl.map() #创建底图对象
m.load_china(facecolor='red', alpha=0.7) #用透明红色加载中国底图
m.show() #查看图片
```
