import xarray as xr
import oyl

##读取nc数据
ds = xr.open_dataset("../datas/era5uvz500.nc")
##读取uv风速，需要.data提取成np的数组，因为后面画图不支持DataArray对象
u, v = ds.u[0].data, ds.v[0].data
##定义一个ds数据集描述的地图对象
m = oyl.ncmap(ds, coast_color='grey', coast_linewidth=0.8)

"""
m.quiver特有参数：
skip:规定xy方向间隔的格点数
legend:是否画比例尺，默认False

plt.quiver参数：
headwidth:箭头宽度与矢量杆的倍数
width:矢量杆的宽度

plt.quiverkey参数：
X, Y : 图的相对位置
U: 比例尺中箭头的长度
label:比例尺的标注
angle:比例尺箭头旋转的角度
labelpos:标注的位置，北极
"""
m.quiver(u, v, skip=(60,60), headwidth=2.5, width=0.002,
         legend=True, X=1.04, Y=0.45, U=10, label='10m/s', angle=90, labelpos='N')

m.show()

 
