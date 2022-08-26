from oyl import ncmap
import xarray as xr

ds = xr.open_dataset("../datas/era5uvz500.nc")
##由于nc文件含有经纬度信息，所以可以使用它的信息来定义地图
m = ncmap(ds)
##同理加载中国地图
m.load_china(facecolor='red')
m.show()
