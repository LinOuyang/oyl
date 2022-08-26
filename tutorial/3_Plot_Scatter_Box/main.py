from oyl import np, plt, map

##定义读取数据的函数
def read(file,skip=1):
    f = open(file,'r')
    f.readlines(skip)
    data = [] 
##    遍历每一行
    for line in f:
##        以空格为间隔读数
        d = [int(i) for i in line.split(' ') if i not in['','\n']]
##        每行读取前7列数
        data.append(d[:7])
    return np.array(data)

##读取maysak台风
data = read('../Datas/maysak.txt')
##第5行是经度，4是纬度，6是气压，代表强度
lon, lat, power = data[:,4]/10, data[:,3]/10, data[:,5]
##定义底图，东经90到160，北纬5到55
m = map([90,160],[5,55])
plt.subplots_adjust(bottom=0.2,left=0.05)
##以蓝色加载长江黄河两条河流线
m.load_river(edgecolor='blue')
##用经纬度连成一条黑色虚线
m.plot(lon,lat,'k--')
##用经纬度画点，用颜色表示强度
c1 = m.scatter(lon,lat,c=power,cmap=plt.cm.Blues_r, s=20,marker='d')
##下面同理
data = read('../Datas/bavi.txt')
lon, lat, power = data[:,4]/10, data[:,3]/10, data[:,5]
m.plot(lon,lat,'k--')
c2 = m.scatter(lon,lat,c=power,cmap=plt.cm.YlGn_r,s=20,marker='d')
##在指定位置用红色虚线画矩形框
m.add_box([122,129],[26,30], color='red', linestyle='--')
##画色标，定义其位置和方向
plt.colorbar(c1,cax=plt.axes([0.12,0.08,0.7,0.025]),orientation='horizontal',extend='both')
plt.colorbar(c2,cax=plt.axes([0.87,0.15,0.025,0.72]),orientation='vertical',extend='both')
plt.show()

