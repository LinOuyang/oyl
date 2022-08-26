import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
import warnings

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import xarray as xr


def _get_unique(dic1, dic2):
    d = {}
    for i in dic1.keys():
        if i not in dic2.keys():
            d.update({i: dic1[i]})
    return d


class map:

    def __init__(self, x=[70,140], y=[0,60], xticks=None, yticks=None, center=110, style='minor', **kwargs):
        """Create a geographical map object.

        The map object can be initializated one time, and be drew many times. Method
        calls are used to set paramters or drawing. Some paramters are shared by all
        the methods of this map object.

        **Arguments:**

        *x*
            A list that contains three or two items. The first two iterms are the
            beginning and the ending of the longitudes. The third iterm is optional
            which means the step from the beginning to the end, if it is not given, we
            would set it 1 as default.

        *y*
            The same as x, but was lattitude's.

        **Optional arguments:**

        *xticks*
            A iterable object. If labeled, label the specific longitudes.If is default,
            draw some label lines in the map.

        *yticks*
            The same as xticks, but was lattitude's.

        *center*
            If the projection is the default PlateCarree, center is the center longitude of this projection.
        
        *style*
            The drawing style, default "minor" means to draw the minor locator.
            Can be set as "minor" and the "norm".

        **Returns:**
            A basemap object.

        *Other parameters:*
        
        *subplot*
            subplot infomation, set to (1,1,1) as default.

        *projection*
            A cartopy.crs object. It is the projection method of map.

        *transform*
            A cartopy.crs object that transform the data projection,
            only used when filling the data.

        *crs*
            A cartopy.crs, was used to define a map.

        *coast*
            Bool. Whether to draw the coast.(True as default)
            
        *coast_linewidth*   : default 1
        *coast_color*   : default "black"
        *coast_linestyle*  : default '-'
        *coast_resolution*  : default "110m"

        *borders*
            Bool. Whether to draw the border of the countries.(False as default)
            
        *border_linewidth*  : 1
        *border_linestyle*  : '-'
        *border_color*  : 'k'
        *border_resolution*  : "110m"

        *land*
            Bool. Whether to draw the land.(False as default)
            
        *land_linewidth*  : 1
        *land_linestyle*  : '-'
        *land_color*  : 'k'
        *land_resolution*  : "110m"
        
        *rivers*
            Bool. Whether to load the rivers.(False as default)     
        *river_linewidth*  : 1
        *river_linestyle*  : '-'
        *river_color*  : 'blue'
        *river_resolution*  : '110m',

        *lake*:  the same as *rivers*

        *lonlat*
            Bool. Whether to label the longitude and latitude.(True as default)

        *grid*: the same as borders

        *labels*
            A list that defines where to draw the lonlat labels.
            Set ['left','bottom'] as default.



        **Examples**
            >>> m = map([70,130,2],[15,55,2])
            >>> m.load_china()
            >>> d = np.arange(31*21).reshape(21,31)
            >>> c = m.contourf(d,cmap=plt.cm.bwr)
            >>> m.show()
            
            Tips:
            If we need to draw some other maps, we can run as:
            >>> m.subplot(2,3,1)
            >>> m.load_province()
            >>> m.contourf(d)
            Or:
            >>> m.axes([0.1,0.1,0.5,0.5])
            >>> m.contourf(d)
            
            If we need to add a shapefile, it runs as:
            >>> m.add_shape("./shapefiles/TPBoundary_2500.shp")
            
            If we need other projection, it should be define in the map initialization:
            >>> import cartopy.crs as ccrs
            >>> m = map([70,130,2],[15,55,2],projection=ccrs.EquidistantConic(110,60))
            
        For more usage, use help(func) or contract me.
        My QQ number is 1439731362.

        """
        if np.max(np.abs(y))>90:
            ErrorMess = "latitude must be in [-90,90]"
            raise ValueError(ErrorMess)
        self._para = {'projection': ccrs.PlateCarree(center), 'transform': ccrs.PlateCarree(),
                      'crs': ccrs.PlateCarree(),
                      'coast': True, 'coast_linewidth': 1, 'coast_color': 'black', 'coast_linestyle': '-',
                      'coast_resolution': '110m',
                      'borders': False, 'border_linewidth': 1, 'border_linestyle': '-', 'border_color': 'k',
                      'border_resolution': '110m',
                      'land': False, 'land_linewidth': 1, 'land_linestyle': '-', 'land_color': 'gray',
                      'land_resolution': '110m',
                      'rivers': False, 'river_linewidth': 1, 'river_linestyle': '-', 'river_color': 'blue',
                      'river_resolution': '110m',
                      'lakes': False, 'lake_linewidth': 0.8, 'lake_linestyle': '-', 'lake_color': 'blue',
                      'lake_resolution': '110m',
                      'lonlat': True, 'grid': False, 'grid_linewidth': 0.9, 'grid_linestyle': ':', 'grid_color': 'k',
                      'labels': ['left', 'bottom'],

                      }
        self._para.update(kwargs)
        x, y = list(x), list(y)
        x[1], y[1] = x[1] + 0.0001, y[1] + 0.0001
        if x[1] > 360:
            x[1] = 359.99
        self.xlims, self.ylims = x[:2], y[:2]
        X, Y = np.arange(*x), np.arange(*y)
        
        self.x, self.y = X, Y
        self.xticks = np.round(np.linspace(*self.xlims, 5)) if xticks is None else np.array(xticks)
        self.xticks[self.xticks > 180] = self.xticks[self.xticks > 180] - 360
        self.xticks = np.sort(self.xticks)
        self.yticks = np.round(np.linspace(*self.ylims, 5)) if yticks is None else np.array(yticks)
        self.MainWidget = False
        self.style = style

        
    def set_style(self, xnums=5, ynums=5, **kwargs):
        from matplotlib.ticker import AutoMinorLocator
        kw = dict(which='major', width=1.1, length=5)
        kw.update(kwargs)
        
        def __set_style(xnums=xnums, ynums=ynums, kw=kw):
            self.ax.tick_params(**kw)
            self.ax.xaxis.set_minor_locator(AutoMinorLocator(xnums))
            self.ax.yaxis.set_minor_locator(AutoMinorLocator(ynums))

        self.__style_func = __set_style

    def __repr__(self):
        a, b = round(self.xlims[0], 2), round(self.xlims[1], 2)
        if b == 359.99:
            b = 360
        c, d = round(self.ylims[0], 2), round(self.ylims[1], 2)
        string = "<oyl.map>\nxlims :   {:} - {:}\nylims :   {:} - {:}".format(a, b, c, d)
        return string

    def gca(self):
        if self.MainWidget:
            return self.ax
        else:
            return plt.gca()
    
    def extent(self, extents=None, crs = None):
        crs = crs if crs else self._para['crs']  
        if not extents:
            extents = self.xlims[0], self.xlims[1], self.ylims[0], self.ylims[1]
        self.ax.set_extent(extents, crs=crs)

    def figure(self, *args, **kwargs):
        """
        See matplotlib.pyplot.figure
        """
        self.fig = plt.figure(*args, **kwargs)
        self.MainWidget = False
        return self.fig

    def __draw_terrain(self):
        if self._para['coast']:
            self.ax.coastlines(resolution=self._para['coast_resolution'], lw=float(self._para['coast_linewidth']),
                               color=self._para['coast_color'], linestyle=self._para['coast_linestyle'])
        if self._para['borders']:
            self.ax.add_feature(cfeature.BORDERS.with_scale(self._para['border_resolution']),
                                linestyle=self._para['border_linestyle'],
                                lw=float(self._para['border_linewidth']), edgecolor=self._para['border_color'])
        if self._para['land']:
            self.ax.add_feature(cfeature.LAND.with_scale(self._para['land_resolution']),
                                linestyle=self._para['land_linestyle'],
                                lw=float(self._para['land_linewidth']), color=self._para['land_color'])
        if self._para['rivers']:
            self.ax.add_feature(cfeature.RIVERS.with_scale(self._para['river_resolution']),
                                linestyle=self._para['river_linestyle'],
                                lw=float(self._para['river_linewidth']), color=self._para['river_color'])
        if self._para['lakes']:
            self.ax.add_feature(cfeature.LAKES.with_scale(self._para['lake_resolution']),
                                linestyle=self._para['lake_linestyle'],
                                lw=float(self._para['lake_linewidth']), color=self._para['lake_color'])

    def __draw_lonlat(self, projection=None):
        proj = self._para["projection"] if projection is None else projection
        if self._para['lonlat']:
            draw_labels = False if isinstance(proj, ccrs.PlateCarree) else True
            if not self._para['grid']:
                self._para['grid_linewidth'] = 0
            gl = self.ax.gridlines(draw_labels=draw_labels, linewidth=self._para['grid_linewidth'],
                                   linestyle=self._para['grid_linestyle'], color=self._para['grid_color'],
                                   alpha=0.8,
                                   xlocs=self.xticks, ylocs=self.yticks)
            for la in ['left', 'bottom', 'top', 'right']:
                if la not in self._para["labels"]:
                    exec(f"gl.{la}_labels=False")
            if isinstance(proj, ccrs.PlateCarree):
                gl.xlocator = mticker.FixedLocator(self.xticks)
                gl.ylocator = mticker.FixedLocator(self.yticks)

                self.ax.set_xticks(self.xticks, crs=self._para['crs'])
                self.ax.set_yticks(self.yticks, crs=self._para['crs'])
                self.ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
                self.ax.yaxis.set_major_formatter(LatitudeFormatter())

    def __apply_style(self):
        if self.style=='minor':
            try:
                self.__style_func()
            except:
                self.set_style()
                self.__style_func()

    def subplot(self, *args, **kwargs):
        """
        See matplotlib.pyplot.subplot
        """
        self.MainWidget = False
        return self.__subplot(*args, **kwargs)

    def __subplot(self, *args, **kwargs):

        if not self.MainWidget:
            if 'projection' not in kwargs:
                kwargs.update({"projection":self._para['projection']})
            self.ax = plt.subplot(*args, **kwargs)
            self.__draw_terrain()
            self.__draw_lonlat(projection=kwargs['projection'])

            self.extent()
            
            self.MainWidget = True
            self.__apply_style()
            return self.ax

    def axes(self, *args, **kwargs):
        """
        See matplotlib.pyplot.axes
        """
        self.MainWidget = False
        self.ax = self.__axe(*args, **kwargs)
        return self.ax

    def __axe(self, *args, **kwargs):

        if not self.MainWidget:
            if 'projection' not in kwargs:
                kwargs.update({"projection":self._para['projection']})
            self.ax = plt.axes(*args, **kwargs)
            self.__draw_terrain()
            self.__draw_lonlat(projection=kwargs['projection'])

            self.extent()
            
            self.MainWidget = True
            self.__apply_style()
            return self.ax

    def subaxes(self, *args, projection=None):
        """
        See matplotlib.pyplot.axes
        """
        proj = projection if projection else self._para['projection']
        self.ax = plt.axes(args, projection=proj)
        return self.ax

    def small_map(self, loc=[0.795, 0.005, 0.2, 0.3], extent=[105,125,0,25], init=True, projection=None):
        """
        Draw a small map in the main map.
        
        *loc*
            The relative location of the whole main map.

        *extent*
            The extent of the small map

        *init*
            Whether to draw the coasts and lands and so on

        return:
            A GeoAxes
        """

        if not self.MainWidget:
            self.__subplot(111)

        a, b, c, d = loc
        x, y, w, h = self.ax.get_position().bounds

        X, Y = x + w*a, y + h*b
        W, H = c*w, d*h
        self.ax = self.subaxes(X, Y, W, H)
        self.ax.set_extent(extent, ccrs.PlateCarree())
        if init:
            self.__draw_terrain()
        return self.ax

    def __contourf(self, *args, **kwargs):
        if len(args) == 1:
            args = self.x, self.y, args[0]

        dic = {'cbar': True, 'fmt': '%d', 'pad': 0.08, 'fraction': 0.04,
               'location': 'right', 'cax': None}
        
        kw = _get_unique(kwargs, dic)
        dic.update(kwargs)
        settings = {'zorder':0, "transform":self._para['transform']}
        settings.update(kw)

        self.contourf_map = self.ax.contourf(*args, **settings)
        if dic['cbar']:
            ori = {'bottom': 'horizontal', 'right': 'vertical'}
            self.colorbar = plt.colorbar(self.contourf_map, orientation=ori[dic['location']], format=dic['fmt'],
                                         pad=dic['pad'], fraction=dic['fraction'], cax=dic['cax'])
        return self.contourf_map

    def contourf(self, *args, **kwargs):
        """
        This function will automatically add latitude and longitude if they are not given.
        This function doesn't need to consider the transform as it is automatically defined.
        The colorbar will be automatically added.
        The colorbar related parameters are:
        *cbar*
            Bool. Whether to add the colorbar.(True as default)
        *location*
            "bottom" or "right", the location of the cbar.  
        More arguments can be found from matplotlib.pyplot.contourf and matplotlib.pyplot.colorbar
        """
        if not self.MainWidget:
            self.__subplot(111)
        return self.__contourf(*args, **kwargs)

    def __contour(self, *args, **kwargs):
        if len(args) == 1:
            args = self.x, self.y, args[0]

        dic = {'clabel': True, 'fmt': '%d', 'inline': True, 'linewidths': 0.85,
               'fontsize': 8, 'color': 'k'}
        kw = _get_unique(kwargs, dic)
        if ('cmap' in kw.keys()) & ('colors' in kw.keys()):
            kw.pop('colors')
        dic.update(kwargs)
        if 'transform' not in kw:
            kw.update({"transform":self._para['transform']})
        self.contour_map = self.ax.contour(*args, linewidths=dic['linewidths'], **kw)

        if dic['clabel']:
            cb = plt.clabel(self.contour_map, inline=dic['inline'], fontsize=dic['fontsize'],
                            colors=kwargs.get('color', kwargs.get('colors', None)), fmt=dic['fmt'])
        return self.contour_map

    def contour(self, *args, **kwargs):
        """
        This function will automatically add latitude and longitude if they are not given.
        This function doesn't need to consider the transform as it is automatically defined.
        The labels will be automatically added.
        The labels related parameters are:
        *clabel*
            Bool. Whether to add the labels.(True as default)
        More arguments can be found from matplotlib.pyplot.contour and matplotlib.pyplot.clabel
        """
        if not self.MainWidget:
            self.__subplot(111)
        return self.__contour(*args, **kwargs)

    def ttest_plot(self, logic_matrix, x=None, y=None, **kwargs):
        """
        This function is uesd for drawing points for the area who passing inspection.
        Not recommended. This function will be abandon in the futrue.
        Use other funcions such as scatter or plot instead.
        """

        dic = {'color': 'k', 'marker': 'o', 'linestyle': '',
               'alpha': 0.8, 'markersize': 1, }

        X, Y = self.x if x is None else x, self.y if y is None else y
        xx, yy = np.meshgrid(X, Y)
        x, y = xx[logic_matrix], yy[logic_matrix]
        self.ax.plot(x, y, color=dic['color'], marker=dic['marker'], linestyle=dic['linestyle'],
                     alpha=dic['alpha'], markersize=dic['markersize'], transform=self._para['transform'])
        return

    def __pcolor(self, *args, **kwargs):
        if len(args) == 1:
            args = self.x, self.y, args[0]


        dic = {'cbar': True, 'fmt': '%d', 'pad': 0.08, 'fraction': 0.04,
               'location': 'right', 'cax': None, 'extend': 'neither'}
        kw = _get_unique(kwargs, dic)
        dic.update(kwargs)
        settings = {'zorder':0, "transform":self._para['transform']}
        settings.update(kw)

        self.pcolor_map = self.ax.pcolor(*args, **settings)
        if dic['cbar']:
            ori = {'bottom': 'horizontal', 'right': 'vertical'}
            cb = plt.colorbar(self.pcolor_map, orientation=ori[dic['location']], format=dic['fmt'],
                              pad=dic['pad'], fraction=dic['fraction'], cax=dic['cax'], extend=dic['extend'])
        return self.pcolor_map

    def pcolor(self, *args, **kwargs):
        """
        This function will automatically add latitude and longitude if they are not given.
        This function doesn't need to consider the transform as it is automatically defined.
        The colorbar will be automatically added.
        The colorbar related parameters are:
        *cbar*
            Bool. Whether to add the colorbar.(True as default)
        *location*
            "bottom" or "right", the location of the cbar.  
        More arguments can be found from matplotlib.pyplot.pcolor and matplotlib.pyplot.colorbar
        """
        if not self.MainWidget:
            self.__subplot(111)
        return self.__pcolor(*args, **kwargs)

    def __quiver(self, *args, skip=None, legend=False, **kwargs):
        if len(args) == 2:
            args = [self.x, self.y, args[0], args[1]]

        dic = dict(X=1.05,Y=0.45,U=10,label='10m/s',angle=90,labelpos='N')
        kw = _get_unique(kwargs, dic)
        dic.update(kwargs)
        for key in kw:
            dic.pop(key)
        if 'transform' not in kw:
            kw.update({"transform":self._para['transform']})

        
        if skip is not None:
            xs, ys = skip
            args[0] = args[0][::xs]
            args[1] = args[1][::ys]
            args[2] = args[2][::ys, ::xs]
            args[3] = args[3][::ys, ::xs]
        self.quiver_map = self.ax.quiver(*args, **kw)
        if legend:
            plt.quiverkey(self.quiver_map, **dic)
        return self.quiver_map

    def quiver(self, *args, skip=None, legend=False, **kwargs):
        """
        This function will automatically add latitude and longitude if they are not given.
        This function doesn't need to consider the transform as it is automatically defined.
        The labels will be automatically added.
        The other parameters are:
        *skip*
            A list that controls the density of x and y directions.
            If the resolution of the data is so high that the wind field is too dense to see clearly, we need to sparse it.
        *legend*
            Bool. Whether to add the wind scale.(False as default)

        **Examples**
        >>>m.quiver(u,v,skip=[3,3],scale=200,legend=True,X=1.04,Y=0.45,U=4,label='4m/s',angle=90,labelpos='N')
        
        More arguments can be found from matplotlib.pyplot.quiver and matplotlib.pyplot.quiverkey
        """
        if not self.MainWidget:
            self.__subplot(111)
        return self.__quiver(*args, skip=skip, legend=legend, **kwargs)

    def __streamplot(self, *args, skip=None, **kwargs):
        if len(args) == 2:
            args = [self.x, self.y, args[0], args[1]]
        kw = dict(color='k', linewidth=0.6, arrowsize=0.8, arrowstyle='<-', transform=self._para['transform'])
        kw.update(kwargs)
        
        
        if skip is not None:
            xs, ys = skip
            args[0] = args[0][::xs]
            args[1] = args[1][::ys]
            args[2] = args[2][::ys, ::xs]
            args[3] = args[3][::ys, ::xs]
        self.stream_map = self.ax.quiver(*args, **kw)
        return self.stream_map

    def streamplot(self, *args, skip=None, **kwargs):
        """
        See matplotlib.pyplot.streamplot
        """
        if not self.MainWidget:
            self.__subplot(111)
        return self.__streamplot(*args, skip=skip, **kwargs)

    def plot(self, *args, **kwargs):
        """
        See matplotlib.pyplot.plot
        """
        if not self.MainWidget:
            self.__subplot(111)
        if 'transform' not in kwargs:
            kwargs.update({"transform":self._para['transform']})
        return self.ax.plot(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        """
        See matplotlib.pyplot.scatter
        """
        if not self.MainWidget:
            self.__subplot(111)
        if 'transform' not in kwargs:
            kwargs.update({"transform":self._para['transform']})
        return self.ax.scatter(*args, **kwargs)

    def add_shape(self, filename, linewidth=1, linestyle='-', encoding='gbk', rec=-1, method=None, **kwargs):
        """
        There are two ways to add a shapefile to the map.
        One is to use cartopy.io to read the shp file and cartopy.feature to add it.
        The other is to use pyshp to read the shp file and add a matplotlib.patches.
        Some shp files are difficult to read. This function would try another method after one fails.

        **Arguments:**

        *filename*
            The shp file path. The three files (.shp, .shx, .dbf) are needed.

        **Optional arguments:**

        *encoding*
            The shp file encoding, only useful when using the pyshp's Reader.

        *method*
            Which way to load the shp. (None as default as the cartopy, and set to the other if it fails.)
            It can be any object that can make logical judgment positive to use pyshp to read the shp file and add a matplotlib.patches.

        *rec*
            A number. Sometimes the shp file contains many records which represents different areas.
            Each area has its ID number. This arguments tells the map which area to draw.
            It is -1 by default as all the areas are shown. (Only useful for the pyshp method)
               
        More arguments can be seen for cartopy.feature.ShapelyFeature, matplotlib.patches.Polygon
        """
        if not self.MainWidget:
            self.__subplot(111)
        from cartopy.io.shapereader import Reader
        dic = dict(edgecolor='k', facecolor=None)
        dic.update(kwargs)
        if not method:
            try:
                dic0 = dic.copy()
                reader = Reader(filename)
                dic0['facecolor'] = dic['facecolor'] if dic['facecolor'] else 'none'
                enshicity = cfeature.ShapelyFeature(reader.geometries(), self._para['transform'], **dic0)
                self.ax.add_feature(enshicity, linewidth=linewidth, linestyle=linestyle)
                reader.close()
            except:
                method = 1
        if method:
            import matplotlib.patches as patches
            from shapefile import Reader
            if isinstance(rec, int) and (rec > 0):
                rec = [rec]
            sf = Reader(filename, encoding=encoding)
            fill = True if dic['facecolor'] else False
            for i, sh in enumerate(sf.shapes()):
                if (rec == -1) or (i in rec):
                    points = np.array(sh.points)
                    if len(sh.parts) == 2:
                        self.ax.add_patch(
                            patches.Polygon(points[sh.parts[1]:], linestyle=None, transform=self._para['transform'],
                                            color=dic.get('color', dic['edgecolor']), fill=fill, alpha=dic.get('alpha', None), 
                                            facecolor=dic['facecolor']))
                        points = points[:sh.parts[1]]
                    self.ax.add_patch(patches.Polygon(points, linewidth=linewidth, linestyle=linestyle,
                                                      transform=self._para['transform'],
                                                      color=dic.get('color', dic['edgecolor']), fill=fill, alpha=dic.get('alpha', None), 
                                                      facecolor=dic['facecolor']))
        return

    def load_province(self, **kwargs):
        """
        Add a China map which contains every province.
        More arguments can be seen for self.add_shape
        """
        file = os.path.dirname(__file__) + '/shapefiles/cnhimap.shp'
        self.add_shape(file, **kwargs)

    def load_china(self, **kwargs):
        """
        Add a China map.
        More arguments can be seen for self.add_shape
        """
        file = os.path.dirname(__file__) + '/shapefiles/china_country.shp'
        self.add_shape(file, **kwargs)

    def load_river(self, **kwargs):
        """
        Add the rivers in China.
        More arguments can be seen for self.add_shape
        """
        file = os.path.dirname(__file__) + '/shapefiles/rivers.shp'
        self.add_shape(file, **kwargs)

    def load_tp(self, **kwargs):
        """
        Add the Qinghai Tibet Plateau.
        More arguments can be seen for self.add_shape
        """
        file = os.path.dirname(__file__) + '/shapefiles/TPBoundary_2500.shp'
        self.add_shape(file, **kwargs)

    def add_box(self, x, y, fill=False, **kwargs):
        """
        Add a rectangle box.
        
        **Arguments:**

        *x*
            A list that contains three or two items. The first two iterms are the
            beginning and the ending of the longitudes.

        *y*
            The same as x, but was lattitude's.
        
        More arguments can be seen for matplotlib.patches.Rectangle
        """
        if not self.MainWidget:
            self.__subplot(111)
        if 'transform' not in kwargs:
            kwargs.update({"transform":self._para['transform']})
        tmp = [x[0], y[0], x[1] - x[0], y[1] - y[0]]
        self.ax.add_patch(
            patches.Rectangle(tmp[:2], tmp[2], tmp[3], fill=fill, **kwargs))
        return

    def text(self, *args, **kwargs):
        """
        See matplotlib.pyplot.text
        """
        if not self.MainWidget:
            self.__subplot(111)
        if 'transform' not in kwargs:
            kwargs.update({"transform":self._para['transform']})
        self.ax.text(*args, **kwargs)

    def title(self, label, *args, **kwargs):
        """
        See matplotlib.pyplot.title
        """
        plt.title(label, *args, **kwargs)

    def show(self):
        if not self.MainWidget:
            self.__subplot(111)
        self.MainWidget = False
        plt.show()

    def savefig(self, *arg, **kwargs):
        """
        See matplotlib.pyplot.savefig
        """
        plt.savefig(*arg, **kwargs)


class ncmap(map):

    def __init__(self, file, xticks=None, yticks=None, center=110, **kwargs):
        """
        Create a map object from a nc file.

         **Arguments:**

        *file*
            The nc file path or a xarray.Dataset object.
            If it is a filepath, we use xarray.open_dataset(file) to open it.
        
        More arguments can be seen for oyl.map
        """

        f = xr.open_dataset(file) if isinstance(file, str) else file
        lat = 'lat' if 'lat' in f.coords else 'latitude'
        lon = 'lon' if 'lon' in f.coords else 'longitude'
        lat, lon = f[lat].data, f[lon].data
        dx, dy = lon[1] - lon[0], lat[1] - lat[0]

        super().__init__([lon.min(), lon.max(), dx], [lat.min(), lat.max(), dy],
                         xticks=xticks, yticks=yticks, center=center, **kwargs)
        self.x = lon
        self.y = lat


class shpmap(map):

    def __init__(self, file, encoding='gbk', shape=True, xticks=None, yticks=None, center=110, **kwargs):

        """
        Create a map object from a nc file.

        **Arguments:**

        *file*
            The nc file path or a xarray.Dataset object.
            If it is a filepath, we use xarray.open_dataset(file) to open it.
        
        More arguments can be seen for oyl.map
        """

        from shapefile import Reader
        try:
            sf = Reader(file, encoding=encoding)
        except:
            sf = Reader(file, encoding='utf-8')
        pts = []
        for i, sp in enumerate(sf.shapes()):
            pts.append(np.array(sp.points))
        pts = np.vstack(pts)
        xmax, xmin = np.max(pts[:, 0]), np.min(pts[:, 0])
        ymax, ymin = np.max(pts[:, 1]), np.min(pts[:, 1])

        super().__init__([xmin, xmax], [ymin, ymax], xticks=xticks, yticks=yticks, center=center, **kwargs)
        if shape:
            self.add_shape(file, encoding=encoding)


def set_xy_delta(step=None, axis='x'):
    if step == None:
        return
    else:
        ax = plt.gca()
    if (axis != 'y') | (axis != 'Y'):
        ax.xaxis.set_major_locator(plt.MultipleLocator(step))
    else:
        ax.yaxis.set_major_locator(plt.MultipleLocator(step))


def font(font='simsun'):
    plt.rcParams['font.sans-serif'] = [font]
    plt.rcParams['axes.unicode_minus'] = False


def mat_bar(matrix, **kwargs):
    para = {'number': True, 'width': 0.2, 'legend': None, 'fmt': '.1f',
            'fontsize': 8,
            'color': ['blue', 'deepskyblue', 'orange', 'red'],
            }
    para.update(kwargs)
    data = np.array([matrix]) if matrix.ndim < 2 else np.array(matrix)
    shp = data.shape

    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    for j in range(shp[0]):
        idx = j if j < len(para['color']) else j % len(para['color'])
        w = 0.5 * para['width'] - para['width'] * shp[0] / 2 + j * para['width']
        x, y = np.arange(1 + w, shp[1] + w + 1), data[j, :]

        ax.bar(x, y, width=para['width'], color=para['color'][idx])

        if para['number']:
            for a, b in zip(x, y):
                txt = '{:' + para['fmt'] + '}'
                ax.text(a - 0.5 * para['width'] + 0.01, b + 0.015 * np.max(data), txt.format(b),
                        fontsize=para['fontsize'])

        if para['legend'] is not None:
            ax.legend(para['legend'], framealpha=0.4)


if __name__ == '__main__':
    d = np.arange(31 * 21).reshape(21, 31)
    m = map([70, 130, 2], [15, 55, 2], projection=ccrs.EquidistantConic(110, 60))
    m.axes([0.1, 0.1, 0.5, 0.5])
    c = m.contourf(d, cmap=plt.cm.bwr)
    m.load_tp()
    m.load_china()
    m.axes([0.6, 0.2, 0.3, 0.3])
    m.load_china()
    c = m.contourf(d, cmap=plt.cm.bwr)
    m.show()
