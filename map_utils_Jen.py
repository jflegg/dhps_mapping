from numpy import *
from numpy import ma
import numpy as np
import warnings


def grid_convert(g, frm, to, validate=False):
    """Converts a grid to a new layout.
      - g : 2d array
      - frm : format string
      - to : format string

      Example format strings:
        - x+y+ (the way Anand does it) means that 
            - g[i+1,j] is west of g[i,j]
            - g[i,j+1] is north of g[i,j]
        - y-x+ (map view) means that 
            - g[i+1,j] is south of g[i,j]
            - g[i,j+1] is west of g[i,j]"""

    # Validate format strings
    if validate:
        for st in [frm, to]:
            validate_format_str(st)

    # Transpose if necessary
    if not frm[0]==to[0]:
        g = transpose(g,axes=[1,0]+range(2,len(g.shape)))

    first_dir = to[1]
    if not first_dir == frm[frm.find(to[0])+1]:
        # print 'First mismatch'
        g=g[::-1,:]

    sec_dir = to[3]
    if not sec_dir == frm[frm.find(to[2])+1]:
        # print 'Second mismatch'
        g=g[:,::-1]

    # print first_dir, sec_dir
    return g


def get_closest_index(vec, val):
    return argmin(abs(vec-val))
    
def clip_vector(vec, minval, maxval):
    i_min,i_max= max(0,get_closest_index(vec, minval)-1), min(len(vec)-1,get_closest_index(vec,maxval)+1)+1
    return i_min, i_max, vec[i_min:i_max]

def isbetween(one,two,test):
    out = test==one
    out |= sign(maximum(one,two)-test) == sign(test-minimum(one,two))
    return out
    
def crr(rx, ry, rm, x, y):
    crossings = zeros((len(x),len(y)),dtype='int')

    xmn = x.min()
    xmx = x.max()
    dx = x[1]-x[0]

    where_vertical = where(isinf(rm))[0]
    if len(where_vertical)>0:
        rxv, ryv_u, ryv_l = rx[where_vertical], ry[where_vertical], ry[where_vertical+1]

    where_sensible = where((rm != 0) * ~isinf(rm))[0]
    if len(where_sensible)>0:
        rxs_l, rxs_u, rms, rys = rx[where_sensible], rx[where_sensible+1], rm[where_sensible], ry[where_sensible]
    
    for i, y_ in enumerate(y):
        # All the vertical crossings at this longitude.
        
        if len(where_vertical)>0:
            xc_vert = rxv[where(isbetween(ryv_u,ryv_l,y_))]    
        else:
            xc_vert = []
        
        if len(where_sensible)>0:
            xc_sens = rxs_l+(y_-rys)/rms                    
            xc_sens = xc_sens[where(isbetween(rxs_l,rxs_u,xc_sens))]
        else:
            xc_sens = []
        
        xc = hstack((xc_sens, xc_vert))
        xc = xc[where((xc>=xmn)*(xc<=xmx))]
            
        if len(xc)>0:
            for xc_ in xc:
                crossings[int(ceil((xc_-xmn)/dx)):, i] += 1  # needed to add the 'int' to this line, since python 3 doesn't like indexing with non-integers
    
    return crossings
            
def clip_raster_to_ring(ring, lon, lat, isin, hole=False):
    dy = diff(ring.xy[1])
    dx = diff(ring.xy[0])
    ring_slope = dx*0
    ring_slope[where(dx==0)]=inf
    where_nonzero = where(dx!=0)
    ring_slope[where_nonzero] = dy[where_nonzero]/dx[where_nonzero]
    llcx,llcy,urcx,urcy = ring.bounds
    
    llcx_i, urcx_i, lon_patch = clip_vector(lon, llcx, urcx)
    llcy_i, urcy_i, lat_patch = clip_vector(lat, llcy, urcy)
    
    # Assume here that view is x+y+.
    crossings = crr(array(ring.xy[0]),array(ring.xy[1]),array(ring_slope),lon_patch,lat_patch)
    isin_ring = (crossings%2==1)
        
    if hole:
        isin[llcx_i:urcx_i, llcy_i:urcy_i] &= ~isin_ring 
    else:
        isin[llcx_i:urcx_i, llcy_i:urcy_i] |= isin_ring 
    
def clip_raster_to_polygon(poly, lon, lat, isin):
    clip_raster_to_ring(poly.exterior, lon, lat, isin)
    for i in poly.interiors:
        clip_raster_to_ring(i, lon, lat, isin, hole=True)
    
def clip_raster(geom, lon, lat, view='y+x+'):
    """
    Returns an array in the desired view indicating which pixels in the
    meshgrid generated from lon and lat are inside geom.
    """    
    # Internal view is x+y+. (IMPORTANT: x+y+ view)
    isin = zeros((len(lon), len(lat)),dtype='bool')

    if isinstance(geom, shapely.geometry.multipolygon.MultiPolygon):
        geoms = geom.geoms
    else:
        geoms = [geom]
    for i,g in enumerate(geoms):
        clip_raster_to_polygon(g, lon, lat, isin)

    #return map_utils.grid_convert(isin, 'x+y+', view) # doesn't work - map_utils.grid_convert broken in python3 for x+y+ to y+x+
    return isin

# if __name__ == '__main__':
#     import pylab as pl    
#     import pickle, map_utils
#     canada = pickle.loads(file('Canada.pickle').read())
#     lon,lat,data,type = map_utils.import_raster('gr10_10k','.')
#     
#     p = canada#.geoms[5251]
#     
#     isin = clip_raster(p, lon, lat)
#     import pylab as pl
#     pl.clf()
#     pl.imshow(isin,extent=(lon.min(),lon.max(),lat.min(),lat.max()),interpolation='nearest')
#     # pl.imshow(isin,interpolation='nearest')
#     # for g in p.geoms:
#     # pl.plot(p.exterior.xy[0],g.exterior.xy[1],'r-')



def Interp(datain,xin,yin,xout,yout,interpolation='NearestNeighbour'):

    """
       Interpolates a 2D array onto a new grid (only works for linear grids), 
       with the Lat/Lon inputs of the old and new grid. Can perfom nearest
       neighbour interpolation or bilinear interpolation (of order 1)'

       This is an extract from the basemap module (truncated)
    """

    # Mesh Coordinates so that they are both 2D arrays
    xout,yout = np.meshgrid(xout,yout)

   # compute grid coordinates of output grid.
    delx = xin[1:]-xin[0:-1]
    dely = yin[1:]-yin[0:-1]

    xcoords = (len(xin)-1)*(xout-xin[0])/(xin[-1]-xin[0])
    ycoords = (len(yin)-1)*(yout-yin[0])/(yin[-1]-yin[0])


    xcoords = np.clip(xcoords,0,len(xin)-1)
    ycoords = np.clip(ycoords,0,len(yin)-1)

    # Interpolate to output grid using nearest neighbour
    if interpolation == 'NearestNeighbour':
        xcoordsi = np.around(xcoords).astype(np.int32)
        ycoordsi = np.around(ycoords).astype(np.int32)
        dataout = datain[ycoordsi,xcoordsi]

    # Interpolate to output grid using bilinear interpolation.
    elif interpolation == 'Bilinear':
        xi = xcoords.astype(np.int32)
        yi = ycoords.astype(np.int32)
        xip1 = xi+1
        yip1 = yi+1
        xip1 = np.clip(xip1,0,len(xin)-1)
        yip1 = np.clip(yip1,0,len(yin)-1)
        delx = xcoords-xi.astype(np.float32)
        dely = ycoords-yi.astype(np.float32)
        dataout = (1.-delx)*(1.-dely)*datain[yi,xi] + \
                  delx*dely*datain[yip1,xip1] + \
                  (1.-delx)*dely*datain[yip1,xi] + \
                  delx*(1.-dely)*datain[yi,xip1]

    return dataout






def interp_basemap_package(datain,xin,yin,xout,yout,checkbounds=False,masked=False,order=1):
    """
    Interpolate data (``datain``) on a rectilinear grid (with x = ``xin``
    y = ``yin``) to a grid with x = ``xout``, y= ``yout``.

    .. tabularcolumns:: |l|L|

    ==============   ====================================================
    Arguments        Description
    ==============   ====================================================
    datain           a rank-2 array with 1st dimension corresponding to
                     y, 2nd dimension x.
    xin, yin         rank-1 arrays containing x and y of
                     datain grid in increasing order.
    xout, yout       rank-2 arrays containing x and y of desired output grid.
    ==============   ====================================================

    .. tabularcolumns:: |l|L|

    ==============   ====================================================
    Keywords         Description
    ==============   ====================================================
    checkbounds      If True, values of xout and yout are checked to see
                     that they lie within the range specified by xin
                     and xin.
                     If False, and xout,yout are outside xin,yin,
                     interpolated values will be clipped to values on
                     boundary of input grid (xin,yin)
                     Default is False.
    masked           If True, points outside the range of xin and yin
                     are masked (in a masked array).
                     If masked is set to a number, then
                     points outside the range of xin and yin will be
                     set to that number. Default False.
    order            0 for nearest-neighbor interpolation, 1 for
                     bilinear interpolation, 3 for cublic spline
                     (default 1). order=3 requires scipy.ndimage.
    ==============   ====================================================

    .. note::
     If datain is a masked array and order=1 (bilinear interpolation) is
     used, elements of dataout will be masked if any of the four surrounding
     points in datain are masked.  To avoid this, do the interpolation in two
     passes, first with order=1 (producing dataout1), then with order=0
     (producing dataout2).  Then replace all the masked values in dataout1
     with the corresponding elements in dataout2 (using numpy.where).
     This effectively uses nearest neighbor interpolation if any of the
     four surrounding points in datain are masked, and bilinear interpolation
     otherwise.

    Returns ``dataout``, the interpolated data on the grid ``xout, yout``.
    """
    # xin and yin must be monotonically increasing.
    if xin[-1]-xin[0] < 0 or yin[-1]-yin[0] < 0:
        raise ValueError('xin and yin must be increasing!')
    if xout.shape != yout.shape:
        raise ValueError('xout and yout must have same shape!')
    # check that xout,yout are
    # within region defined by xin,yin.
    if checkbounds:
        if xout.min() < xin.min() or \
           xout.max() > xin.max() or \
           yout.min() < yin.min() or \
           yout.max() > yin.max():
            raise ValueError('yout or xout outside range of yin or xin')
    # compute grid coordinates of output grid.
    delx = xin[1:]-xin[0:-1]
    dely = yin[1:]-yin[0:-1]
    if max(delx)-min(delx) < 1.e-4 and max(dely)-min(dely) < 1.e-4:
        # regular input grid.
        xcoords = (len(xin)-1)*(xout-xin[0])/(xin[-1]-xin[0])
        ycoords = (len(yin)-1)*(yout-yin[0])/(yin[-1]-yin[0])
    else:
        # irregular (but still rectilinear) input grid.
        xoutflat = xout.flatten(); youtflat = yout.flatten()
        ix = (np.searchsorted(xin,xoutflat)-1).tolist()
        iy = (np.searchsorted(yin,youtflat)-1).tolist()
        xoutflat = xoutflat.tolist(); xin = xin.tolist()
        youtflat = youtflat.tolist(); yin = yin.tolist()
        xcoords = []; ycoords = []
        for n,i in enumerate(ix):
            if i < 0:
                xcoords.append(-1) # outside of range on xin (lower end)
            elif i >= len(xin)-1:
                xcoords.append(len(xin)) # outside range on upper end.
            else:
                xcoords.append(float(i)+(xoutflat[n]-xin[i])/(xin[i+1]-xin[i]))
        for m,j in enumerate(iy):
            if j < 0:
                ycoords.append(-1) # outside of range of yin (on lower end)
            elif j >= len(yin)-1:
                ycoords.append(len(yin)) # outside range on upper end
            else:
                ycoords.append(float(j)+(youtflat[m]-yin[j])/(yin[j+1]-yin[j]))
        xcoords = np.reshape(xcoords,xout.shape)
        ycoords = np.reshape(ycoords,yout.shape)
    # data outside range xin,yin will be clipped to
    # values on boundary.
    if masked:
        xmask = np.logical_or(np.less(xcoords,0),np.greater(xcoords,len(xin)-1))
        ymask = np.logical_or(np.less(ycoords,0),np.greater(ycoords,len(yin)-1))
        xymask = np.logical_or(xmask,ymask)
    xcoords = np.clip(xcoords,0,len(xin)-1)
    ycoords = np.clip(ycoords,0,len(yin)-1)
    # interpolate to output grid using bilinear interpolation.
    if order == 1:
        xi = xcoords.astype(np.int32)
        yi = ycoords.astype(np.int32)
        xip1 = xi+1
        yip1 = yi+1
        xip1 = np.clip(xip1,0,len(xin)-1)
        yip1 = np.clip(yip1,0,len(yin)-1)
        delx = xcoords-xi.astype(np.float32)
        dely = ycoords-yi.astype(np.float32)
        dataout = (1.-delx)*(1.-dely)*datain[yi,xi] + \
                  delx*dely*datain[yip1,xip1] + \
                  (1.-delx)*dely*datain[yip1,xi] + \
                  delx*(1.-dely)*datain[yi,xip1]
    elif order == 0:
        xcoordsi = np.around(xcoords).astype(np.int32)
        ycoordsi = np.around(ycoords).astype(np.int32)
        dataout = datain[ycoordsi,xcoordsi]
    elif order == 3:
        try:
            from scipy.ndimage import map_coordinates
        except ImportError:
            raise ValueError('scipy.ndimage must be installed if order=3')
        coords = [ycoords,xcoords]
        dataout = map_coordinates(datain,coords,order=3,mode='nearest')
    else:
        raise ValueError('order keyword must be 0, 1 or 3')
    if masked and isinstance(masked,bool):
        dataout = ma.masked_array(dataout)
        newmask = ma.mask_or(ma.getmask(dataout), xymask)
        dataout = ma.masked_array(dataout,mask=newmask)
    elif masked and is_scalar(masked):
        dataout = np.where(xymask,masked,dataout)
    return dataout




def interp_geodata(lon_old, lat_old, data, lon_new, lat_new, mask=None, chunk=None, view='y-x+', order=1, nan_handler=None):
    """
    Takes gridded data, interpolates it to a non-grid point set.
    """
    def chunker(v,i,chunk):
        return v[i*chunk:(i+1)*chunk]
        
    lat_argmins = np.array([np.argmin(np.abs(ln-lat_old)) for ln in lat_new])
    lon_argmins = np.array([np.argmin(np.abs(ln-lon_old)) for ln in lon_new])

    if view[0]=='y':
        lat_index = 0
        lon_index = 1
        lat_dir = int(view[1]+'1')
        lon_dir = int(view[3]+'1')
    else:
        lat_index = 1
        lon_index = 0
        lat_dir = int(view[3]+'1')
        lon_dir = int(view[1]+'1')

    N_new = len(lon_new)
    out_vals = zeros(N_new, dtype=float)

    if chunk is None:
        data = data[:]
        if mask is not None:
            data = ma.MaskedArray(data, mask)
        dconv = grid_convert(data,view,'y+x+')        
        for i in range(N_new):
            out_vals[i] = interp_basemap_package(dconv,lon_old,lat_old,lon_new[i:i+1],lat_new[i:i+1],order=order)
    
        if nan_handler is not None:
            where_nan = np.where(np.isnan(out_vals))
            out_vals[where_nan] = nan_handler(lon_old, lat_old, dconv, lon_new[where_nan], lat_new[where_nan], order)
    
        
    else:
        where_inlon = [np.where((lon_argmins>=ic*chunk[lon_index])*(lon_argmins<(ic+1)*chunk[lon_index]))[0] for ic in range(len(lon_old)/chunk[lon_index])]
        where_inlat = [np.where((lat_argmins>=jc*chunk[lat_index])*(lat_argmins<(jc+1)*chunk[lat_index]))[0] for jc in range(len(lat_old)/chunk[lat_index])]
        
        # Always iterate forward in longitude and latitude.
        for ic in range(data.shape[lon_index]/chunk[lon_index]):
            for jc in range(data.shape[lat_index]/chunk[lat_index]):

                # Who is in this chunk?
                where_inchunk = intersect1d(where_inlon[ic],where_inlat[jc])
                if len(where_inchunk) > 0:

                    # Which slice in latitude? 
                    if lat_dir == 1:
                        lat_slice = slice(jc*chunk[lat_index],(jc+1)*chunk[lat_index],None)
                    else:
                        lat_slice = slice(len(lat_old)-(jc+1)*chunk[lat_index],len(lat_old)-jc*chunk[lat_index],None)

                    # Which slice in longitude?
                    if lon_dir == 1:
                        lon_slice = slice(ic*chunk[lon_index],(ic+1)*chunk[lon_index],None)
                    else:
                        lon_slice = slice(len(lon_old)-(ic+1)*chunk[lon_index],len(lon_old)-ic*chunk[lon_index],None)

                    # Combine longitude and latitude slices in correct order
                    dslice = [None,None]
                    dslice[lat_index] = lat_slice
                    dslice[lon_index] = lon_slice
                    dslice = tuple(dslice)

                    dchunk = data[dslice]
                    if mask is not None:
                        mchunk = mask[dslice]
                        dchunk = ma.MaskedArray(dchunk, mchunk)

                    latchunk = chunker(lat_old,jc,chunk[lat_index])                
                    lonchunk = chunker(lon_old,ic,chunk[lon_index])

                    dchunk_conv = grid_convert(dchunk,view,'y+x+')

                    # for index in where_inchunk:
                    out_vals[where_inchunk] = interp_basemap_package(dchunk_conv, lonchunk, latchunk, lon_new[where_inchunk], lat_new[where_inchunk], order=order)
                    
                    if nan_handler is not None:
                        where_nan = np.where(np.isnan(out_vals[where_inchunk]))
                        out_vals[where_inchunk][where_nan] = nan_handler(lonchunk, latchunk, dchunk_conv, lon_new[where_inchunk][where_nan], lat_new[where_inchunk][where_nan], order)                

    return out_vals



