from __future__ import division
import numpy as np
import pyfits

WMIN = 5710
WMAX = 9265

MASKATMOS = [(1, WMIN),
             (5870, 5910),
             (6275, 6325),
             (6350, 6385),
             (7550, 7700),
             (WMAX, 1e9)]


def read_adp(filename):
    """ Read object ids from a VIMOS mask preparation file (vmmps .adp
    file)."""
    fh = open(filename)
    f = fh.readlines()
    fh.close()
    iobj = [i for i,r in enumerate(f) if 'ID' in r]
    temp = []
    # ra, dec in degrees? or ra in hours?
    for i in iobj:
        if 'SLIT' not in f[i]: continue
        #print f[i], f[i+1], f[i+2]
        vals = [val.split()[-1].strip('"') for val in f[i:i+7]]
        slit = int(val.split()[0].split('.')[1][4:])
        num = int(vals[0])
        ra, dec = vals[1:3]
        X,Y,DIMX,DIMY =  [float(v) for v in vals[3:]]
        try:
            ra = float(ra)
            dec = float(dec)
        except ValueError:
            ra, dec = s2dec(ra,dec)

        temp.append((num, slit, ra, dec,  X,Y,DIMX,DIMY, filename))

    dtype = [('num', 'i4'), ('slit', 'i4'), ('ra','f8'), ('dec','f8'),
             ('x','f8'), ('y','f8'), ('dimx', 'f8'), ('dimy','f8'),
             ('filename', 'S70')]
    return np.rec.fromrecords(temp, dtype=dtype)

def read_adp_header(hd):
    """ Read object ids copied from a vmmps ADP file from the header
    of a pipeline output file."""

    pre = 'HIERARCH ESO INS SLIT%i'
    i = 1
    temp = []
    while (pre % i + ' ID') in hd:        
        num = int(hd[pre % i + ' ID'])
        #stype = hd[pre % i + ' TYPE']
        ra = hd[pre % i + ' OBJ RA']
        dec = hd[pre % i + ' OBJ DEC']
        X = hd[pre % i + ' X']
        Y = hd[pre % i + ' Y']
        DIMX = hd[pre % i + ' DIMX']
        DIMY = hd[pre % i + ' DIMY']
        temp.append((num, ra, dec, X, Y, DIMX, DIMY))
        i += 1

    dtype = [('num', 'i4'), ('ra','f8'), ('dec','f8'),
             ('x','f8'), ('y','f8'), ('dimx', 'f8'), ('dimy','f8')]

    return np.rec.fromrecords(temp, dtype=dtype)


def get_pos_ind(IDtuple):
    """ Get the start and end positions of the slit image, the
    spectrum centre and edges, and index of a spectrum given an
    IDtuple.

    Returns
    -------
    i0, i1, pos, ind
    """
    quad, iext, slit_id, obj = IDtuple
    table = pyfits.getdata('object_sci_table.fits', iext)
    islit = dict((n,i) for i,n in enumerate(table.slit_id))

    itable = islit[slit_id]
    pos = (table[itable]['start_%i' % obj],
           table[itable]['object_%i' % obj],
           table[itable]['end_%i' % obj])
    ind = table[itable]['row_%i' % obj]

    i1 = None
    i0 = table[itable]['position']
    if itable != 0:
        i1 = table[itable-1]['position']

    return i0, i1, pos, ind

def get_object_IDs(filename):
    """ IDs is a list of individual objects. Each entry is the
    quadrant, extension number, slit id and object number.

    Note slit edge y in im are in otable.position. Each slit can have up to
    three objects. the object y centres and edges are in object_1,
    start_1 and end_1 (similar for 2 and 3). Some of these are nan if
    there are less than three objects in the slit.
    """

    fh = pyfits.open(filename)
    quad = fh[0].header['HIERARCH ESO OCS CON QUAD']
    IDs = []
    for iext in range(1, len(fh)):
        otable = fh[iext].data
        nobj = len([n for n in otable.dtype.names if n.startswith('object_')])
        for i in xrange(len(otable)):
            for j in xrange(1, nobj + 1):
                row = otable[i]
                if row['row_%i' % j] != -1:
                    IDs.append( (quad, iext, row['slit_id'], j) )
    return IDs

def get_1d_2d_spectra(IDtuple):
    """ Given a tuple of mask id, quadrant number, slit_id and object
    number, return the 1d extracted flux, error, the 2d image of the
    slit in which the object falls, and the y-coordinate of the objet
    in the slit.

    Note each slit can have several objects (up to 9?), each one has a
    different object number. The object y centres and edges are in
    object_?, start_?  and end_?, where '?' is the object number in
    the slit.

    Returns
    -------
    wa, fl, er, sky, image, (ystart, yobj, yend)
    """
    quad, iext, slit_id, obj = IDtuple
    image = pyfits.getdata('mos_science_flux_extracted.fits', iext)
    fluxes = pyfits.getdata('mos_science_flux_reduced.fits', iext)
    skies = pyfits.getdata('mos_sci_sky_reduced.fits', iext)
    errors = pyfits.getdata('mos_sci_error_flux_reduced.fits', iext)
    hd = pyfits.getheader('mos_science_flux_extracted.fits', iext)

    wa = hd['CRVAL1'] + np.arange(hd['NAXIS1']) * hd['CD1_1']

    i0, i1, pos, ind = get_pos_ind(IDtuple)

    fl = fluxes[ind]
    er = errors[ind]
    sky = skies[ind]

    return wa, fl, er, sky, image[i0:i1, :], pos-i0


def get_1st_order_region(wa, sky, msky, wmin=0, thresh=2, dwlo=230, dwhi=130):
    """ Find the region of a sky spectrum that is significantly
    different to an expected (median) sky spectrum.
    
    wa: wavelength array, length N
    sky: sky array, length N
    msky: median sky array, length N

    Returns the start and end indices of the region to be masked.  Or
    None,None if there is no masked region.
    """
    i0, i1 = None, None
    
    sky = sky * np.median(msky) / np.median(sky)
    iwa = wa.searchsorted(wmin)
    diff = np.abs(msky[iwa:] - sky[iwa:]) / msky[iwa:]
    imax = np.argmax(diff)
    if diff[imax] > thresh:
        i = iwa+imax
        i0, i1 = wa.searchsorted([wa[i] - 230, wa[i] + 130])

    return i0, i1


def apply_distortion(xccd1, yccd1, hd):
    """ Apply the distortion correction from a preimage header to
    preimaging pixel coordinates. This should be done before running
    the WCS transformation.
    """
    xord = hd['HIERARCH ESO PRO CCD SKY XORD'] + 1
    yord = hd['HIERARCH ESO PRO CCD SKY YORD'] + 1
    xccd1, yccd1 = [], []
    for x, y in zip(xccd0, yccd0):
        xnew = 0
        ynew = 0
        for i in xrange(xord):
            for j in xrange(yord):
                aij = float(hd['HIERARCH ESO PRO CCD SKY X_%i_%i' % (i,j)])
                bij = float(hd['HIERARCH ESO PRO CCD SKY Y_%i_%i' % (i,j)])
                # we are swapping the coefficents compared to the
                # order the manual tells us to use!
                xnew += bij * x**i * y**j
                ynew += aij * x**i * y**j

        xccd1.append(xnew)
        yccd1.append(ynew)
    return xccd1, yccd1


#NT code from here.
def median_MAD(a,m=1.4826):
    """Median absolute deviation"""
    import numpy as np
    median = np.median(a)
    error  = np.fabs(a-median)
    error  = np.median(error)
    return m*error

def get_D4000_break(wa,fl,er,z):
    """Gets the 4000A break (with associated error) as the ratio
    between the two continuum regions (at both sides), for a given
    galaxy spectrum (wa,fl,er) observed at a given redshift
    (z). Returns a tuple with ratio and associated error."""
    bands = {'D4000': (3750,3950,4050,4250)}
    Nmin = 15. #minimum number of pixels in continuum bands required (has to be >0).
    
    for key in bands.keys():
        cond1 = (wa >= bands[key][0]*(1+z)) & (wa <= bands[key][1]*(1+z))
        cond2 = (wa >= bands[key][2]*(1+z)) & (wa <= bands[key][3]*(1+z))
        cs1 = (wa[cond1]>=7550)&(wa[cond1]<=7700) #sky absorption
        cs2 = (wa[cond2]>=7550)&(wa[cond2]<=7700) #sky absorption
        
        if  (np.sum(cond1)<Nmin) | (np.sum(cond2)<Nmin) | (z<0):#line
                                                        #too close to
                                                        #spectrum edge
                                                        #or gap or no
                                                        #redshift
            
            ratio     = -99.  # no measurement
            ratio_err = -99.
            
        elif (np.sum(cs1)>0)|(np.sum(cs2)>0):
            ratio     = -99.  # no measurement
            ratio_err = -99.
            
            
        else:
            clvl1 = np.median(fl[cond1]) # continuum level 1
            clvl2 = np.median(fl[cond2]) # continuum level 2
            ratio = clvl2/clvl1
            
            #error estimate
            clvl1_err = np.std(fl[cond1])
            clvl2_err = np.std(fl[cond2])
            
            ratio_err = (1./clvl1)**2 * clvl2_err**2 + (clvl2/clvl1/clvl1)**2 * clvl1_err**2
            ratio_err = np.sqrt(ratio_err)
            print key, ratio, ratio_err
    return ratio, ratio_err
    
    
def get_flux_EW_bands(wa,fl,er,z):
    """Gets the flux and equiv. widths (with associated error) of the
    bands (defined inside this function) for a given galaxy spectrum
    (wa,fl,er) observed at a given redshift (z). Return 2 dictionaries,
    1 for fluxes and one for equivalent widths respectively."""
    import pylab as pl
    bands = {'OII':  (3645,3715,3715,3740,3740,3810),
             'Hd':   (4010,4080,4085,4120,4120,4190),
             'Hc':   (4200,4270,4320,4360,4360,4430),
             'Hb':   (4760,4830,4830,4900,5030,5100),
             'OIII': (4760,4830,4990,5030,5030,5100),
             'Ha':   (6460,6530,6530,6610,6610,6680),
             'SII':  (6625,6695,6695,6760,6760,6830)}   
        
    
    maskatmos = [(1, WMIN),
             (5870, 5910),
             (6275, 6325),
             (6350, 6385),
             (7550, 7720),
             (WMAX, 1e9)]
    
    Nmin = 5. #minimum number of pixels in continuum bands required (has to be >0).
    flux  = {}
    EW    = {}
    for key in bands.keys():
        #estimating the continuum level around the line
        cond1 = (wa >= bands[key][0]*(1+z)) & (wa <  bands[key][1]*(1+z)) #blue cont
        condl = (wa >= bands[key][2]*(1+z)) & (wa <= bands[key][3]*(1+z)) #feature
        cond2 = (wa >= bands[key][4]*(1+z)) & (wa <  bands[key][5]*(1+z)) #red cont
        csf   = (wa[condl]>=7550)&(wa[condl]<=7720)  #sky absprption
        cs1   = (wa[cond1]>=7550)&(wa[cond1]<=7720)
        cs2   = (wa[cond2]>=7550)&(wa[cond2]<=7720)
        if  (np.sum(cond1)<Nmin)|(np.sum(cond2)<Nmin)|(z<0):#line
                                                        #too close to
                                                        #spectrum edge
                                                        #or gap or no
                                                        #redshift
            
            Fl     = -99.  # no measurement
            Wl     = -99.  
            Fl_err = -99.
            Wl_err = -99.
        
        elif (np.sum(csf)>0)|(np.sum(cs1)>0)|(np.sum(cs2)>0):
            Fl     = -99.  # no measurement
            Wl     = -99.  
            Fl_err = -99.
            Wl_err = -99.
            
        
        else: # line well within the spectrum
            clvl1 = np.median(fl[cond1]) # continuum level 1
            clvl2 = np.median(fl[cond2]) # continuum level 2
            cont  = (clvl2 - clvl1)/(np.mean(wa[cond2])-np.mean(wa[cond1]))*(wa-np.mean(wa[cond1]))+clvl1 #linear interpolation 
            
            #estimating the line flux & equiv. width 
            dl    = np.mean(wa[condl][1:]-wa[condl][:-1]) #mean pixel size [wavelen. units]
            n     = np.sum(condl) # number of pixels for the line
            Fl    = (np.sum(fl[condl] - cont[condl]))*dl  #integrated (flux - continuum)
            Wl    = (np.sum(1-fl[condl]/cont[condl]))*dl  #observed equiv. width (<0 for emission)
            Fl    = Fl / (1.+z) #rest frame 
            Wl    = Wl / (1.+z) #rest frame
            
            #error estimation (neglecting z error)
            clvl1_err = median_MAD(fl[cond1])
            clvl2_err = median_MAD(fl[cond2])
            cont_err  = np.sqrt(clvl1_err**2 + clvl2_err**2)/2.
            Fl_err    = np.sum(er[condl]**2) + (n*cont_err)**2
            Fl_err    = np.sqrt(Fl_err)*dl/(1.+z)
            Wl_err    = np.sum((er[condl]/cont[condl])**2) + np.sum((fl[condl]/(cont[condl]**2))**2)*cont_err**2
            Wl_err    = np.sqrt(Wl_err)*dl/(1.+z)
                    
            #print key,Fl,Fl_err,Wl,Wl_err
        flux[key] = (Fl,Fl_err)
        EW[key]   = (Wl,Wl_err)
    return flux, EW


if __name__ == '__main__':
    IDs = get_object_IDs('object_sci_table.fits')

    fig = pl.figure(figsize=(13, 5))
    fig.subplots_adjust(right=0.99, top=0.98, left=0.05, bottom=0.1)
    mediansky = np.median(pyfits.getdata('mos_sci_sky_reduced.fits', 1), 0)
    for i in range(len(IDs)):
        wa, fl, er, sky, im, pos = get_1d_2d_spectra(IDs[i])
        print i, IDs[i], pos
     
        pl.cla()
        pl.plot(wa, fl)
        pl.plot(wa, er)
        pl.plot(wa, sky)
        fmax = np.abs(1.5*np.percentile(fl, 98))
        Y = np.arange(im.shape[0]) / im.shape[0] * 0.2*fmax + 1.2*fmax
        pl.ylim(-0.1*fmax, 1.4*fmax)
        vmax = np.percentile(im.ravel(), 98)
        pl.pcolormesh(wa, Y, im, vmin=-0.2*vmax, vmax=2*vmax, cmap=pl.cm.hot)
        pl.axhline(1.2*fmax + 0.2*fmax*(pos[0]/im.shape[0]),
                   color='LawnGreen', lw=0.5)
        pl.axhline(1.2*fmax + 0.2*fmax*(pos[2]/im.shape[0]),
                   color='LawnGreen', lw=0.5)
        pl.show()
        pl.waitforbuttonpress()

