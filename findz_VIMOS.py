#!/usr/bin/env python
import pyfits
import numpy as np
import matplotlib.pyplot as pl

from astro.io import saveobj
from astro.utilities import between, adict
from astro.convolve import convolve_psf
from astro.spec import find_cont
import astro.sed as sed

from VIMOS_util import \
     get_object_IDs, get_1st_order_region, get_1d_2d_spectra, \
     WMIN, WMAX, MASKATMOS

from glob import glob
import sys, os

PLOT = 0

np.seterr(all='ignore')

# regions to mask when doing cross correlation; atmospheric lines and
# fringing.

LOG10_MASKATMOS = np.log10(MASKATMOS)

def prepare_templates(templates, convolve_pix=None, plot=0):
    """ Add continua to templates and interpolate to log-linear wav
    scale.  Returns a new set of templates.
    """
    tnew = []
    for t in templates:
        # fudge to remove bad template areas
        print t.label
        wgood = t.wa[t.fl > 1e-99]
        w0,w1 = wgood[0], wgood[-1]
        cond = between(t.wa, w0, w1)
        wa0, fl0 = t.wa[cond], t.fl[cond]
        if plot:
            ax = pl.gca()
            ax.cla()
            pl.plot(wa0, fl0, 'b')
        # rebin to a log-linear wavelength scale. To shift wavescale
        # to redshift z, add log10(1+z) to wa or twa.
        dw = (np.log10(wa0[-1]) - np.log10(wa0[0])) / (len(wa0) - 1)
        logtwa = np.log10(wa0[0]) + dw * np.arange(len(wa0))
        twa = 10**logtwa
        tfl = np.interp(twa, wa0, fl0)
        if t.label.startswith('sdss') and convolve_pix is not None:
            # convolve to VIMOS resolution
            tfl = convolve_psf(tfl, convolve_pix)
        fwhm1 = len(wa0) // 7
        fwhm2 = len(wa0) // 9
        #print fwhm1, fwhm2
        tco = find_cont(tfl, fwhm1=fwhm1, fwhm2=fwhm2)
        if plot:
            pl.plot(twa, tfl, 'r')            
            pl.plot(twa, tco, 'r--')
            pl.show()
            raw_input()

        t1 = adict(logwa=logtwa, fl=tfl, co=tco, label=t.label)
        tnew.append(t1)

    return tnew

#@profile
def xcorr_template(wa, fl, er, co, logtwa, tfl, tco, redshifts, plot=False):
    """ Cross-correlate a template with a spectrum for a series of
    redshifts. Return the value of cross correlation at each
    redshift. Peaks in the xcorr denote matches between the spectrum
    and template.
    """
    logwa = np.log10(wa)
    wrange = WMAX - WMIN
    if plot:
        fig = pl.gcf()
        pl.clf()
        ax = fig.add_subplot(111)
        artist_template, = ax.plot(wa, fl, 'b',alpha=0.5)
        artist_spec, = ax.plot([], [], 'r', alpha=0.5)
        artist_templateco, = ax.plot([], [], 'b--',alpha=0.5)
        artist_specco, = ax.plot([], [], 'r--')
        artist_line1, = ax.plot([], [], 'k:')
        artist_line2, = ax.plot([], [], 'k:')
        for w0,w1 in MASKATMOS:
            ymin,ymax = ax.get_ylim()
            ax.fill([w0,w0,w1,w1], [ymin,ymax,ymax,ymin], color='y', alpha=0.3)

    wshifts = np.log10(np.asarray(redshifts) + 1)
    xcorr = []
    nchi2vals = []
    masked = np.empty(len(logtwa), bool)
    for wshift in wshifts:
        #print 'z', 10**wshift -1
        logtwa_z = logtwa + wshift
        # In principle we could do this loop just once for a given set of
        # templates and redshifts, instead of once for each spectrum.
        masked.fill(0)
        i0 = 0
        for w0,w1 in LOG10_MASKATMOS:
            i,j = i0 + logtwa_z[i0:].searchsorted([w0, w1])
            masked[i:j] = True
            i0 = j

        if masked.all():
            xcorr.append(0)
            continue
        notmasked = ~masked
        
        fl0 = np.interp(logtwa_z, logwa, fl)
        er0 = np.interp(logtwa_z, logwa, er)
        co0 = np.interp(logtwa_z, logwa, co)
        # find the common overlapping wavelength range
        logtwa_z_notmasked = logtwa_z[notmasked]
        logwmin = max(logtwa_z_notmasked[0], logwa[0])
        logwmax = min(logtwa_z_notmasked[-1], logwa[-1])
        good = notmasked & between(logtwa_z, logwmin, logwmax) & (er0 > 0)
        if good.sum() == 0:
            xcorr.append(0)
            continue            
        # weight by (approx) good wavelength range over (approx) total
        # available wavelength range
        lw = logtwa_z[good]
        weight = (10**lw[-1] - 10**lw[0]) / wrange 
        assert not np.isnan(weight)
        assert 0 <= weight <= 1, '%.3f, %.1f - %.1f' % (weight,10**lw[0],10**lw[-1])
        if weight < 0.1:
            #print 'z=%.3f' % (10**wshift - 1), t.label, 'Less than 10% overlap'
            xcorr.append(0)
            continue

        fl1 = fl0[good]
        co1 = co0[good]
        er1 = er0[good]
        tfl1 = tfl[good]
        tco1 = tco[good]
        # scale template to match spectrum and then subtract continua
        mult = np.median(fl1) / np.median(tfl1)
        fl2 = fl1 - co1
        tfl2 = (tfl1 - tco1)* mult
        if plot:
            w = 10**logtwa_z
            artist_template.set_data(w, tfl*mult)
            artist_spec.set_data(w, fl0)
            artist_templateco.set_data(w, tco*mult)
            artist_specco.set_data(w, co0)
            artist_line1.set_data([10**logwmin]*2, [ymin, ymax])
            artist_line2.set_data([10**logwmax]*2, [ymin, ymax])

        val = (fl2 * tfl2).sum() / np.sqrt((tfl2**2).sum() * (fl2**2).sum())
        assert not np.isnan(val)
        resid = (fl1 - tfl1*mult)/er1
        # chi2 per deg freedom
        nchi2 = np.dot(resid, resid) / len(resid)
        #print nchi2
        if plot:
            ax.set_title('%.4f %.3f' % (val,nchi2))
            ax.set_xlim(5000, 9500)
            ymax = np.percentile(fl1, 95)
            #ax.plot(10**logtwa_z[good], -0.1*ymax + 0.05*ymax*resid, '.k',alpha=0.5)
            ax.set_ylim(-0.2*ymax, 1.5*ymax)
            pl.show()
            raw_input('z=' + str(10**wshift - 1) + ', %i points' % good.sum())

        xcorr.append(val * weight)
        nchi2vals.append(nchi2)

    return np.array(xcorr), np.array(nchi2vals)


if 1:
    #####################################
    # Prepare templates
    #####################################

    temp0 =  sed.get_SEDs('sdss') #+ sed.get_SEDs('LBG')
    # remove BAL and high_lum qsos
    temp1 = [t for t in temp0 if
             'qsoBAL' not in t.label and 'highlum' not in t.label]

    # note sdss templates have resolution of 2000 or 150 km/s, VIMOS
    # spectra have resolution of ~210 or 1428 km/s. So convolve sdss
    # templates with Gaussian of width sqrt(1428^2 - 150^2) = 1420 km/s,
    # or 20.6 pixels (sdss pixels are 69 km/s).
    #
    # (20.6 seems too big, do 10 instead)
    #
    templates = prepare_templates(temp1, convolve_pix=10, plot=PLOT)
    saveobj('templates.sav', templates, overwrite=1)

    #######################################
    # Do xcorr
    #######################################
    IDs = get_object_IDs('object_sci_table.fits')

    if not os.path.lexists('xcorr'):
        os.mkdir('xcorr')

    # median sky
    msky = np.median(pyfits.getdata('mos_sci_sky_reduced.fits', 1), axis=0)
    msky[msky == 0] = np.median(msky)

    if PLOT:
        pl.figure(figsize=(10, 6))

    for ind,ID in enumerate(IDs[:5]):
        print '%i of %i, cross-correlating %s with:' % (ind+1, len(IDs), ID)

        wa, fl, er, sky, im, pos = get_1d_2d_spectra(ID)
        
        # mask regions where the extracted sky differs strongly from
        # the median extracted sky (this is usually a first-order
        # line)
        i0,i1 = get_1st_order_region(wa, sky, msky, wmin=WMIN)
        if i0 is not None:
            print '  Masking 1st order region'
            er[i0:i1] = 0

        # fwhm1 and fwhm2 need to be tweaked depending on resolution,
        # size, etc
        sp_co = find_cont(fl, fwhm1=200, fwhm2=150)
        if PLOT:
            pl.cla()
            plot(wa, fl, wa, sp_co, wa, 50*er)
            mult = np.median(fl) / np.median(msky)
            plot(wa, msky *mult, wa, sky*mult)
            pl.show()
            raw_input('')
        xcorrs = []
        nchi2 = []
        redshifts = []
        for t in templates:
            # find xcorr
            tname = t.label.split('/')[-1][:-4]

            if tname.startswith('star'):
                z = [0]
            elif tname.startswith('gal'):
                z = np.arange(0, 1.5, 0.001)
            elif tname.startswith('lbg'):
                z = np.arange(1.5, 4.0, 0.001)
            elif tname.startswith('qso'):
                z = np.arange(0.5, 4.0, 0.001)
            if not tname.startswith('star'):
                print  '  %s  %.2f < z < %.2f' % (t.label, min(z), max(z))
            xcorr,nchi = xcorr_template(wa, fl, er, sp_co,
                                        t.logwa, t.fl, t.co, z, plot=PLOT)
            nchi2.append(nchi)
            xcorrs.append(xcorr)
            redshifts.append(z)

        name = ('xcorr/q%i_e%i_s%04i_o%i_xcorr.sav' % ID)
        print 'Saving',  name
        saveobj(name, adict(z=redshifts, xcorr=xcorrs, nchi2=nchi2),
                overwrite=1)
