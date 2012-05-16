from __future__ import division
import pyfits
import numpy as np
import matplotlib.pyplot as pl

import astro
from astro.fit import scale_by_median
from astro.io import readtxt, loadobj
from astro.utilities import between
from astro.convolve import convolve_psf
from astro.spec import plotlines
from astro.plot import axvfill
import Tkinter as tkinter

from glob import glob
import sys, os
from subprocess import call

from VIMOS_util import \
     get_1d_2d_spectra, get_1st_order_region, WMIN, WMAX, MASKATMOS

np.seterr(all='ignore')

#CMAP = pl.cm.gray
CMAP = pl.cm.hot

WMIN_PLOT = 5500
WMAX_PLOT = 9500

Ckms = 299792.458         # speed of light km/s, exact

LINES = readtxt(astro.datapath + 'linelists/galaxy_lines',
                names='wa,name,select')

def measure_nchi2(twa, tfl, wa, fl, er):
    masked = np.zeros(len(twa), bool)
    i0 = 0
    for w0,w1 in MASKATMOS:
        i,j = i0 + twa[i0:].searchsorted([w0, w1])
        masked[i:j] = True
        i0 = j

    fl1 = np.interp(twa, wa, fl)
    er1 = np.interp(twa, wa, er) 

    resid = (fl1[~masked] - tfl[~masked]) / er1[~masked]
    return np.dot(resid, resid) / len(twa)


class FindZWrapper(object):
    """ A wrapper for the figure that displays a fit."""
    def __init__(self, redshifts, vals, nchi2, templ, spec,
                 objname, fig, spec2d=None, wa2d=None, spec2dpos=None,
                 msky=None):
        """ Remember the fit info. And initiate all the plots."""

        self.__dict__.update(locals())
        self.__dict__.pop('self')

        fig.clf()
        self.valsmax, self.zatmax = [], []
        for i in range(len(vals)):
            ind = np.argmax(vals[i])
            self.valsmax.append(vals[i][ind])
            self.zatmax.append(redshifts[i][ind])

        ind = np.argmax(self.valsmax) 
        self.i = ind
        
        bot1, height1 =  0.12, 0.19
        bot2, height2 =  bot1 + height1 + 0.05, 0.09
        bot3, height3 =  bot2 + height2 + 0.04, 0.33
        bot4, height4 =  bot3 + height3 + 0.001, 0.13
        width, left = 0.96, 0.02

        # disable any existing key press callbacks
        cids = list(fig.canvas.callbacks.callbacks['key_press_event'])
        for cid in cids:
            fig.canvas.callbacks.disconnect(cid)

        self.cid = fig.canvas.mpl_connect('key_press_event', self.on_keypress)

        # set up the list of emission/absorption lines
        keys = '1 2 3 4 5 6 7 8 9 0 f1 f2 f3 f4 f5'.split()
        lines_sel = LINES[LINES.select == 1]
        self.lines_sel = lines_sel
        lines = [(keys[i], l.name,l.wa) for i,l in enumerate(lines_sel)]

        
        helpmsg = []
        for key,ion,wa in lines:
            wa = '%.0f' % wa
            helpmsg.append('%s: %s %s\n' % (key, ion, wa))
        helpmsg.append(""" 
?: Print this message
s: Smooth (toggle)

k: Skip without saving results
m: Maybe
y: Accept
n: Reject

space: new template or redshift
x: new redshift (don't move to nearest xcorr peak)
""")
        self.help = ''.join(helpmsg)
        self.lines = LINES
        self.keys = keys

        ax0 = fig.add_axes((left, bot1, width, height1))
        ax1 = fig.add_axes((left, bot2, width, height2))
        ax2 = fig.add_axes((left, bot3, width, height3))
        ax3 = fig.add_axes([left, bot4, width, height4], sharex=ax2)
        self.ax = [ax0, ax1, ax2, ax3]

        for ax in [ax1, ax2, ax3]:
            ax.minorticks_on()

        # plot the maximum xcorr vals for each template
        plot_xcorr(ax0, self.valsmax, [t.label for t in self.templ],
                   self.zatmax)

        z = self.zatmax[ind]
        # if it's a galaxy template, find the nearest xcorr peak
        if len(self.vals[ind]) > 1:
            z = self.find_xcorr_peak(z)

        ax1.axhline(0, color='0.7')
        self.art_fit, = ax1.plot([],[], 'k', lw=1.5, zorder=6)
        if len(vals[ind]) > 1:
            self.art_vals, = ax1.plot(redshifts[ind], vals[ind], color='0.7')
            #nchi = np.log10(nchi2[ind]) 
            #self.art_nchi, = ax1.plot(redshifts[ind], nchi, 'r',lw=1)
            ymax = vals[ind].max()
            ax1.set_ylim(-0.5*ymax, 1.2*ymax)
            ax1.set_xlim(redshifts[ind][0], redshifts[ind][-1])
        else:
            self.art_vals, = ax1.plot([0], vals[ind], color='0.7')
            ax1.set_ylim(-0.5, 1)
            ax1.set_xlim(redshifts[ind][0] - 0.01, redshifts[ind][0] + 0.01)
            #self.art_nchi, = ax1.plot([0], np.log10(nchi2[ind]), 'r')
        ymin, ymax = ax1.get_ylim()
        self.art_zline, = ax1.plot([z,z], [ymin, ymax], color='r',lw=2)

        self.plot_spec2d()

        # plot the template with highest xcorr value.
        ax2.set_autoscale_on(0)
        t = templ[ind]
        twa = 10**t.logwa * (1 + z)
        self.smoothed = False

        self.plot_spec_templ(twa, t.fl)
        title = '%s, z=%.4f, %s' % (self.objname, z, self.templ[ind].label)
        self.title = self.fig.suptitle(title)
        self.art_lines1 = None
        self.plotlines(z+1)

        self.art_highlight, = ax0.plot([ind], self.valsmax[ind], 'o', ms=15, mew=5,
                                       mec='orange', mfc='None', alpha=0.7)
        
        for ax in self.ax:
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_visible(False)

        self.z = z

        self.wait = True
        self.zgood = None
        ax2.set_xlim(WMIN_PLOT, WMAX_PLOT)
        self.fig.canvas.draw()
        print self.help

    def plot_spec2d(self):
        if self.spec2d is None:
            return
        self.ax[3].cla()
        i = min(self.spec2d.shape[0]//2, 3)
        v1 = np.percentile(self.spec2d[i:-i, :].ravel(), 90)
        wdelt = self.wa2d[1] - self.wa2d[0]
        yvals = np.arange(self.spec2d.shape[0])*wdelt
        self.ax[3].pcolormesh(
            self.wa2d, yvals,
            self.spec2d, vmin=-v1/5, vmax=2*v1, cmap=CMAP)

        self.ax[3].axhline(wdelt * self.spec2dpos[0], ls='--', color='LawnGreen')
        self.ax[3].axhline(wdelt * self.spec2dpos[2], ls='--', color='LawnGreen')

        self.ax[3].set_ylim(yvals[0], yvals[-1])


    def update(self, fitpeak=True):
        """ Update the figure after changing either the template
        (self.i) or template redshift (self.z).
        """
        i, z = self.i, self.z
        
        self.art_highlight.set_data(i, self.valsmax[i])
        self.art_zline.set_xdata(z)
        self.fig.canvas.draw()

        if len(self.vals[i]) > 1:
            if fitpeak:
                self.z = self.find_xcorr_peak(z)
                self.art_zline.set_xdata(self.z)

            ymax = self.vals[i].max()
            self.ax[1].set_ylim(-0.5*ymax, 1.5*ymax)
        else:
            self.ax[1].set_ylim(-0.5, 1.5)

        self.plotlines(self.z+1)

        # plot the new template (or same template at different z)
        t = self.templ[i]
        twa = 10**t.logwa * (1 + self.z)
        sp = self.spec
        
        mult = scale_by_median(twa, t.fl, sp.wa, sp.fl, mask=MASKATMOS)
        if mult is None:
            tfl = []
            twa = []
        else:
            tfl = t.fl * mult
        
        #nchi2 = measure_nchi2(twa, tfl, sp.wa, sp.fl, sp.er)
        self.art_templ2.set_data(twa, tfl)
    
        self.title.set_text('%s, z=%.4f, %s' % (self.objname, z, self.templ[i].label))
        self.art_vals.set_data([self.redshifts[i], self.vals[i]])

        #nchi = np.log10(self.nchi2[i])
        #self.art_nchi.set_data([self.redshifts[i], nchi])


        z0, z1 = self.redshifts[i][0], self.redshifts[i][-1]
        self.ax[1].set_xlim(z0 - 0.01, z1 + 0.01)
        self.ax[2].set_xlim(WMIN_PLOT, WMAX_PLOT)

    def find_xcorr_peak(self, z):
        """ Find a xcorr peak."""
        redshifts = np.asarray(self.redshifts[self.i])
        if len(redshifts) == 1:
            return
        i0,ic,i1 = redshifts.searchsorted([z-0.01, z, z+0.01])

        vals = self.vals[self.i]
        j = ic
        while True:
            if j <= 1 or j >= (len(vals) - 2):
                break
            sleft = np.sign((vals[j] - vals[j-1])/
                            (redshifts[j] - redshifts[j-1]))
            sright = np.sign((vals[j+1] - vals[j])/
                             (redshifts[j+1] - redshifts[j]))

            if sleft != sright:
                break
            elif sleft > 0:
                j += 1
            elif sleft < 0:
                j -= 1
            else:
                break

        j = max(0, j)
        j = min(len(vals)-1, j)
        return redshifts[j]

    def plotlines(self, zp1):
        """ Plot the positions of expected lines given a redshift
        """
        if self.art_lines1:
            for artist in self.art_lines1:
                try:
                    artist.remove()
                except ValueError:
                    # axes have been removed since we added these lines
                    break
        self.art_lines1 = plotlines(
            zp1-1, self.ax[2], lines=self.lines, labels=1, ls='dotted', color='0.3',
            atmos=False)

    def on_keypress(self, event):
        if event.key == '?':
            print self.help
        elif event.key == 'y':
            print 'Redshift = %.3f' % self.z
            self.zgood = self.z
            self.zconf = 'a'
            self.disconnect()
        elif event.key == 'm':
            print 'Maybe redshift = %.3f' % self.z
            self.zgood = self.z
            self.zconf = 'b'
            self.disconnect()
        elif event.key == 'n':
            print 'Rejecting, z set to -1'
            self.zgood = -1.
            self.zconf = 'c'
            self.disconnect()
        elif event.key == 'k':
            print 'Skipping, not writing anything'
            self.zconf = 'k'
            self.disconnect()
        elif event.key == 's':
            if self.smoothed:
                self.art_spec2.set_data(self.spec.wa, self.spec.fl)
                self.smoothed = False
            else:
                fl = convolve_psf(self.spec.fl, 5)
                self.art_spec2.set_data(self.spec.wa, fl)
                self.smoothed = True
            self.fig.canvas.draw()
        elif event.inaxes == self.ax[1]:
            if event.key == ' ':
                self.z = event.xdata
                self.update()
            elif event.key == 'x':
                self.z = event.xdata
                self.update(fitpeak=False)            
        elif event.inaxes == self.ax[0]:
            if event.key == ' ':
                i = int(np.round(event.xdata))
                self.i = min(max(i, 0), len(self.zatmax)-1)
                self.z = self.zatmax[self.i]
                self.update()
        elif event.inaxes == self.ax[2]:
            try:
                j = self.keys.index(event.key)
            except ValueError:
                return
            line = self.lines_sel[j]
            zp1 = event.xdata / line['wa']
            #print 'z=%.3f, %s %f' % (zp1 - 1, line['name'], line['wa'])
            self.z = zp1 - 1
            self.update()

    def disconnect(self):
        t = self.title.get_text()
        self.title.set_text(t + ' ' + self.zconf)
        self.fig.canvas.mpl_disconnect(self.cid)
        self.wait = False

    def plot_spec_templ(self, twa, tfl):
        """ Initiate plots of the spectrum and a template, scaled to
        matched the spectrum.
        """
        sp = self.spec

        ax2 = self.ax[2]
        ax2.axhline(0, color='0.7')
        if self.smoothed:
            self.art_spec2, = ax2.plot(sp.wa, convolve_psf(sp.fl,5), 'k',lw=0.5,ls='steps-mid')
        else:
            self.art_spec2, = ax2.plot(sp.wa, sp.fl, 'k', lw=0.5, ls='steps-mid')
        ax2.plot(sp.wa, sp.er, 'k', lw=0.25)
        good = sp.er > 0
        if (~good).any():
            ax2.plot(sp.wa[~good], sp.er[~good], 'k.', ms=4)
        mult = scale_by_median(twa, tfl, sp.wa, sp.fl, mask=MASKATMOS)
        if mult is None:
            self.art_templ2, = ax2.plot([], [], color='r', alpha=0.7)
        else:
            self.art_templ2, = ax2.plot(twa, mult*tfl, color='r', alpha=0.7)

        ymax = np.abs(np.percentile(sp.fl[good & (sp.wa>5700)], 98))
        offset = 0.15 * ymax
        temp = max(np.median(sp.fl), np.median(sp.er))
        sky = 0.15 * temp / np.median(sp.sky)*sp.sky - offset
        axvfill(MASKATMOS, ax=ax2, color='y', alpha=0.3)
        ax2.fill_between(sp.wa, sky, y2=-offset, color='g', lw=0, alpha=0.3, zorder=1)
        #pdb.set_trace()
        if self.msky is not None:
            s = self.msky
            msky = s * np.median(sky + offset) / np.median(s)
            ax2.plot(sp.wa, msky - offset, color='g', lw=1.5, zorder=1)

        ax2.set_ylim(-ymax*0.15, 1.5*ymax)
        #pl.legend(frameon=0, loc='upper left')


def plot_xcorr(ax, vals, labels, z):
    """ Plot the redshifts and xcorr values for a spectrum with
    respect to a range of templates.

    ax: mpl axis
    vals: xcorr values (1 = correlated)
    labels: template name.
    z: redshifts corresponding to the xcorr values. Ignored for stars.
    """
    #print labels, z
    vals, labels, z = (np.asarray(a) for a in (vals, labels, z))
    labels = np.array([l.replace('.fits', '').replace('.dat', '') for l in labels])
    star = np.array([l.startswith('sdss/star') for l in labels])
    gal = ~star
    ax.set_autoscale_on(0)
    xvals = np.arange(len(vals))
    #import pdb; pdb.set_trace()
    ax.plot(xvals[star], vals[star], 'ok')
    ax.plot(xvals[gal], vals[gal], 'or')
    for yval in 0,1,-1:
        ax.axhline(yval, color='k', ls='dotted')
    ax.set_xlim(-1, xvals.max() + 1)
    ax.set_ylim(-0.29, 1.09)
    for i,xval in enumerate(xvals[gal]):
        ax.text(xval, vals[gal][i] + 0.4, z[gal][i],
                ha='center', fontsize=10, rotation=90)

    ax.set_xticks(xvals)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)


def interactive_fit(redshifts, vals, nchi2, templ, sp, outfile, filename,
                    fig, spec2d=None, wa2d=None, spec2dpos=None, msky=None):
    fig.clf()
    objname = filename.split('/')[-1].replace('_xcorr.sav', '')
    wrapper = FindZWrapper(redshifts, vals, nchi2, templ, sp, objname, fig=fig,
                           spec2d=spec2d, wa2d=wa2d, spec2dpos=spec2dpos,
                           msky=msky)

    # wait until user decides on redshift
    try:
        while wrapper.wait:
            pl.waitforbuttonpress()
    except (tkinter.TclError, KeyboardInterrupt):
        print "\nClosing\n"
        sys.exit(1)

    if wrapper.zconf == 'k':
        # skip
        return
    print filename
    plotname = filename.replace('_xcorr.sav','.png')
    print 'saving to ', plotname
    fig.savefig(plotname)

    outfile.write('%20s %10s % 8.5e %1s\n' % (
        filename.split('/')[-1].split('_xcorr')[0], templ[wrapper.i].label,
        wrapper.zgood, wrapper.zconf))
    outfile.flush()

if 1:
    templates = loadobj('templates.sav')
    # if output file exists, rename the existing file so we don't
    # overwrite it.
    filenames = sorted(glob('xcorr/*xcorr.sav'))
    outfilename = 'findz_plot.out'
    newname = outfilename
    rename = 0
    while os.path.lexists(newname):
        rename += 1
        print '%s exists' % newname 
        newname = outfilename  + '.' + str(rename) 
    if rename:
        print "Renaming %s to %s" % (outfilename, newname)
        call('mv %s %s' % (outfilename, newname), shell=1)

    print 'Opening output file %s' % outfilename
    outfile =  open(outfilename, 'w')

    msky = np.median(pyfits.getdata('mos_sci_sky_reduced.fits', 1), axis=0)
    msky[msky == 0] = np.median(msky)

    #process all spectra:
    fig = pl.figure(figsize=(15.2, 9))
    for filename in filenames:

        plotname = filename.replace("_xcorr.sav", ".png")
        if os.path.lexists(plotname):
            c = raw_input("%s exists, redo it? (n) " % plotname)
            if c.lower() != 'y':
                continue

        name = filename.split('/')[-1]
        ID = tuple(int(val[1:]) for val in name.split('_')[:4])
        wa, fl, er, sky, im, pos = get_1d_2d_spectra(ID)

        for a in wa, fl, er, sky, im:
           if np.isnan(a).any():
               a[np.isnan(a)] = 0
            
        i0,i1 = get_1st_order_region(wa, sky, msky, wmin=WMIN)
        if i0 is not None:
            print 'masking 1st order region...'
            er[i0:i1] = 0

        print 'Loading results for', filename
        results = loadobj(filename)
        sp = np.rec.fromarrays([wa, fl, er, sky], names='wa,fl,er,sky')

        print filename.split('/')[-1][:-10]
        interactive_fit(
            results.z, results.xcorr, results.nchi2, templates, sp,
            outfile, filename, fig, spec2d=im, wa2d=wa, spec2dpos=pos,
            msky=msky)
