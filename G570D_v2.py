## G570D Benchmark Retrieval 

#!/usr/bin/env python

"""This is Brewster: the golden retriever of smelly atmospheres"""
from __future__ import print_function

import multiprocessing
import time
import numpy as np
import scipy as sp
import emcee
import testkit
import ciamod
import TPmod
import settings
import os
import gc
import sys
import pickle
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from schwimmbad import MPIPool

__author__ = "Ben Burningham"
__copyright__ = "Copyright 2015 - Ben Burningham"
__credits__ = ["Ben Burningham"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Ben Burningham"
__email__ = "burninghamster@gmail.com"
__status__ = "Development"

# Run Name
runname = "G570D_1205" #EDITED ----------------

# Read observed spectrum as text file w/ columsns: wl (microns), flux (w/m2/um), flux error
obspec = np.asfortranarray(np.loadtxt("G570D_2MassJcalib.dat",dtype='d',unpack='true')) 

# Wavelength Range
w1 = 1.0 
w2 = 2.5

# FWHM of data 
fwhm = 0.0 

# DISTANCE (in parsecs)
dist = 5.84 

# Patches and Clouds! --------------------------- 
# Must be at least 1 of each, but can turn off cloud below
npatches = 1
nclouds = 1 

# set up array for setting patchy cloud answers
do_clouds = np.zeros([npatches], dtype='i')

# Which patchdes are cloudy
do_clouds[:] = 0 

# set up cloud detail arrays
cloudnum = np.zeros([npatches, nclouds], dtype='i')
cloudtype = np.zeros([npatches, nclouds], dtype='i')

# Now fill cloud details. What kind of clouds and shape are they?
# Cloud types
# 1:  slab cloud
# 2: deep thick cloud , we only see the top
# 3: slab with fixed thickness log dP = 0.005 (~1% height)
# 4: deep thick cloud with fixed height log dP = 0.005
# In both cases the cloud properties are density, rg, rsig for real clouds
# and dtau, w0, and power law for cloudnum = 89 or 99 for grey
# See cloudlist.dat for other cloudnum

cloudnum[:,0] = 5
cloudtype[:,0] = 1
# cloudnum[:,1] = 2 
# cloudtype[:,1] = 2 

# second patch turn off top cloud
#cloudnum[1,0] = 5
#cloudtype[1,0] = 1

# Are we assuming chemical equilibrium, or similarly precomputed gas abundances?
# Or are we retrieving VMRs (0)
chemeq = 0

# Are we doing H- bound-free, free-free continuum opacities?
# (Is the profile going above 3000K in the photosphere?)
do_bff = 0

# Profile Type --------------------
# Type 1 is the knots for a spline
# Type 2 is a Madhusudhan & Seager 2009 parameterised profile, no inversion
# i.e. T1,P1 == T2,P2
# Type 3 is Madhusudhan & Seager 2009 with an inversion
proftype = 1
pfile = "t1700g1000f3.dat"

# set up pressure grids in log(bar) cos its intuitive
logcoarsePress = np.arange(-4.0, 2.5, 0.53)
logfinePress = np.arange(-4.0, 2.4, 0.1)

# but forward model wants pressure in bar
coarsePress = pow(10,logcoarsePress)
press = pow(10,logfinePress)

# Cross Sections -----------------
xpath = "/home/cnavarrete/mendel-nas1/BDNYC/Linelists/" 
xlist = "gaslistR10k.dat" 

# Gas List
gaslist = ['h2o','ch4', 'co', 'nh3','k','na'] 

ngas = len(gaslist)

# Chose an alternative cross section if desired
# Use Mike's (Burrows) Alkalis?
#Use Allard (=0), Burrow's(=1), and new Allard (=2)
malk = 1
# Use Mike's CH4?
mch4 = 0 

# EMCEE ------------------------

# Emcee details
ndim = 22
nwalkers = ndim * 16
nburn = 10000
niter = 30000

# Testing ---------------------
runtest = 1 #Is this a test of restart?
make_arg_pickle = 2 # Set to 0 if no and run, Set to 1 for write and exit (no run), Set to 2 for write and continue
fresh = 0 #Originally above vector state init, moved here for testing

# Output
outdir = "/home/cnavarrete/mendel-nas1/BDNYC/brewster/G570D_Results"

# Are we using DISORT for radiative transfer? 
use_disort = 0 

# use the fudge factor / tolerance parameter?
do_fudge = 1

# Naming Files -----------------
# full final sampler with likelihoods, chain, bells and whistles
finalout = runname+".pk1"
chaindump = runname+"_last_nc.pic" # periodic dumps/snapshots , chain 
picdump = runname+"_snapshot.pic" # The whole thing w/ probs

# Names for status file runtimes
statfile = "status_ball"+runname+".txt"
rfile = "runtimes_"+runname+".dat"

#Scale Factor r2d2 from 1 Rj Radius
r2d2 = (71492e3)**2. / (dist * 3.086e+16)**2.

# Vector State Init -------------------------------------
p0 = np.empty([nwalkers,ndim])
if (fresh == 0):
    # ----- "Gas" parameters (Includes gases, gravity, logg, scale factor, dlambda, and tolerance parameter) --
    # # For Non-chemical equilibrium
    p0[:,0] = (0.5*np.random.randn(nwalkers).reshape(nwalkers)) - 3.5 # H2O
    p0[:,1] = (0.5*np.random.randn(nwalkers).reshape(nwalkers)) - 5.0 # CH4
    p0[:,2] = (0.5*np.random.randn(nwalkers).reshape(nwalkers)) - 3.0 # CO 
    p0[:,3] = (0.5*np.random.randn(nwalkers).reshape(nwalkers)) - 4.6 # NH3
    p0[:,4] = (0.5*np.random.randn(nwalkers).reshape(nwalkers)) - 6.5 # Na+K
    p0[:,5] = 0.1*np.random.randn(nwalkers).reshape(nwalkers) + 4.0  # gravity
    p0[:,6] = r2d2 + (np.random.randn(nwalkers).reshape(nwalkers) * (0.5*r2d2))  # scale factor 1
    p0[:,7] = np.random.randn(nwalkers).reshape(nwalkers) * 20  # dlambda
    p0[:,8] = np.log10((np.random.rand(nwalkers).reshape(nwalkers) * (max(obspec[2,:]**2)*(0.1 - 0.01))) + (0.01*min(obspec[2,10::3]**2))) # tolerance parameter 1
    

    # If you do Chemical Equilibrium you will have these parameters instead
    # p0[:, 0] = (0.1 * np.random.randn(nwalkers).reshape(nwalkers)) - 0.5  # met
    # p0[:, 1] = (0.1 * np.random.randn(nwalkers).reshape(nwalkers)) + 1  # CO
    # p0[:, 2] = 0.1 * np.random.randn(nwalkers).reshape(nwalkers) + 5.4  # gravity *** put it near SED value **
    # p0[:, 3] = r2d2 + (np.random.randn(nwalkers).reshape(nwalkers) * (0.1 * r2d2))  # scale factor
    # p0[:, 4] = np.random.randn(nwalkers).reshape(nwalkers) * 0.001  # dlambda
    
    # ------ If you have a cloud, you will always need cloud parameters. ------
    
    # Slab cloud params
    # These parameters should be commented out or adjusted for
    # e.g grey cloud or power law cloud, or no cloud, or deck cloud
    # this example is a "real" cloud with Hansen a and b parameters
    # p0[:,10] = np.random.rand(nwalkers).reshape(nwalkers) # optical depth
    # p0[:,11] = -2. + 0.5 * np.random.randn(nwalkers).reshape(nwalkers) # cloud top pressure
    # p0[:,12] = np.random.rand(nwalkers).reshape(nwalkers) # cloud thickness in pressure
    # p0[:,13] = -1. + 0.1*np.random.randn(nwalkers).reshape(nwalkers) # Hansen a if "real" cloud or single scattering albedo between 0 and 1 (np.rand=uniform distribution) for 89/99 cloud
    # p0[:,14] = 0.1*np.random.rand(nwalkers).reshape(nwalkers) # Hansen b for "real" cloud or Power law for 89/99 cloud
    
    # # Deck cloud params
    # These parameters should be commented out or adjusted for
    # e.g grey cloud or power law cloud, or no cloud, or deck cloud
    # this example is a "real" cloud with Hansen a and b parameters
    # p0[:,20] = 0.5+ 0.2* np.random.rand(nwalkers).reshape(nwalkers) # cloud top pressure
    # p0[:,21] = np.random.rand(nwalkers).reshape(nwalkers) # cloud thickness in pressure
    # p0[:,22] = -1. + 0.1*np.random.randn(nwalkers).reshape(nwalkers) # Hansen a if "real" cloud or single scattering albedo between 0 and 1 (np.rand=uniform distribution) for 89/99 cloud
    # p0[:,23] = np.abs(0.1+ 0.01*np.random.randn(nwalkers).reshape(nwalkers)) # Hansen b for "real" cloud or Power law for 89/99 cloud
   
    # ------ And now the T-P params. --------
   
    # PROFILE TYPE 1, 
    p0[:, ndim-14] = 50. + (np.random.randn(nwalkers).reshape(nwalkers))  # gamma - removes wiggles unless necessary to profile
    BTprof = np.loadtxt("BTtemp800_45_13.dat")
    for i in range(0, 13):  # 13 layer points ====> Total: 13 + 13 (gases+) +no cloud = 26
        p0[:,ndim-13 + i] = (BTprof[i] - 200.) + (150. * np.random.randn(nwalkers).reshape(nwalkers))
    for i in range(0, nwalkers):
        while True:
            Tcheck = TPmod.set_prof(proftype, coarsePress, press, p0[i, ndim-13:])
            if min(Tcheck) > 1.0:
                break
            else:
                for i in range(0,13):
                    p0[:,ndim-13 + i] = BTprof[i] + (50. * np.random.randn(nwalkers).reshape(nwalkers))

    # PROFILE TYPE 2 
    # p0[:,24] = 0.39 + 0.1*np.random.randn(nwalkers).reshape(nwalkers)
    # p0[:,25] = 0.14 +0.05*np.random.randn(nwalkers).reshape(nwalkers)
    # p0[:,26] = -1.2 + 0.2*np.random.randn(nwalkers).reshape(nwalkers)
    # p0[:,27] = 2.25+ 0.2*np.random.randn(nwalkers).reshape(nwalkers)
    # p0[:,28] = 4200. + (500.*  np.random.randn(nwalkers).reshape(nwalkers))
    # for i in range (0,nwalkers):
    #     while True:
    #         Tcheck = TPmod.set_prof(proftype, coarsePress, press, p0[i, ndim-5:])
    #         if min(Tcheck) > 1.0:
    #             break
    #         else:
    #             p0[i, ndim-5] = 0.39 + 0.01*np.random.randn()
    #             p0[i, ndim-4] = 0.14 + 0.01*np.random.randn()
    #             p0[i, ndim-3] = -1.2 + 0.2*np.random.randn()
    #             p0[i, ndim-2] = 2. + 0.2*np.random.randn()
    #             p0[i, ndim-1] = 4200. + (200.*np.random.randn())


    # PROFILE TYPE 3
    # p0[:, 5] = 0.39 + 0.1 * np.random.randn(nwalkers).reshape(nwalkers)  # alpha
    # p0[:, 6] = 0.14 + 0.05 * np.random.randn(nwalkers).reshape(nwalkers)  # beta
    # p0[:, 7] = -1.2 + 0.2 * np.random.randn(nwalkers).reshape(nwalkers)  # p0
    # p0[:, 8] = -1.2 + 0.2 * np.random.randn(nwalkers).reshape(
    #             nwalkers)  # p1 or p0 the one that doesn't disappear
    # p0[:, 9] = 2.25 + 0.2 * np.random.randn(nwalkers).reshape(nwalkers)  # p2
    # p0[:, 10] = 4200. + (500. * np.random.randn(nwalkers).reshape(nwalkers))  # base temp (T3)
    # for i in range(0, nwalkers):
    #     while True:
    #         Tcheck = TPmod.set_prof(proftype, coarsePress, press, p0[i, ndim - 6:])
    #         if min(Tcheck) > 1.0:
    #             break
    #         else:
    #             p0[i, ndim - 6] = 0.39 + 0.01 * np.random.randn()
    #             p0[i, ndim - 5] = 0.39 + 0.01 * np.random.randn()
    #             p0[i, ndim - 4] = 0.14 + 0.01 * np.random.randn()
    #             p0[i, ndim - 3] = -1.2 + 0.2 * np.random.randn()
    #             p0[i, ndim - 2] = 2. + 0.2 * np.random.randn()
    #             p0[i, ndim - 1] = 4200. + (200. * np.random.randn())



if (fresh != 0):
    fname=chaindump
    pic=pickle.load(open(fname,'rb'))
    p0=pic
    if (fresh == 2):
        for i in range(0,9):
            p0[:,i] = (np.random.rand(nwalkers).reshape(nwalkers)*0.5) + p0[:,i]


prof = np.full(13, 100.)
if proftype == 9:
    modP, modT = np.loadtxt(pfile, skiprows=1, usecols=(1, 2), unpack=True)
    tfit = InterpolatedUnivariateSpline(np.log10(modP), modT, k=1)
    prof = tfit(logcoarsePress)


# Now we'll get the opacity files into an array
inlinetemps,inwavenum,linelist,gasnum,nwave = testkit.get_opacities(gaslist,w1,w2,press,xpath,xlist,malk)

# Get the cia bits
tmpcia, ciatemps = ciamod.read_cia("CIA_DS_aug_2015.dat",inwavenum)
cia = np.asfortranarray(np.empty((4,ciatemps.size,nwave)),dtype='float32')
cia[:,:,:] = tmpcia[:,:,:nwave] 
ciatemps = np.asfortranarray(ciatemps, dtype='float32')

# grab BFF and Chemical grids
bff_raw,ceTgrid,metscale,coscale,gases_myP = testkit.sort_bff_and_CE(chemeq,"chem_eq_tables_P3K.pic",press,gaslist)


settings.init()
settings.runargs = gases_myP,chemeq,dist, cloudtype,do_clouds,gasnum,cloudnum,inlinetemps,coarsePress,press,inwavenum,linelist,cia,ciatemps,use_disort,fwhm,obspec,proftype,do_fudge, prof,do_bff,bff_raw,ceTgrid,metscale,coscale


# Now we set up the MPI bits
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit()
 

# Write the arguments to a pickle if needed
if (make_arg_pickle > 0):
    pickle.dump(settings.runargs,open(outdir+runname+"_runargs.pic","wb"))
    if( make_arg_pickle == 1):
        sys.exit()


sampler = emcee.EnsembleSampler(nwalkers, ndim, testkit.lnprob, pool=pool)
# '''
# run the sampler
print("running the sampler")
clock = np.empty(80000)
k = 0
times = open(rfile, "w")
times.close()
if runtest == 0 and fresh == 0:
    pos, prob, state = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    p0 = pos
for result in sampler.sample(p0, iterations=niter):
    clock[k] = time.perf_counter()
    if k > 1:
        tcycle = clock[k] - clock[k-1]
        times = open(rfile, "a")
        times.write("*****TIME FOR CYCLE*****\n")
        times.write(str(tcycle))
        times.close()
    k = k+1
    position = result.coords
    f = open(statfile, "w")
    f.write("****Iteration*****")
    f.write(str(k))
    f.write("****Reduced Chi2*****")
    f.write(str(result.log_prob * -2.0/(obspec.shape[1] / 3.0)))
    f.write("****Accept Fraction*****")
    f.write(str(sampler.acceptance_fraction))
    f.write("*****Values****")
    f.write(str(result.coords))
    f.close()

    if (k==10 or k==1000 or k==1500 or k==2000 or k==2500 or k==3000 or k==3500 or k==4000 or k==4500 or k==5000 or k==6000 or k==7000 or k==8000 or k==9000 or k==10000 or k==11000 or k==12000 or k==15000 or k==18000 or k==19000 or k==20000 or k==21000 or k==22000 or k==23000 or k==24000 or k==25000 or k==26000 or k==27000 or k==28000 or k==29000 or k == 30000 or k == 35000 or k == 40000 or k == 45000 or k == 50000 or k == 55000 or k == 60000 or k == 65000):
        chain=sampler.chain
        lnprob=sampler.lnprobability
        output=[chain,lnprob]
        pickle.dump(output,open(outdir+picdump,"wb"))
        pickle.dump(chain[:,k-1,:], open(chaindump,'wb'))



# get rid of problematic bit of sampler object
del sampler.__dict__['pool']

def save_object(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

pool.close()

save_object(sampler, outdir+finalout)

