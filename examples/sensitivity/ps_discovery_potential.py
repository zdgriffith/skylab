###############################################################################
# @file   ps_discovery_potential.py
# @author Josh Wood
# @date   Apr 26 2017
# @brief  Calculate discovery potential flux for point source analysis using
#         Rene Reimann's diffuse muon neutrino sample. Discovery potential
#         flux is the flux required for a 5 sigma detection 50% of the time.
#
# @dependencies
#   /home/jwood/public_html/joint-icecube-hawc/icecube-ana/diffuse-dataset/diffuse_dataset.py
###############################################################################

import sys
import numpy 
import diffuse_dataset

from   skylab.psLLH       import PointSourceLLH
from   skylab.ps_model    import ClassicLLH
from   skylab.ps_injector import PointSourceInjector
from   math               import sin

###############################################################################

# @function Print()
# @brief    Print to screen and log file
def Print(string, log):

  print string
  log.write(string + '\n')

# END Print()

###############################################################################

#############
# CONSTANTS #
#############

deg2rad = numpy.pi/180.
rad2deg = 1./deg2rad

GeV = 1
TeV = 1000*GeV

#############
# CONSTANTS #
#############

###############################################################################

#############
# ARGUMENTS #
#############

import argparse

p = argparse.ArgumentParser(description="Calculates Discovery Potential" + \
                                        "at Given Declination",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--season", default="IC86, 2012-15",
                help="Season name (default=\"IC86, 2012-15\")") 
p.add_argument("--dec", default=0.0, type=float,
                help="Source declination in J2000 epoch [deg] (default=0)")
p.add_argument("--ra", default=180.0, type=float,
                help="Source right ascension in J2000 epoch [deg] (default=180)")
p.add_argument("--index", default=2.0, type=float,
                help="Index for simple power law:     " + \
                     "dN/dE = A (E / E0)^-index (default=2.0)")
p.add_argument("--E0", default=1*TeV, type=float,
                help="E0 in GeV for simple power law: " + \
                     "dN/dE = A (E / E0)^-index (default=1000)")
p.add_argument("--ns_bounds", default=[0, 100], nargs=2, type=float,
                help="Range in signal events to search for discovery potential flux (default= 0 100)")
p.add_argument("--nsample", default=1000, type=int,
                help="Number of signal samples used to compute discovery potential (default=1000)")
p.add_argument("--tolerance", default=0.02, type=float,
                help="Flux tolerance fraction on final flux (default=0.02)")
p.add_argument("--seed", default=1, type=int,
                help="Seed for RNG (default=1)")
p.add_argument("--useEXPbackground", action='store_true',
                help="Use data as background rather than MC")

args = p.parse_args()

(livetime, expfile, mcfile, tag) = diffuse_dataset.season(args.season)

exp = numpy.load(expfile)
mc  = numpy.load(mcfile)

log = open((tag + "dec%+06.2f_index%.2f_ps_discovery_potential.log") % (args.dec, args.index), "w", 0)

#############
# ARGUMENTS #
#############

###############################################################################

####################
# DECLINATION BINS #
####################

sinDec_min  = -sin(5*deg2rad)
sinDec_max  = 1.
nbin        = 60+1
bin_width   = (sinDec_max - sinDec_min)/nbin
sinDec_bins = numpy.arange(sinDec_min, sinDec_max+0.1*bin_width, bin_width)

####################
# DECLINATION BINS #
####################

###############################################################################

################
# SKYLAB SETUP #
################

# point source location
src_ra  =  args.ra*deg2rad;
src_dec = args.dec*deg2rad;

# switch for choosing data vs MC for background PDF
if args.useEXPbackground: MCbackground = None
else:                     MCbackground = ["conv","astro"]

# create classic likelihood model
llh_model = ClassicLLH(sinDec_bins=sinDec_bins,
                       sinDec_range=[sinDec_min, sinDec_max],
                       MCbackground=MCbackground)

# specify a point source likelihood based on classicLLH
psllh = PointSourceLLH(exp, mc, livetime,
                       scramble=True, seed=args.seed,
                       llh_model=llh_model,
                       mode = "box",
                       delta_ang=numpy.radians(10.*1.0),
                       nside=128)

# make an injector to model point sources
print ""
inj = PointSourceInjector(gamma=args.index,E0=args.E0,seed=args.seed)
inj.fill(src_dec, mc, livetime)

################
# SKYLAB SETUP #
################

###############################################################################

##################
# 5 SIGMA SEARCH #
##################

Print("\nSearching for 5 sigma ... ", log)
ns_min = args.ns_bounds[0]
ns_max = args.ns_bounds[1]
nstep  = 10
delta  = (ns_max - ns_min)/nstep

results = numpy.empty( [3, nstep+1], dtype=float )

# step through ns values within [ns_min, ns_max]
for step in range(0,nstep+1):

  # compute signal for this step
  ns = ns_min + step*delta

  # compute average TS from 5 samples
  nsample = 5
  avgTS   = 0

  for i in range(0,nsample):

    # get event sample for source
    ni, sample = inj.sample(src_ra, ns, poisson=False).next()

    # compute TS
    TS, Xmin = psllh.fit_source(src_ra,src_dec,inject=sample,scramble = True)

    # add to average
    avgTS = avgTS + TS/nsample

  # END for (i)

  # compute average significance
  sigma = numpy.sqrt(avgTS)

  Print(" [%3d/%d] ns %7.2f, sigma %6.2f" % (step+1, nstep+1, ns, sigma), log)

  results[0][step] = ns                  # x
  results[1][step] = sigma               # y
  results[2][step] = numpy.sqrt(nsample) # 1/error

# END for (step)

# linear fit
p1, p0 = numpy.polyfit(results[0],     # x = ns
                       results[1],     # y = sigma
                       1,              # 1st order poly
                       w = results[2]) # 1/error = sqrt(nsample)

ns_5sigma = (5 - p0) / p1
Print("Rough Estimate: ns = %.2f @ 5 sigma" % ns_5sigma, log)

##################
# 5 SIGMA SEARCH #
##################

###############################################################################

##############################
# DISCOVERY POTENTIAL SEARCH #
##############################

Print("\nSearching for Discovery Threshold (50% >= 5 sigma) ... ", log)

if ns_5sigma > 10:
  ns_min = 0.5*ns_5sigma
  ns_max = 1.5*ns_5sigma
else:
  ns_min = 0
  ns_max = 10

nstep  = 10
delta  = (ns_max - ns_min)/nstep

ns_lower = ns_min
ns_upper = ns_max

# step through ns values within [ns_min, ns_max]
for step in range(0, nstep+1):

  # compute signal for this step
  ns = ns_min + step*delta

  # compute number of discoveries over 100 samples
  nsample = args.nsample
  ndisc   = 0
  avg_ni  = 0
  for i in range(0,nsample):

    # get event sample for source
    ni, sample = inj.sample(src_ra, ns, poisson=True).next()

    # compute TS
    TS, Xmin = psllh.fit_source(src_ra,src_dec,inject=sample,scramble = True)
#    print "  - [%d] ni %.2f, sigma %.2f" % (step, ni, np.sqrt(TS))
    if numpy.sqrt(TS) >= 5: ndisc = ndisc + 1

    avg_ni = avg_ni + float(ni)/nsample

  # END for (i)

  P = float(ndisc)/nsample
  Print(" [%3d/%d] ns %.2f, discoveries %3d/%d = %.2f" % (step+1, nstep+1, ns, ndisc, nsample, P), log)

  if (P > 0.5 and ns < ns_upper): ns_upper = ns
  if (P < 0.5 and ns > ns_lower): ns_lower = ns

Print("Discovery Threshold Bounds: %.2f < ns < %.2f" % (ns_lower, ns_upper), log)

if (ns_lower > ns_upper):
  Print("Uh-oh. We screwed up. %.2f > %.2f is just plain wrong." % (ns_lower, ns_upper), log)
  exit(0)

##############################
# DISCOVERY POTENTIAL SEARCH #
##############################

###############################################################################

######################################
# DISCOVERY POTENTIAL REFINED SEARCH #
######################################

Print("\nRefining Search ... ", log)

tol = args.tolerance # minimum tolerance on final flux
ns_min = ns_lower
ns_max = ns_upper
delta  = tol*ns_min
nstep  = int( (ns_max - ns_min)/delta )

# step through ns values within [ns_min, ns_max]
for step in range(0, nstep+1):

  # compute signal for this step
  ns = ns_min + step*delta

  # compute number of discoveries over 100 samples
  nsample = args.nsample
  ndisc   = 0
  avg_ni  = 0
  for i in range(0,nsample):

    # get event sample for source
    ni, sample = inj.sample(src_ra, ns, poisson=True).next()

    # compute TS
    TS, Xmin = psllh.fit_source(src_ra,src_dec,inject=sample,scramble = True)
#    print "  - [%d] ni %.2f, sigma %.2f" % (step, ni, np.sqrt(TS))
    if numpy.sqrt(TS) >= 5: ndisc = ndisc + 1

    avg_ni = avg_ni + float(ni)/nsample

  # END for (i)

  P = float(ndisc)/nsample
  Print(" [%3d/%d] ns %.2f, discoveries %3d/%d = %.2f" % (step+1, nstep+1,ns, ndisc, nsample, P), log)

  if (P > 0.5 and ns < ns_upper): ns_upper = ns
  if (P < 0.5 and ns > ns_lower): ns_lower = ns

# END for (step)

Print("Discovery Threshold Bounds: %.2f < ns < %.2f" % (ns_lower, ns_upper), log)

ns_avg = (ns_lower + ns_upper)/2.
norm   = inj.mu2flux( ns_avg )
Print("\n---", log)
Print("Source Declination:  %.2f deg" % args.dec, log)
Print("Flux Normalization:  A = %.2e TeV^-1 cm^-2 s^-1 @ %.2f TeV" % (norm*TeV, args.E0/TeV), log)
Print("Spectral Assumption: dN/dE = A (E / %.2f TeV)^-%.2f" % (args.E0/TeV, args.index), log)
if args.useEXPbackground: Print("Background:          EXP", log)
else:                     Print("Background:          MC",  log)
Print("---\n", log)

log.close()

######################################
# DISCOVERY POTENTIAL REFINED SEARCH #
######################################

###############################################################################
