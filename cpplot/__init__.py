import numpy
import scipy.stats as stats
import matplotlib.figure as figure


defmarkers = ["o", "s", "v", "^", "<", ">", "s", "X", "P", "D", "*"]*10
defcolors = ["black", "orange", "blue", "red", "green"]*10
defmarkerfills = defcolors


def intbins(h):
  return numpy.arange(len(h)+1)


def divuncorr(xs, ys, dxs, dys):
  zs = numpy.where(ys == 0.0, 0.0, xs / ys)
  dzs = numpy.sqrt(dxs**2 + (zs * dys)**2) / ys
  return zs , dzs


def compare \
  ( plt, xs, ys, labels, xlabel, ylabel
  , linewidths=None, colors=defcolors, markers=defmarkers
  , markerfills=None
  , xticks=None, xticklabels=None
  , alphas=None, markeroffsets=False
  , errorfills=None
  ):

  if markerfills is None:
    markerfills = colors

  if linewidths is None:
    linewidths = list(map(lambda x : 0, colors))

  if alphas is None:
    alphas = list(map(lambda x : 1, colors))

  if errorfills is None:
    errorfills = list(map(lambda x : False, colors))


  ncurves = len(ys)

  markerlocs = []

  for i in range(ncurves):
    thesexs , thesexerrs = xs[i]

    if markeroffsets:
      xrange = numpy.stack([thesexs - thesexerrs[0], thesexs + thesexerrs[1]])

      # the marker is offset w.r.t. the midpoint of the errorbar
      # but never at the left or right edge.
      thisxs = xrange[0] + (i + 1) / (ncurves + 1) * (xrange[1] - xrange[0])

      thisxerrs = numpy.stack([ thisxs - xrange[0] , xrange[1] - thisxs])
      markerlocs.append((thisxs, thisxerrs))

    else:
      markerlocs.append((thesexs, thesexerrs))

    zs , zerrs = ys[i]

    if not errorfills[i]:
      plt.errorbar \
        ( markerlocs[i][0]
        , zs
        , xerr=markerlocs[i][1]
        , yerr=zerrs
        , label=labels[i]
        , marker=markers[i]
        , color=colors[i]
        , markerfacecolor=markerfills[i]
        , linewidth=linewidths[i]
        , elinewidth=2
        , zorder=i
        , alpha=alphas[i]
        )

    else:
      plt.plot \
        ( markerlocs[i][0]
        , zs
        , label=labels[i]
        , color=colors[i]
        , markerfacecolor=markerfills[i]
        , linewidth=linewidths[i]
        , zorder=i
        , alpha=alphas[i]
        )

      plt.fill_between \
        ( markerlocs[i][0]
        , zs + zerrs[1]
        , zs - zerrs[0]
        , color=colors[i]
        , zorder=i-0.5
        , alpha=alphas[i]*0.5
        )

  plt.set_ylabel(ylabel)
  plt.set_xlabel(xlabel)
  if xticks is not None:
    plt.set_xticks(xticks)
  if xticklabels is not None:
    plt.set_xticklabels(xticklabels)

  return plt


def comparehist \
  ( plt, ys, binning, labels, xlabel, ylabel
  , colors=defcolors, markers=defmarkers
  , alphas=None
  , linewidths=None, markerfills=None
  , xticks=None, xticklabels=None
  , errorfills=None
  , markeroffsets=False
  ):

  if markerfills is None:
    markerfills = colors

  if linewidths is None:
    linewidths = list(map(lambda x : 0, colors))

  if alphas is None:
    alphas = list(map(lambda x : 1, colors))


  xs = (binning[1:]+binning[:-1]) / 2.0
  xerrs = ((binning[1:]-binning[:-1]) / 2.0)
  xerrs = numpy.stack([xerrs , xerrs])

  return \
    compare \
    ( plt, [(xs, xerrs)] * len(ys) , ys, labels, xlabel, ylabel
    , colors=colors, markers=markers
    , linewidths=linewidths, markerfills=markerfills
    , alphas=alphas
    , xticks=xticks, xticklabels=xticklabels
    , markeroffsets=markeroffsets
    , errorfills=errorfills
    )


def comparehistratio \
  ( plt , ys, binning, labels, xlabel, ylabel
  , colors=defcolors, markers=defmarkers
  , alphas=None
  , linewidths=None, markerfills=None
  , xticks=None, xticklabels=None
  ):

  if markerfills is None:
    markerfills = colors

  if linewidths is None:
    linewidths = list(map(lambda x : 0, colors))

  if alphas is None:
    alphas = list(map(lambda x : 1, colors))


  xs = (binning[1:]+binning[:-1]) / 2.0
  xerrs = ((binning[1:]-binning[:-1]) / 2.0)
  xerrs = numpy.stack([xerrs , xerrs])

  plt.plot([binning[0], binning[-1]], [1.0, 1.0], lw=1, color="gray", zorder=999)

  nom, nomerr = ys[0]
  for i in range(1, len(ys)):
    zs , zerrs = divuncorr(ys[i][0], nom, ys[i][1][0], nomerr)

    plt.errorbar \
      ( xs
      , zs
      , xerr=xerrs
      , yerr=zerrs
      , label=labels[i]
      , marker=markers[i]
      , color=colors[i]
      , markerfacecolor=markerfills[i]
      , linewidth=linewidths[i]
      , elinewidth=2
      , zorder=i
      , alpha=alphas[i]
      )

  plt.set_ylabel(ylabel)

  plt.set_xlabel(xlabel)
  if xticks is not None:
    plt.set_xticks(xticks)
  if xticklabels is not None:
    plt.set_xticklabels(xticklabels)

  return plt


def hist(m, binning, weights=None, normalized=False):
  return numpy.histogram(m, bins=binning, weights=weights, density=normalized)[0]


def zeroerr(h):
  return h , numpy.zeros_like(h)


def poiserr(xs):
  uncerts = numpy.sqrt(xs)
  uncerts = numpy.stack([uncerts, uncerts], axis=0)

  return xs , uncerts


def stderr(nom, vars):
  diffs2 = list(map(lambda v: (v - nom)**2, vars))
  return nom , numpy.sqrt(sum(diffs2))


def binom(k, n):
  if n == 0:
    return (0, 0)

  r = stats.binomtest(k, n).proportion_ci(0.68)

  return (r.low, r.high)


def divbinom(ks, ns):
  uncerts = [ binom(k, n) for k , n in zip(ks, ns) ]

  cv = ks / ns
  uncerts = numpy.array(uncerts).T

  uncerts[0] = cv - uncerts[0]
  uncerts[1] = uncerts[1] - cv

  return cv , uncerts
