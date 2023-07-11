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
  ( xs, ys, labels, xlabel, ylabel
  , linewidths=None, colors=defcolors, markers=defmarkers
  , markerfills=None, ratio=False
  , ratioylabel=None
  , xticks=None, xticklabels=None
  , alphas=None, markeroffsets=False
  ):

  if markerfills is None:
    markerfills = colors

  if linewidths is None:
    linewidths = list(map(lambda x : 0, colors))

  if alphas is None:
    alphas = list(map(lambda x : 1, colors))


  fig = figure.Figure(figsize=(8, 8))

  if ratio:
    plt = fig.add_subplot(3, 1, (1, 2))
  else:
    plt = fig.add_subplot(111)

  ncurves = len(ys)

  xs , xerrs = xs

  markerlocs = []

  for i in range(ncurves):
    if markeroffsets:
      xrange = numpy.stack([xs - xerrs[0], xs + xerrs[1]])
      print(xrange)

      # the marker is offset w.r.t. the midpoint of the errorbar
      # but never at the left or right edge.
      thisxs = xrange[0] + (i + 1) / (ncurves + 2) * (xrange[1] - xrange[0])

      thisxerrs = numpy.stack([ thisxs - xrange[0] , xrange[1] - thisxs])
      print(thisxerrs)
      markerlocs.append((thisxs, thisxerrs))

    else:
      markerlocs.append((xs, xerrs))

    zs , zerrs = ys[i]

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

  plt.set_ylabel(ylabel)

  if ratio:
    plt.axes.xaxis.set_ticklabels([])
    plt = fig.add_subplot(3, 1, 3)

    plt.plot([xs[0] - xerrs[0], xs[-1] + xerrs[-1]], [1.0, 1.0], lw=1, color="gray", zorder=999)

    nom, nomerr = ys[0]
    for i in range(1, len(ys)):
      xs = markerlocs[i][0]
      xerrs = markerlocs[i][1]
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
  
  plt.set_xlabel(xlabel)
  if xticks is not None:
    plt.set_xticks(xticks)
  if xticklabels is not None:
    plt.set_xticklabels(xticklabels)
  if ratioylabel is not None:
    plt.set_ylabel(ratioylabel)

  return fig


def comparehist \
  ( ys, binning, labels, xlabel, ylabel
  , colors=defcolors, markers=defmarkers, ratio=False
  , alphas=None
  , linewidths=None, markerfills=None
  , xticks=None, xticklabels=None
  , ratioylabel=None
  , markeroffsets=False
  ):

  xs = (binning[1:]+binning[:-1]) / 2.0
  xerrs = ((binning[1:]-binning[:-1]) / 2.0)
  xerrs = numpy.stack([xerrs , xerrs])

  return \
    compare \
    ( (xs, xerrs), ys, labels, xlabel, ylabel
    , colors=colors, markers=markers, ratio=ratio
    , linewidths=linewidths, markerfills=markerfills
    , alphas=alphas
    , xticks=xticks, xticklabels=xticklabels
    , ratioylabel=ratioylabel
    , markeroffsets=markeroffsets
    )


def hist(m, binning, normalized=False):
  return numpy.histogram(m, bins=binning, density=normalized)[0]


def zeroerr(h):
  return h , numpy.zeros_like(h)


def poiserr(xs):
  uncerts = numpy.sqrt(xs)
  uncerts = numpy.stack([uncerts, uncerts], axis=0)

  return xs , uncerts


def stderr(nom, vars):
  diffs2 = list(map(lambda v: (v - nom)**2, vars))
  return nom , numpy.sqrt(sum(diffs2))


def divbinom(ks, ns):
  def binom(k, n):
    if n == 0:
      return (0, 0)

    r = stats.binomtest(k, n).proportion_ci(0.68)
    cv = k / n

    return (r.low, r.high)

  uncerts = [ binom(k, n) for k , n in zip(ks, ns) ]

  cv = ks / ns
  uncerts = numpy.array(uncerts).T

  uncerts[0] = cv - uncerts[0]
  uncerts[1] = uncerts[1] - cv

  return ks / ns , uncerts

