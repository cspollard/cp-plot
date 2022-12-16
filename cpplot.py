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
  , lw=0, colors=defcolors, markers=defmarkers
  , markerfills=None, ratio=False
  , ratioylabel=None
  , xticks=None, xticklabels=None
  ):
  if markerfills is None:
    markerfills = colors

  fig = figure.Figure(figsize=(8, 8))

  if ratio:
    plt = fig.add_subplot(3, 1, (1, 2))
  else:
    plt = fig.add_subplot(111)

  xs , xerrs = xs

  for i in range(len(ys)):
    zs , zerrs = ys[i]

    plt.errorbar \
      ( xs
      , zs
      , xerr=xerrs
      , yerr=zerrs
      , label=labels[i]
      , marker=markers[i]
      , color=colors[i]
      , markerfacecolor=markerfills[i]
      , linewidth=lw
      , elinewidth=2
      , zorder=i
      )

  plt.set_ylabel(ylabel)

  if ratio:
    plt.axes.xaxis.set_ticklabels([])
    plt = fig.add_subplot(3, 1, 3)

    plt.plot([xs[0] - xerrs[0], xs[-1] + xerrs[-1]], [1.0, 1.0], lw=1, color="gray", zorder=999)

    nom, nomerr = ys[0]
    for i in range(1, len(ys)):
      # TODO
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
        , linewidth=lw
        , elinewidth=2
        , zorder=i
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
  , lw=0, markerfills=None
  , xticks=None, xticklabels=None
  , ratioylabel=None
  ):

  xs = (binning[1:]+binning[:-1]) / 2.0
  xerrs = ((binning[1:]-binning[:-1]) / 2.0)

  return \
    compare \
    ( (xs, xerrs), ys, labels, xlabel, ylabel
    , colors=colors, markers=markers, ratio=ratio
    , lw=lw, markerfills=None
    , xticks=xticks, xticklabels=xticklabels
    , ratioylabel=ratioylabel
    )


def hist(m, binning, normalized=False):
  return numpy.histogram(m, bins=binning, density=normalized)[0]


def zeroerr(h):
  return h , numpy.zeros_like(h)


def poiserr(xs):
  uncerts = numpy.sqrt(xs)
  uncerts = numpy.stack([uncerts, uncerts], axis=0)

  return xs , uncerts


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

