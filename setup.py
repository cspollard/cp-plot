from distutils.core import setup
setup \
  ( name='cpplot'
  , version='0.1'
  , packages=['cpplot']
  , install_requires= \
    [ "numpy"
    , "scipy"
    , "matplotlib"
    ]
  )