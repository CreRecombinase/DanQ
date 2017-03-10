#!/bin/bash

LIBGPU_RELEASE="0.6.1"

INSTALL_PREFIX="${HOME}/rcc2/libgpuarray-${LIBGPU_RELEASE}"

wget "https://github.com/Theano/libgpuarray/archive/v${LIBGPU_RELEASE}.tar.gz";
tar -xvzf "v${LIBGPU_RELEASE}.tar.gz";
cd "${LIBGPU_RELEASE}";
mkdir build;
cd build ;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}";
make;
make install;
cd .. ;
python setup.py build_ext -L "${INSTALL_PREFIX}/lib/" -I "${INSTALL_PREFIX}/include" ;
python setup.py install --user



