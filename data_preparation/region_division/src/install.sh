# install libboost, libpcl, cmake
sudo apt-get update -y
sudo apt-get install -y libboost-all-dev libpcl-dev cmake

# install 3rd package (cnpy)
workdir=$PWD
lib_cnpy="3rd/cnpy/build"
mkdir ${lib_cnpy}
cd ${lib_cnpy}
cmake ..
make

# install our project
cd ${workdir}
mkdir build
cd build
cmake ..
make
