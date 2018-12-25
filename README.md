# Compute
Introduce some common compute stacks

## ROCm

* Add apt repo
```
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
```
* Install ROCm packages
```
sudo apt update
sudo apt install rocm-dkms
```
* Grant the permission
```
sudo usermod -a -G video $LOGNAME
```
* Verify the driver
```
/opt/rocm/bin/rocminfo
/opt/rocm/opencl/bin/x86_64/clinfo
```
* Remove the ROCm
```
sudo apt autoremove rocm-dkms
```

Reference:
* [ROCm Installation Guide](https://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#installing-from-amd-rocm-repositories)

## SYCL
Glone triSYCL to the same fold as Compute.git location
```Shell
git clone https://github.com/zjunweihit/triSYCL.git
```
Ubuntu18.04: install the required packages
```Shell
sudo apt install clang g++ libboost-dev libboost-log-dev cmake
```
triSYCL API: 
* [triSYCL](http://trisycl.github.io/triSYCL/Doxygen/triSYCL/html/)
