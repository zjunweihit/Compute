# Install MPICH
* Ubuntu 18.04

```
sudo apt install mpich mpich-doc
```

# Install Open MPI

No `mpicc` in Open MPI packages.

* Ubuntu 18.04

```
sudo apt install openmpi-bin openmpi-common openmpi-doc
```

# Note

If installed both Open MPI and MPICH in Ubuntu system, it has to switch `mpirun` to MPICH. Otherwise, it will not work to run `mpich`(OpenMPI by default) with binary built by `mpicc`(MPICH)
```
sudo update-alternatives --config mpirun
There are 2 choices for the alternative mpirun (providing /usr/bin/mpirun).

  Selection    Path                     Priority   Status
  ------------------------------------------------------------
  * 0            /usr/bin/mpirun.openmpi   50        auto mode
    1            /usr/bin/mpirun.mpich     40        manual mode
    2            /usr/bin/mpirun.openmpi   50        manual mode

Press <enter> to keep the current choice[*], or type selection number: 1
update-alternatives: using /usr/bin/mpirun.mpich to provide /usr/bin/mpirun (mpirun) in manual mode
```
