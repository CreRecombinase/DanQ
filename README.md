# Installation instructions for RCC (GPU queue)


1) Copy the directory `Modules` to your home directory

` cp -r Modules ~/`

2) Tell `module` about your new custom module folder (be sure *not* to put a trailing `/`)

   `module use $HOME/Modules/rcc`

3) load the RCC DanQ/0.2.0 module

   `module load DanQ/0.2.0`

4) Install Keras
   `pip install keras==0.2.0 --user --force-reinstall --upgrade`

5) Install Theano

 `pip install Theano==0.8.0  --user --force-reinstall --upgrade`

6) Install the version of seya distributed with DanQ (for brnn)

   `tar -xvzf DanQ_seya.tar.gz`
   `cd DanQ_seya`
   `pip install . --upgrade --user --force-reinstall`
  `cd ..`

7) For some reason, CuDNN the CUDA neural network library, doesn't seem to want to play with this old version of Theano, so we'll skip this step for now ([see here](http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html#) for more information(?))

8) copy the `Theanorcs` folder to your desktop
This has .theanorc files for the two different GPU queues [click here for more info](http://deeplearning.net/software/theano/library/config.html#environment-variables)

`cp -r Theanorcs ~/`

   

After installation you should only have to run `module use Modules/rcc` followed by `module load DanQ/0.2.0` and you'll be ready to go

# Installation instructions for Midway2 (GPU2 queue)


1) Copy the directory `Modules` to your home directory

` cp -r Modules ~/`

2) Tell `module` about your new custom module folder (be sure *not* to put a trailing `/`)

   `module use $HOME/Modules/rcc`

3) load the RCC DanQ/0.2.0 module

   `module load DanQ/0.2.0`

4) Install the latest Keras
   `pip install keras --user --force-reinstall --upgrade`

5) Install Theano

 `pip install Theano  --user --force-reinstall --upgrade`

6) Install libgpuarray

see `scripts/install_libgpuarray.sh`


## Accessing the two GPU queues

Info about the GPU queues can be found [here](https://rcc.uchicago.edu/docs/using-midway/#types-of-compute-nodes)

Suffice to say that for the RCC/Midway1 queue use something like

`sinteractive --time=01:30:00 --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=30000 --partition=gpu`

And for the RCC/Midway2 queue use
`sinteractive --time=01:30:00 --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=30000 --partition=gpu2`







