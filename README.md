
# Installation

First download this repo 
```
git clone https://github.com/benlawson/verbose-robot.git
```

### install Caffe 
this was quite difficult for me (see the [website](http://caffe.berkeleyvision.org/installation.html) for steps on how to do it - I think my computer is just too old)

### install OpenCV
assuming Anaconda and conda are installed:
```bash
conda install -c menpo opencv
```

### install other various python dependancies
```bash
pip install joblib numpy sklearn
```

### install darknet
```bash
git submodule update --init --recursive
cd yolo-9000/darknet
cd darknet 
make
```
*note:* to use the python bindings you will have to manually change the path in "./my_yolo/detector.py" to point "./yolo-9000/darknet/python"

### install rtree library
See the websites for more info
[libspatialindex](https://libspatialindex.github.io/install.html)

[Rtree](http://toblerity.org/rtree/)

```bash
git clone git@github.com:libspatialindex/libspatialindex.git
cd libspatialindex
autogen.sh
./configure
make
make install

easy_install Rtree
```

# slides
see [link](https://cs-people.bu.edu/balawson/cs585/project.html)


