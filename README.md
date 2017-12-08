# file useage
 
0. generate scene video

0a. dots example: run ```python examples/dots.py```
    then ```ffmpeg -pattern_type glob -framerate 30 -i "data/*.jpg" dots.mp4 -y```

0b. youtube video, watch the video as normal 
the following steps set up the associate code
(this produces a detections.joblib file)
```
cd my_yolo
youtube-dl <video_url>
mv <downloaded video> scene.mp4
mkdir temp_photos
cd temp_photos
ffmpeg -i scene.mp4 -r 4/1 $filename%05d.jpg
python detector.py
```

on computer to capture videos (opencv required)
1. run ```python watch_video.py```

2. play video once the webcam light begins
(realistically you can do this all on one computer (but chromebook low powered))

3. copy video file "video.joblib" using scp 

3.5. rename "video.joblib" to "example.joblib"

4. run ```python scripts/joblib2jpg.py```

5. run ```python main.py```
The gaze predictions will be saved to "predictions.joblib"

6a. to generate dots demo video with predictions, run ```python examples/tracked_dots.py```

6b. to generate object detection video, after step 0b, run ```python examples/yolo_video.py```


7. to make a stand-alone face/eye detection use ```python scripts/make_detections.py```


# ffmpeg usage
to extract frames from a video at 4 frames a second (4/1) 
```bash
ffmpeg -i scene.mp4 -r 4/1 $filename%05d.jpg
```
to merge all jpgs into an mp4
```bash
 ffmpeg -pattern_type glob -framerate 4 -i "test/*.jpg" detections_pacrim.mp4 -y
```



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


