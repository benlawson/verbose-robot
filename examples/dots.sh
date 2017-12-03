rm data/*.jpg
python dots.py
ffmpeg -pattern_type glob -framerate 30 -i "data/*.jpg" output.mp4 -y
