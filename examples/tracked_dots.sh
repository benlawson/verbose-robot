rm data/*.jpg
python tracked_dots.py
ffmpeg -pattern_type glob -framerate 30 -i "data/*.jpg" tracked_output.mp4 -y
