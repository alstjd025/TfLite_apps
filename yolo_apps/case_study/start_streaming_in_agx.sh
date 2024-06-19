#1. slow (framerate 60/1 does not work)
#gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! nvvidconv ! x264enc bitrate=16000000 speed-preset=superfast ! rtph264pay ! udpsink port=5000 host=192.168.0.3

#2. fast (use nvv4l2h64 encoder)
#gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=30/1, format=NV12' ! nvvidconv flip-method=2 ! nvv4l2h264enc insert-sps-pps=true bitrate=16000000 ! rtph264pay ! udpsink port=5000 host=192.168.0.3

#3. fast (use omxh264 encoder)
#gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=30/1, format=NV12' ! omxh264enc insert-sps-pps=true bitrate=16000000 ! rtph264pay ! udpsink port=5000 host=192.168.0.3

#Experiment
#gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=15/1, format=NV12' ! omxh264enc insert-sps-pps=true bitrate=16000000 ! rtph264pay ! udpsink port=5000 host=192.168.0.3

#Experiment
#gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=30/1, format=NV12' ! x264enc insert-sps-pps=true bitrate=16000000 ! rtph264pay ! udpsink port=5000 host=192.168.0.3


#Experiment
#gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=30/1, format=NV12, display_width=416, display_height=416' ! omxh264enc insert-sps-pps=true bitrate=16000000 ! rtph264pay ! udpsink port=5000 host=192.168.0.3

#Experiment [FINAL]
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12' ! nvvidconv ! 'video/x-raw, width=416, height=416' ! omxh264enc insert-sps-pps=true bitrate=16000000 ! rtph264pay ! udpsink port=5000 host=192.168.0.3

