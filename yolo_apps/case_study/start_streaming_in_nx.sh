#1. use decodebin docoder
gst-launch-1.0 -v udpsrc port=5000 ! "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse ! decodebin ! videoconvert ! autovideosink

#2. use avdec_h264 decoder
#gst-launch-1.0 -v udpsrc port=5000 ! "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink

#3. use omxh264dec decoder
#gst-launch-1.0 -v udpsrc port=5000 ! "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! autovideosink

#4. use nvv4l2 decoder (Don't work in opencv)
#gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! rtph264depay ! h264parse ! nvv4l2decoder ! videoconvert ! autovideosink

#@. display fps using decodebin decoder
#gst-launch-1.0 -v udpsrc port=5000 ! "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse ! decodebin ! videoconvert ! fpsdisplaysink text-overlay=false

