scheduler : scheduler.cc
	g++ -o scheduler scheduler.cc -I/usr/include/opencv4\
		-pthread -g -I/home/nvidia/FBF-TF\
		-I/home/nvidia/FBF-TF/tensorflow/lite/tools/make/downloads/flatbuffers/include\
		-L/home/nvidia/FBF-TF/tensorflow/lite/tools/make/gen/linux_aarch64/lib\
		-I/home/nvidia/FBF-TF/tensorflow/lite/tools/make/downloads/absl\
		-L/home/nvidia/FBF-TF/tensorflow/lite/tools/make/downloads/flatbuffers/build\
		-lopencv_gapi\
		-ltensorflow-lite\
		-lflatbuffers /lib/aarch64-linux-gnu/libdl.so.2\
		-lopencv_stitching -lopencv_alphamat -lopencv_aruco -lopencv_bgsegm\
		-lopencv_bioinspired -lopencv_ccalib\
		-lopencv_dnn_objdetect\
		-lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype\
		-lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_intensity_transform\
		-lopencv_line_descriptor -lopencv_mcc -lopencv_quality -lopencv_rapid -lopencv_reg\
		-lopencv_rgbd -lopencv_saliency -lopencv_sfm -lopencv_stereo -lopencv_structured_light\
		-lopencv_phase_unwrapping -lopencv_superres -lopencv_surface_matching\
		-lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_videostab\
		-lopencv_optflow -lopencv_videoio\
		-lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect\
		-lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann\
		-lopencv_xphoto -lopencv_photo -lopencv_imgproc\
		-lopencv_core\
		/home/nvidia/FBF-TF/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so\
		/home/nvidia/FBF-TF/bazel-bin/tensorflow/lite/delegates/xnnpack/libxnnpack_delegate.so\
		/usr/lib/aarch64-linux-gnu/libEGL.so\
		/usr/lib/aarch64-linux-gnu/libGL.so /usr/lib/aarch64-linux-gnu/libGLESv2.so
	

scheduler_odroid : scheduler.cc
	g++ -o scheduler scheduler.cc -I/usr/include/opencv4\
		-pthread -g -I/home/odroid/FBF-TF\
		-I/home/odroid/FBF-TF/tensorflow/lite/tools/make/downloads/flatbuffers/include\
		-L/home/odroid/FBF-TF/tensorflow/lite/tools/make/gen/bbb_armv7l/lib\
		-I/home/odroid/FBF-TF/tensorflow/lite/tools/make/downloads/absl\
		-L/home/odroid/FBF-TF/tensorflow/lite/tools/make/downloads/flatbuffers/build\
		-lopencv_gapi\
		-ltensorflow-lite\
		-lflatbuffers /lib/arm-linux-gnueabihf/libdl.so.2\
		-lopencv_stitching -lopencv_aruco -lopencv_bgsegm\
		-lopencv_bioinspired -lopencv_ccalib\
		-lopencv_dnn_objdetect\
		-lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype\
		-lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_intensity_transform\
		-lopencv_line_descriptor -lopencv_mcc -lopencv_quality -lopencv_rapid -lopencv_reg\
		-lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light\
		-lopencv_phase_unwrapping -lopencv_superres -lopencv_surface_matching\
		-lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_videostab\
		-lopencv_optflow -lopencv_videoio\
		-lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect\
		-lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann\
		-lopencv_xphoto -lopencv_photo -lopencv_imgproc\
		-lopencv_core\
		/home/odroid/FBF-TF/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so\
		/home/odroid/FBF-TF/bazel-bin/tensorflow/lite/delegates/xnnpack/libxnnpack_delegate.so\
		/usr/lib/arm-linux-gnueabihf/libEGL.so\
		/usr/lib/arm-linux-gnueabihf/libGL.so /usr/lib/arm-linux-gnueabihf/libGLESv2.so
