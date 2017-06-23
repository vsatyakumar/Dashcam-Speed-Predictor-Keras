# Dashcam-Speed-Predictor-Keras
This is an attempt to create LCRN (CNN+LSTM) based Speed Predictor from Dashcam video data.

Notes:
1. TDeveloped a ego-speed predictor using Convolutions (feature extraction) and Bidirectional LSTMs to capture long range temporal correlations for accurate speed prediction. Since the timesteps = 10, and stride = 2, effectively, 1 second of real world data is captured for reach prediction. Moreover, the effects of extending to 20, 30 timesteps need to studied. More experiments being run.
2. The data for train.py is features (Resnet50 : without FC layers) extracted from frames of a training video shot at 20FPS.
3. train2.py is an end-to-end regressor.

Update :

A new version is in the works...
