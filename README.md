# Dashcam-Speed-Predictor-Keras
This is an attempt to create LCRN (CNN+LSTM) based Speed Predictor from Dashcam video data.
Uses video and files from Comma.Ai: http://commachallenge.s3-us-west-2.amazonaws.com/speed_challenge_2017.tar

Notes:
1. Developed an ego-speed predictor using Convolutions (feature extraction) and Bidirectional LSTMs to capture long range temporal correlations for accurate speed prediction. Since the timesteps = 10, and stride = 2, effectively, 1 second of real world data is captured for each prediction step. The training performance of extending to 20, 30 timesteps need to studied. More experiments being run.
2. The data for train.py is features (Resnet50 : without FC layers) extracted from frames of a training video shot at 20FPS.
3. train2.py is an end-to-end regressor.

Update :

A new version is in the works...
