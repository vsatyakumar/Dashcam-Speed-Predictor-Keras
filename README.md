# Dashcam-Speed-Predictor-Keras
This is an attempt to create LCRN (CNN+LSTM) based Speed Predictor from Dashcam video data.

Notes:
1. This in an attempt to create a car speed predictor using Convolutions and Bidirectional GRUs. 
2. The data for train.py is features (Resnet50 : without FC layers) extracted from frames of a training video shot at 20FPS.
3. train2.py is an end-to-end regressor.
