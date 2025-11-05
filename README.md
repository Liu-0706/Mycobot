Project Introduction:

This project aims to model and compensate for the motion errors and loose connections of the MyCobot280Pi robot arm through machine learning methods (such as Gaussian process regression and Kalman filter), thereby improving the accuracy and stability of the end effector.

Environment:

pip install -r requirements.txt

move.py does not use the model, but only uses the Mycobot native code to move. move_gpr.py uses the Gaussian Process Regression (GPR) model and the Kalman Filter (KF).
Run move.py and move_gpr.py separately to compare the differences
