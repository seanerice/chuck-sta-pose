# Concept
Control a musical piece with gesture control. 

I will leverage existing tech solutions to design a multifaceted, neural-network-driven system which allows the user to conduct a musical piece with nothing more than a laptop and a webcam.

# Execution
There are three discrete components required to achieve the goal.

1. Audio backend
    * ChucK audio engine
    * Use gesture classification and params to control piece
2. Pose tracking software
    * https://github.com/tommymitch/posenetosc (based on TF PoseNet implementation)
    * Stream bone pose data out over OSC
    * Preprocess data
3. Gesture classifier
    * Multi-class gesture classifier
    * Classify sequences of pose data
   
# Install
1. Clone this repo.
2. Clone the repo https://github.com/tommymitch/posenetosc
   * Move the `osc.send(message)` on line 300 to line 301 so it is after the `}`. This ensures the message is sent correctly.
   
## Run
1. Run `chuck.sh`
2. Run `node bridge.js` in the `posenetosc` repo
3. Run `yarn watch` in the `posenetosc` repo. This will open up a window in your browser.
4. Load, train, and run the wekinator model in `/pose_parameters`.


## Pose Parameterization
Using Wekinators continuous model, I can supplant the gesture classifications with data about the pose.

Here are some of proposed parameters:

`hand_x`, `hand_y`: 0.0 - 1.0

`arm_angle`: 0.0 - 1.0

`elbow_angle`: 0.0 - 1.0
