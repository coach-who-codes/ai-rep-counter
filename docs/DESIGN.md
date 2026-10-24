# Design notes

Decisions I made, the alternatives I considered, and why.

## 1. Pretrained pose model + rules, not an end-to-end classifier

**Option A - train a video model** (e.g. a 3D-CNN that outputs "rep / no rep" and "good / bad form").
Needs thousands of labelled clips, a GPU, and it's a black box when it's wrong.

**Option B (chosen) - pretrained pose estimation + interpretable rules.**
MediaPipe already solves the hard perception problem (where are the joints?). The fitness logic on top is
geometry a coach can read and tune: "a squat is full depth when the knee angle is below 100 degrees".
It runs in real time on a laptop CPU, and every decision can be explained to a client.

The ML upgrade path is clear: the rule engine already logs every rep's angle trace, so once I label my
clients' reps I can train a sequence model (1D-CNN / GRU) on those traces and compare it to the rules.

## 2. Pixel coordinates, not normalised ones

MediaPipe returns x and y in 0..1, but x is scaled by frame width and y by frame height. On a 16:9 video
an angle computed from normalised coordinates is distorted. Converting to pixels first fixes that.

## 3. Hysteresis via three thresholds

Counting "every time the angle drops below 100" double counts when the angle wobbles around 100.
The state machine needs the angle to pass *top -> rep-started -> full-depth -> top*, so small wobbles
can't create a rep. The gap between thresholds is the hysteresis band. Reps shorter than 0.4 s are
rejected as tracking glitches.

## 4. EMA smoothing

Keypoints jitter by a few pixels per frame. An exponential moving average (alpha = 0.4) removes most of
it with ~2 frames of lag - fine at 30 fps. A Kalman or One-Euro filter would track fast movement
better; EMA was the simplest thing that worked.

## 5. Side selection

Filmed side-on, one side of the body is hidden and its keypoints have low visibility. The counter picks
the side with higher total visibility for the joints that exercise needs, and only switches side
*between* reps so the angle can't jump mid-rep.

## 6. Testing with a kinematic simulator

Real video has no ground-truth angle. `simulate.py` builds skeletons from limb lengths and target angles
(forward kinematics), so tests know the exact answer. This lets CI test counting, half reps, jitter,
dropped frames, low visibility and each form rule - without a camera or model download.

## 7. Scoring

Each completed rep starts at 100 and loses 25 per distinct fault (a fault repeated across frames in the
same rep counts once). A half rep scores 0. Session form score = mean over all attempts, so half reps
pull it down - which matches how a coach would judge the set.

## Known limitations

- 2D angles depend on camera angle (side-on works best).
- One person per frame.
- Thresholds are coaching heuristics, not learned from data.
- Lighting, baggy clothes and occlusion (e.g. a bench) reduce keypoint confidence.
