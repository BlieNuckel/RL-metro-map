# Score Function Versions

This document contains information for each version of the scoring structure built. 

## Version Entry Structure

### Numbering

Versions are incremential (e.g. 1, 2, 3...) with no sub-versions for the sake of simplicity.

### Content

Each version entry must contain the following information:
- Commit number to find full code
- Name of logs folder in "version_logs"
- Descriptions of 
    - what changed in action space
    - what changed in observation space
    - what changed in state space
    - other relevant details
- Table reflecting changes to reward functions
- Summary of what issues demanded the change

# Versions

## **Version 1** | [069abb4](https://github.com/BlieNuckel/RL-metro-map/commit/069abb4f8ca4c5d6914eb7ff3a044908da601692)

Logs folder: RewardFunctions_v1\
_Model missing_

&nbsp;

### **Action Space**

The action space currently supports 6 actions:

- Move forward
- Turn 45° left and move forward
- Turn 90° left and move forward
- Turn 45° right and move forward
- Turn 90° right and move forward
- Place stop and move forward

### **Observation Space**

The observation space currently holds the following information:

- The board with the currently drawn fields
- The stops remaining in the current line
- The stops remaining in all the remaining lines
- The remaining number of lines
- The number of consecutive line overlaps
- The number of turns within the last X steps
- The max number of steps allowed (X in previous point)
- The current direction of movement\

### **State Space**

### **Reward Functions**

Currently implements the following rewards and punishment:

|Name|Reward function|
|----|---------------|
|Stop overlap|Punishment (-300) for placing a stop on top of another stop|
|Out of bounds|Punishment (-500) for moving out of bounds|
|Line overlap|Line overlapping punished exponentially harder, based on consecutive overlaps being drawn|
|Stop adjacency|Rewards placing the same stops next to each other, punishes if placed anywhere else|
|Relative Positions|Rewards relative angle to other stops with values between 0 and 1|
|Stop distribution|Rewards stop distribution based on if the stop is placed a correct distance apart from the previous placed stop. Punishes if incorrect distance is used|
|Minimize turns|Punishes turning too much based on a set "lookback" distance of moves|



## **Version 2** | [665d78c](https://github.com/BlieNuckel/RL-metro-map/commit/665d78cdaf782f7f2370ceaa598fa6f94ee4eb47)

Logs folder: RewardFunctions_v2\
_Model missing_

&nbsp;

### **Reward Functions**
|Name|Reward function|
|----|---------------|
|Line overlap|Now has a limit on how small a punishment it can give, clamping it to -60|
|Minimize turns|Negative limit implemented, clamping it to -40|
|**NEW**||
|Promote spreading|Reward function meant to promote the lines spreading out to fill available space. Adds a reward that grows and settles at a max, based on the current distance from the starting position|

### **Issues attempted to fix**
The previous build had a reward directly tied to the length of the episode. This makes sense as the more steps it takes the more mistakes it will produce.

Besides this all previous reward functions were clamped between 1 million and -1 million, but as some reward functions (line overlap, and minimize turns) went to -infinity it made for a poor reward function when values were clamped (as one good thing adding a rewrd wouldn't be visible).



## **Version 3** | [5d2bd34](https://github.com/BlieNuckel/RL-metro-map/commit/5d2bd3404ce7acf97026e983cd9f1093b4f68878)

Logs folder: RewardFunctions_v3\

&nbsp;

### **Reward Functions**
|Name|Reward function|
|----|---------------|
|Line overlap|Previous reward function incorrectly always punished the algorithm, as it was wrong. Updated to properly score lower when more than 1 crossing is done consecutively.|

All reward functions are now scoring between 1 and -1.

### **Issues attempted to fix**
The previous version still had the issue of reward being tied directly to episode length (inversely), to a higher degree than expected. 

The intention was to use pre-determined coefficients for each reward function to amplify the output (between 1 and -1) to have a bigger or smaller impact, but I forgot to connect the coefficients so they were unused while running.

### **Generated Map**
![generated map](./generated_maps/RewardFunctions_v3.png)



## **Version 4** | [caf0bc8](https://github.com/BlieNuckel/RL-metro-map/commit/caf0bc808931b9e894fd3aacdb4a60f0446dd67b)

Logs folder: RewardFunctions_v4\

&nbsp;

### **Other Changes**
Updated overlap checking when stepping, to ensure that if a line is drawn on top of an existing stop it will receive the stop overlap punishment. Previously stop overlap was only detected upon placing the stop.

### **Reward Functions**
|Name|Reward function|
|----|---------------|
|Stop overlap|Weight: 10|
|Out of bounds|Weight: 10|
|Line overlap|Weight: 8|
|Stop adjacency|Weight: 5|
|Relative Positions|Weight: 8|
|Stop distribution|Weight: 8|
|Minimize turns|Weight: 3|
|Promote spreading|Weight: 5|

All reward functions now have their approrpriate weights assigned.

### **Issues attempted to fix**
The error of not using the defined coefficients in the reward functions has been fixed, so now each function actually has a different weight in scoring.

The previous version featured some issues where stops and lines were placed on top of each other without it being an issue. This could be due to the weights not being present, but brought attention to an issue with overlaps only being detected if created in specific orders.



## **Version 5** | [02feccf](https://github.com/BlieNuckel/RL-metro-map/commit/02feccf5bd2df0d32edbdb6cfdf264a1f9bff41b)

Logs folder: RewardFunctions_v5\

&nbsp;

### **Observation Space**
The "board" is no longer hashed and returned as an ndarray. Rather a list of previous actions with an adjustable max count is used to infer what fields are already taken by stops and lines. This may be a little too indirect and there may be a demand for somehow returning a sparse version of the board where stops and lines are still mapped to coordinates.

Included a new field for the agent to know its current location. It already knew its current direction, but current location should help improve inference of overlap and out of bounds.

### **Reward Functions**
|Name|Reward function|
|----|---------------|
|Line overlap|This will now result in termination if more than 1 overlap occurs consecutively. This is because crossings are allowed, but lines should not be able to run along one another|
|Stop overlap|This will now result in termination if any stop is placed on top of an already existing stop or line. Similarly it terminates if a line is placed on an already placed stop|

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v5.png)
*Map generated from final model after 1,000,000 timesteps*

![best generated map](./generated_maps/RewardFunctions_v5_best_model.png)
*Map generated from best model after 700,000 timesteps*


### **Issues attempted to fix**
The previous version was training very slowly and had issues with replay buffer size, hence, the observation space restructure. The main issue this version aims to resolve besides faster train times was it would produce a model that would run in circles. This is believed to be because overlaps were only punished, but didn't end in termination.

Besides this the previous version also had some meta issues, relating to the training data not being randomized. This version has updated systems for loading data from JSON files, which will allow us to introduce new training maps when T-Kartor provides them.