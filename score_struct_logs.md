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

|Name|Reward function change|
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
|Name|Reward function change|
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
|Name|Reward function change|
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
|Name|Reward function change|
|----|---------------|
|Stop overlap|Weight: 10|
|Out of bounds|Weight: 10|
|Line overlap|Weight: 8|
|Stop adjacency|Weight: 5|
|Relative positions|Weight: 8|
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
|Name|Reward function change|
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



## **Version 6** | [f724864](https://github.com/BlieNuckel/RL-metro-map/commit/f724864e39f782b4be136ae2e8675dce8a24ec85)

Logs folder: RewardFunctions_v6\

&nbsp;

### **Observation Space**
The observation space now contains a list of the real-life stops' relative angles, to hopefully help point the agent to how it's scored when it places stops (as this is scored relative to the real stops' relative angles).

**Other Changes**

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Line overlap|Weight increased from 10 to 20 to have a higher impact|
|Stop overlap|Weight increased from 10 to 20 to have a higher impact|
|Out of bounds|Weight increased from 10 to 100 to have a higher impact|
|Relative positions|Weight increased from 9 to 14. This allows it to be much more impactful than other rules, as it is not following this at all currently.|
|Stop adjacency|Weight increased from 5 to 6|
|Stop distribution|Weight decreased from 5 to 4.5|
|Minimize turns|Weight increased from 3 to 3.5|
|Promote spreading|Weight decreased from 5 to 1.5. We just want it to be a suggestive reward, it's nowhere near as required as the other rules|
|**New**||
|Finished line|Gives a large reward when a line is completely finished|
|Time alive|Gives a scaled down linear reward with respect to the lifetime of the agent. This replaces "promote spreading" to helpfully encourage the agent to follow relative positions for placement learning, as opposed to just drawing straight lines outwards. Has a small weight of 1.5 to not override stop distribution to stay alive longer than needed.|
|**Removed**||
|Promote spreading|Removed in favor of "Time alive"|

### **Generated Maps**
![best_model](./generated_maps/RewardFunctions_v6_best_model.png)
*Best model created throughout the training of v6.*

### **Issues attempted to fix**
The previous version had a hard time learning to understand it should not run into itself, as this is now a terminal state. It was so bad it didn't even manage to get to the point where it would complete entire maps, much less getting to the point where other issues were considered such as spacing between stops or attempting to place stops correctly relatively.

Part of the issue might be the weights not being extreme enough on certain important aspects such as the relative positioning. Another issue could be that the list of previous actions doesn't appropriately suggest to the algorithm where it can and cannot go. Lastly it could be a lack of fore-sight (regarding stop placements), that results in the problem.



## **Version 7** | [f724864](https://github.com/BlieNuckel/RL-metro-map/commit/f724864e39f782b4be136ae2e8675dce8a24ec85)

Logs folder: RewardFunctions_v7\

&nbsp;

### **Other Changes**
New hyper parameters tested to see if learning could be improved.

learning_starts = 100000
learning_rate = 1e-4
target_update_interval = 1000
train_freq = 4
exploration_final_eps = 0.01

These hyperparameters are not optimized, but taken from an optimization of the DQN algorithm on another environment.

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v7_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v7_best_model.png)
*Best generated map made throughout training of v7*

### **Issues attempted to fix**
The previous version would complete the run, but would not manage to achieve any turning. Hyper parameters were changed in an attempt to help with learning speeds.



## **Version 8** | [0ce07ef](https://github.com/BlieNuckel/RL-metro-map/commit/0ce07ef83319fa45589268c23afd470646fa1a77)

Logs folder: RewardFunctions_v8\

&nbsp;

### **Observation Space**
The observation space has been updated in several ways. First off, the list of previous actions has been removed and replaced with 2 lists with 8 entries. These 8 entries describe in which directions of the current position there are lines and stops. A similar array of 8 entries has been added to describe the existence of out-of-bounds blocks next to the current position.

The relative angles observation has been updated to only include the angles of the next stop to place to all other stops. For the stops that have not yet been placed on the map, a normalized position of the stop's real position is used to find the angle. This is meant to hopefully give the algorithm a way to "look ahead" when placing things.

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v8_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v8_best_model.png)
*Best generated map made throughout training of v8*

### **Issues attempted to fix**
The previous version continued to exhibit similar issues as previously, where it would not understand how to avoid collisions. Similarly it continued to draw exclusively straight lines with no regard for the positions the stops should have.



## **Version 9** | [2814959](https://github.com/BlieNuckel/RL-metro-map/commit/2814959d73653e0d200a79ea5e43b987871b660a)

Logs folder: RewardFunctions_v9\

&nbsp;

### **Reward Functions**
|Name|Reward function change|
|Relative stop position|Changed weight to 4|
|Minimize turns|Changed weight to 10 because algorithm always draws squiggly lines|
|Time alive|Changed weight to 1, as algorithm seems to rely on this reward to get a high reward|

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v9_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v9_best_model.png)
*Best generated map made throughout training of v9*

The first time these models were run the images were not saved, but both reached much further than seen in these presented pictuers. The generated map by the best model was much akin to that of v8 with only the first line being drawn and squiggly lines, but with some stops being placed.

### **Issues attempted to fix**
v8 had better luck than previous versions in attempting to move away from simply straight lines. It seems, however, it has become overreliant on the "time alive" reward function, which was meant for previous versions to help learn avoiding death.

There also seemed to be some inconsistencies with turns not being punished enough (squiggly lines), but this could be an artifact of the "stay alive" reward.



## **Version 10** | []()

Logs folder: RewardFunctions_v10\

&nbsp;

### **Reward Functions**
|Name|Reward function change|
|Relative stop positions|Changed allowed range with full reward to be +-90 degrees compared to real positions. This should hopefully mean that if a position is placed South in reality should preferably be placed SE, S, or SW in the drawing. Also updated the first stop of each line to always return an angle difference of 0, to promote placing it immediately.|
|**Removed**||
|Time alive|The reward was based on an increasing value over episode (not per-step), which caused it to be hugely impactful the further it stepped. Removed to test |

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v10_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v10_best_model.png)
*Best generated map made throughout training of v10*


### **Issues attempted to fix**
Previous version showed very similar issues to v8, with only drawing a single line and this time adding a few stops. Similarly the "time alive" reward was skewing the resulting line because the reward grew disproportionately with every step taken.

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

|Name|Reward function change|
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
|Name|Reward function change|
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
|Name|Reward function change|
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
|Name|Reward function change|
|----|---------------|
|Stop overlap|Weight: 10|
|Out of bounds|Weight: 10|
|Line overlap|Weight: 8|
|Stop adjacency|Weight: 5|
|Relative positions|Weight: 8|
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
|Name|Reward function change|
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



## **Version 6** | [f724864](https://github.com/BlieNuckel/RL-metro-map/commit/f724864e39f782b4be136ae2e8675dce8a24ec85)

Logs folder: RewardFunctions_v6\

&nbsp;

### **Observation Space**
The observation space now contains a list of the real-life stops' relative angles, to hopefully help point the agent to how it's scored when it places stops (as this is scored relative to the real stops' relative angles).

**Other Changes**

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Line overlap|Weight increased from 10 to 20 to have a higher impact|
|Stop overlap|Weight increased from 10 to 20 to have a higher impact|
|Out of bounds|Weight increased from 10 to 100 to have a higher impact|
|Relative positions|Weight increased from 9 to 14. This allows it to be much more impactful than other rules, as it is not following this at all currently.|
|Stop adjacency|Weight increased from 5 to 6|
|Stop distribution|Weight decreased from 5 to 4.5|
|Minimize turns|Weight increased from 3 to 3.5|
|Promote spreading|Weight decreased from 5 to 1.5. We just want it to be a suggestive reward, it's nowhere near as required as the other rules|
|**New**||
|Finished line|Gives a large reward when a line is completely finished|
|Time alive|Gives a scaled down linear reward with respect to the lifetime of the agent. This replaces "promote spreading" to helpfully encourage the agent to follow relative positions for placement learning, as opposed to just drawing straight lines outwards. Has a small weight of 1.5 to not override stop distribution to stay alive longer than needed.|
|**Removed**||
|Promote spreading|Removed in favor of "Time alive"|

### **Generated Maps**
![best_model](./generated_maps/RewardFunctions_v6_best_model.png)
*Best model created throughout the training of v6.*

### **Issues attempted to fix**
The previous version had a hard time learning to understand it should not run into itself, as this is now a terminal state. It was so bad it didn't even manage to get to the point where it would complete entire maps, much less getting to the point where other issues were considered such as spacing between stops or attempting to place stops correctly relatively.

Part of the issue might be the weights not being extreme enough on certain important aspects such as the relative positioning. Another issue could be that the list of previous actions doesn't appropriately suggest to the algorithm where it can and cannot go. Lastly it could be a lack of fore-sight (regarding stop placements), that results in the problem.



## **Version 7** | [f724864](https://github.com/BlieNuckel/RL-metro-map/commit/f724864e39f782b4be136ae2e8675dce8a24ec85)

Logs folder: RewardFunctions_v7\

&nbsp;

### **Other Changes**
New hyper parameters tested to see if learning could be improved.

learning_starts = 100000
learning_rate = 1e-4
target_update_interval = 1000
train_freq = 4
exploration_final_eps = 0.01

These hyperparameters are not optimized, but taken from an optimization of the DQN algorithm on another environment.

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v7_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v7_best_model.png)
*Best generated map made throughout training of v7*

### **Issues attempted to fix**
The previous version would complete the run, but would not manage to achieve any turning. Hyper parameters were changed in an attempt to help with learning speeds.



## **Version 8** | [0ce07ef](https://github.com/BlieNuckel/RL-metro-map/commit/0ce07ef83319fa45589268c23afd470646fa1a77)

Logs folder: RewardFunctions_v8\

&nbsp;

### **Observation Space**
The observation space has been updated in several ways. First off, the list of previous actions has been removed and replaced with 2 lists with 8 entries. These 8 entries describe in which directions of the current position there are lines and stops. A similar array of 8 entries has been added to describe the existence of out-of-bounds blocks next to the current position.

The relative angles observation has been updated to only include the angles of the next stop to place to all other stops. For the stops that have not yet been placed on the map, a normalized position of the stop's real position is used to find the angle. This is meant to hopefully give the algorithm a way to "look ahead" when placing things.

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v8_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v8_best_model.png)
*Best generated map made throughout training of v8*

### **Issues attempted to fix**
The previous version continued to exhibit similar issues as previously, where it would not understand how to avoid collisions. Similarly it continued to draw exclusively straight lines with no regard for the positions the stops should have.



## **Version 9** | [2814959](https://github.com/BlieNuckel/RL-metro-map/commit/2814959d73653e0d200a79ea5e43b987871b660a)

Logs folder: RewardFunctions_v9\

&nbsp;

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Relative stop position|Changed weight to 4|
|Minimize turns|Changed weight to 10 because algorithm always draws squiggly lines|
|Time alive|Changed weight to 1, as algorithm seems to rely on this reward to get a high reward|

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v9_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v9_best_model.png)
*Best generated map made throughout training of v9*

The first time these models were run the images were not saved, but both reached much further than seen in these presented pictuers. The generated map by the best model was much akin to that of v8 with only the first line being drawn and squiggly lines, but with some stops being placed.

### **Issues attempted to fix**
v8 had better luck than previous versions in attempting to move away from simply straight lines. It seems, however, it has become overreliant on the "time alive" reward function, which was meant for previous versions to help learn avoiding death.

There also seemed to be some inconsistencies with turns not being punished enough (squiggly lines), but this could be an artifact of the "stay alive" reward.



## **Version 10** | [fe1460a](https://github.com/BlieNuckel/RL-metro-map/commit/fe1460a6265f8ec93a53a09f4eb7c0b10cc099e2)

Logs folder: RewardFunctions_v10\

&nbsp;

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Relative stop positions|Changed allowed range with full reward to be +-90 degrees compared to real positions. This should hopefully mean that if a position is placed South in reality should preferably be placed SE, S, or SW in the drawing. Also updated the first stop of each line to always return an angle difference of 0, to promote placing it immediately.|
|**Removed**||
|Time alive|The reward was based on an increasing value over episode (not per-step), which caused it to be hugely impactful the further it stepped. Removed to test |

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v10_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v10_best_model.png)
*Best generated map made throughout training of v10*


### **Issues attempted to fix**
Previous version showed very similar issues to v8, with only drawing a single line and this time adding a few stops. Similarly the "time alive" reward was skewing the resulting line because the reward grew disproportionately with every step taken.



## **Version 11** | [fe1460a](https://github.com/BlieNuckel/RL-metro-map/commit/fe1460a6265f8ec93a53a09f4eb7c0b10cc099e2)

Logs folder: RewardFunctions_v11\

&nbsp;

**Other Changes**
Used QRDQN algorithm for training, rather than DQN. Everything is the same as v10, to allow for more direct comparison.

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v11_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v11_best_model.png)
*Best generated map made throughout training of v11*


### **Issues attempted to fix**
No immediate issues, simply an investigation of QRDQN vs DQN.



## **Version 12** | [3a0df90](https://github.com/BlieNuckel/RL-metro-map/commit/3a0df904f4a2aab86413264635112f0a59c44872)

Logs folder: RewardFunctions_v12\

&nbsp;

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Minimize turns|Changed allowed turn from +-90 to +-22.5, as it was too lenient, when stops only had to be placed North or South of each other.|

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v12_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v12_best_model.png)
*Best generated map made throughout training of v12*


### **Issues attempted to fix**
Was still producing too straight lines, when it really needed to be making some more turns.



## **Version 13** | [d3a808a](https://github.com/BlieNuckel/RL-metro-map/commit/d3a808af2e860c49ade95c24369ba51431b96749)

Logs folder: RewardFunctions_v13\

&nbsp;

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Relative stop positions|Changed allowed turn from +-90 to +-22.5, as it was too lenient, when stops only had to be placed North or South of each other.|
|Minimize turns|Changed score function to be based on the total angle of turns in the previous X steps, rather than the number of turns. Hopefully this will help avoid things like loops or 180 turns.|

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v13_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v13_best_model.png)
*Best generated map made throughout training of v13*


### **Issues attempted to fix**
Previous version traveled far in straight lines. This was likely an issue caused by the positive reinforcement given by "minimize turns" where the algorithm would receive a positive reward every time it didn't turn.



## **Version 14** | [4b5fb22](https://github.com/BlieNuckel/RL-metro-map/commit/4b5fb2219aa334e0fe09732ab745532ee6bdcb80)

Logs folder: RewardFunctions_v14\

&nbsp;

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Stop adjacency|Changed to giving 1 when placing correctly instead of 2/5.|
|Stop distribution|Changed to reward on every step the algorithm is within the target stop distribution. The reward will then fall liniearly on any step over this to allow for small variations, to achieve adjacency.|

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v14_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v14_best_model.png)
*Best generated map made throughout training of v14*


### **Issues attempted to fix**
Previous version ended its training as fast as possible, because rewards are too punishing. Stops need to be forced to be placed to receive decent rewards. This will hopefully train it to find a better way of placing the stops to get more points.



## **Version 15** | [1941cc4](https://github.com/BlieNuckel/RL-metro-map/commit/1941cc4043f89cef4def660c8bd939297922b8e2)

Logs folder: RewardFunctions_v15\

&nbsp;

### **General Notes**
This version has been through many revisions as results were consistently showing the same signs as seen in previous versions. As such this version is actually the culmination of 4-5 failed versions that in the end amounted to the version documented here. The subversions were interesting, but often revealed nothing new in terms of relationships between rewards and outcomes.

### **Observation Space**
The major change that made this version perform better than the previous, was an optimization of the use of the angles. It was discovered that scaling the stop distribution down was done incorrectly and generally wasn't sustainable in most cases.

The system was simplified and now scores based on the angle of the stop to be placed to all other already placed stops. This does mean there's a lack of "future" insight, however, it does provide better results than attempting to use scaling for "future" insight.

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Time Alive|Re-introduced, this time with a score based on the current timestep being <= total_stop_count * expected_stop_distribution. This rewards taking steps within the expected length of the run, but punishes going on for longer than needed.|
|Relative stop positions|Was expanded to allow placement within 90 degree difference as opposed to only 45 degree placement. (Commit has this wrong, saying 22.5. 45 is what was really used for this version.)|
|**New**||
|Stop placed|A base reward for placing a stop to help avoid never placing stops.|


### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v15_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v15_best_model.png)
*Best generated map made throughout training of v15*


### **Issues attempted to fix**
v14 returned to the pattern of ending each episode as fast as possible. This is most likely due to life being too punishing and it already reaching a decent score out of the gate. I am at a bit of a loss as to how to go about changing its behaviour withuot re-introducing some of the previous issues where it would farm rewards in different ways.



## **Version 16** | [ed4b58b](https://github.com/BlieNuckel/RL-metro-map/commit/ed4b58bc923fe0be32341af8a235493c7503614a)

Logs folder: RewardFunctions_v16\

&nbsp;

### **Other Changes**
A more appropriate use of random has been implemented when picking random input data and random input values (the few values that are at all randomized).

### **Observation Space**
Introduced new observation value "next_stop_direction", which transforms the mean angle currently used as a pointer for the next stop's position into a octidirectional direction. Hopefully this can help create a better correlation between the amount of angle and the best direction to go.

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Stop distribution|Weight changed to 8.5 as this was not previously respected.|
|Stop adjacency|Weight changed to 10 to accomodate "stop distribution" weight increase.|
|Stop relative positions|Weight changed to 6.5. Hopefully the small increase will help weigh it slightly higher and give higher corner resolution.|
|Minimize turns|Weight changed to 3.5 to not fall too behind the newly increased weights for other stop related rewards. Has not been increased as much to achieve slightly higher corner resolution.|

Changed angle difference to only get max reward when within +-22.5 of real angle, to promote adhering more to real stop placements.

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v16_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v16_best_model.png)
*Best generated map made throughout training of v16*


### **Issues attempted to fix**
v15 brought an increase in stop placement accuracy, but still exhibits some behaviors that should be reconsidered.

Stop distancing is still not following appropriate spacing rules, relative stop positioning doesn't quite achieve the level of distinguishing that could be wanted, and the lines that do turn, seem to do so in weird/unexpected angles or directions.



## **Version 17** | [8a397ad](https://github.com/BlieNuckel/RL-metro-map/commit/8a397ad769ae579b6351e4b263cfdbdaf94b05eb)

Logs folder: RewardFunctions_v17\

&nbsp;

### **Observation Space**
Added two new observation fields: "nearest_adjacent_position" and "adjacent_to_other_stop" which help find the distance to the nearest adjacent spot and tell if the agent is currently next to another stop respectively. These two will hopefully help achieve better adjacent placement.

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Stop distribution|Weight increased to 10. Function will now reward increasingly while under the wished for distribution and sharply decreasingly once the wished for distribution has been passed.|
|Stop adjacency|Weight increased to 12. Function will now punish half if stop is not placed next to any other stop and fully if placed next to an incorrect stop.|
|Stop relative positions|Weight increased to 10.|
|Minimize turns|Lowered weight from 3.5 to 2, as there is still very little turning happening.|

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v17_final_model.png)
*Final generated map at 2 million timesteps*

![best generated map](./generated_maps/RewardFunctions_v17_best_model.png)
*Best generated map made throughout training of v17*


### **Issues attempted to fix**
v16 turned out worse than v15 in resemlance of the actual geographical maps, which is overall fine. It did, however, also lose most other variation and returned to straight lines.

The main issues boil down to not turning enough and still not appropriately placing stops, both based on spacing and adjacency.



## **Version 19** | []()

Logs folder: RewardFunctions_v19\

&nbsp;

### **Environment State**
The state has been updated to support infinite drawings (no boundaries) up until a specified maximum steps. This removes the entire grid functionality of the application.

Outside of this, all the data points that must be drawn are now scaled down and centralized around 0. These positions are then used as guides for approximately where to put the stops by the agent. This is intended to achieve similar results as with the angles, but hopefully more accurate.

### **Observation Space**
Removed the following observations as they did not mesh well with the new infinite grid or failed to work with the stop location-based placement: stop_spacing, steps_since_stop, stop_angle_diff, next_stop_direction, and out_of_bounds_in_adjacent_fields.

Added the following observations: next_stop_distance.

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|**New**||
|Distance to real stop|Reward that functions similar to "stop relative position" by scoring the agent positively at each step if it gets closer to the real stop position and punishing if it moves further away. Once within a 25 unit radius, the reward is a full 1 to "minimize turns" conflicts.|
|**Removed**||
|Time alive|This did not fit the new infinite grid, as stop spacing has been temporarily disregarded in an attempt to achieve any decent map output.|
|Stop relative position|Angle based version removed and replaced with "distance to real stop"|
|Stop distribution|Temporarily removed to attempt to achieve better maps with real positions.|

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v18_final_model.png)\
*Final generated map at 2 million timesteps. It is very hard to see, but it is simply a very long straight line.*

![best generated map](./generated_maps/RewardFunctions_v18_best_model.png)\
*Best generated map made throughout training of v18*


### **Issues attempted to fix**
v17 started to show some promising behaviour of actually turning, but still seemed to get directions and such wrong. Similarly the stop adjacency is still problematic and not working as intended.



## **Version 19** | [b4e1fbf](https://github.com/BlieNuckel/RL-metro-map/commit/b4e1fbf8029f9f016f410e609f2f19833f214e8d)

Logs folder: RewardFunctions_v19\

&nbsp;

### **Observation Space**
Stop adjacency was triggering one too late and therefore giving incorrect information regarding the current positions adjacency. This is fixed.

Real stop distance will now always be 0 for the first stop in a line, to promote placing it right at the beginning.

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|**Removed**||
|Minimize Turns|Temporarily disabled to hopefully achieve better results (more geographically accurate) with less fully straight lines.|

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v19_final_model.png)\
*Final generated map at 2 million timesteps.*

![best generated map](./generated_maps/RewardFunctions_v19_best_model.png)\
*Best generated map made throughout training of v19*


### **Issues attempted to fix**
v18 ended up killing itself in the best version and draws very long, straight lines.



## **Version 20** | [c74d7d2](https://github.com/BlieNuckel/RL-metro-map/commit/c74d7d22bfb4606206b42fd87790ea3a20aec2e9)

Logs folder: RewardFunctions_v20\

&nbsp;

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Stop Placed|No longer unconditional reward for placing stop, but will reward 0 if not within a 25 unit radius of the real stop's position.|

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v20_final_model.png)\
*Final generated map at 2 million timesteps.*

![best generated map](./generated_maps/RewardFunctions_v20_best_model.png)\
*Best generated map made throughout training of v20*


### **Issues attempted to fix**
The straight lines were not gone after removing "minimize turns", this means something else is behind the straight lines forming. It is likely a result of the base reward for placing a stop.



## **Version 21** | [66d9272](https://github.com/BlieNuckel/RL-metro-map/commit/66d92729cbc1ae10643bb7f550a267599d6e9752)

Logs folder: RewardFunctions_v21\

&nbsp;

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Stop adjacent|Stop scoring stop adjacent on the first placement of the stop, as it was giving a positive reward.|
|Distance to real stop|Increased weight from 10 to 30, to hopefully promote finding correct spot for stop.|

### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v21_final_model.png)\
*Final generated map at 2 million timesteps.*

![best generated map](./generated_maps/RewardFunctions_v21_best_model.png)\
*Best generated map made throughout training of v21*


### **Issues attempted to fix**
v20 still struggles with straight lines and not discovering where the right placement for the stops are.



## **Version 22** | []()

Logs folder: RewardFunctions_v22\

&nbsp;

### **Environment State**
The normalization of stop positions was invalid and created random variation between each reset. This lead to never training on the same layout, which obviously caused some issues with drawing valid maps.

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Distance to real stop|Previous setup promoted looping around the stops, as any movement inside the 25 unit radius gave a high reward. Reward is now 0 when within 25 units.|


### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v22_final_model.png)\
*Final generated map at 2 million timesteps.*

![best generated map](./generated_maps/RewardFunctions_v22_best_model.png)\
*Best generated map made throughout training of v22*


### **Issues attempted to fix**
Both v20 and v21 includes loops and v21 distinctly avoids placing stops.



## **Version 23** | []()

Logs folder: RewardFunctions_v23\

&nbsp;

### **Environment State**
The normalization of stop positions was invalid and created random variation between each reset. This lead to never training on the same layout, which obviously caused some issues with drawing valid maps.

### **Reward Functions**
|Name|Reward function change|
|----|---------------|
|Distance to real stop|This was accidentally punishing for every stop placed, because of the "move forward" action that happens after every stop placement. This resulted in never placing stops, as it was too punishing. Now avoids this.|
|Stop placed|Now punishes rather than giving 0, when placing a stop too far from its real position.|


### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v23_final_model.png)\
*Final generated map at 2 million timesteps.*

![best generated map](./generated_maps/RewardFunctions_v23_best_model.png)\
*Best generated map made throughout training of v23*


### **Issues attempted to fix**
v22 continued the pattern of creating loops, but this time discovered that moving too far was bad. Still avoids placing stops.



## **Version 24** | []()

Logs folder: RewardFunctions_v24\

&nbsp;

### **Observation Space**
Removed observations that were previously used for scoring functions, that have been temporarily disabled, in an attempt to simplify structure of environment.


### **Generated Maps**
![final generated map](./generated_maps/RewardFunctions_v24_final_model.png)\
*Final generated map at 2 million timesteps.*

![best generated map](./generated_maps/RewardFunctions_v24_best_model.png)\
*Best generated map made throughout training of v24*


### **Issues attempted to fix**
v21, 22, and 23 all avoid placing stops. 23 starts approaching a valid-ish shape of the path, but stil while not placing any stops.