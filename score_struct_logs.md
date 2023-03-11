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

### Action Space

The action space currently supports 6 actions:

- Move forward
- Turn 45° left and move forward
- Turn 90° left and move forward
- Turn 45° right and move forward
- Turn 90° right and move forward
- Place stop and move forward

### Observation Space

The observation space currently holds the following information:

- The board with the currently drawn fields
- The stops remaining in the current line
- The stops remaining in all the remaining lines
- The remaining number of lines
- The number of consecutive line overlaps
- The number of turns within the last X steps
- The max number of steps allowed (X in previous point)
- The current direction of movement\

### State Space

### Reward Functions

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

### Action Space
No changes to the action space

### Observation Space
No changes to the observation space

### State Space
No changes to the state space

### Reward Functions
|Name|Reward function|
|----|---------------|
|Line overlap|Now has a limit on how small a punishment it can give, clamping it to -60|
|Minimize turns|Negative limit implemented, clamping it to -40|
|**NEW**||
|Promote spreading|Reward function meant to promote the lines spreading out to fill available space. Adds a reward that grows and settles at a max, based on the current distance from the starting position|

### Issues attempted to fix
The previous build had a reward directly tied to the length of the episode. This makes sense as the more steps it takes the more mistakes it will produce.

Besides this all previous reward functions were clamped between 1 million and -1 million, but as some reward functions (line overlap, and minimize turns) went to -infinity it made for a poor reward function when values were clamped (as one good thing adding a rewrd wouldn't be visible).



