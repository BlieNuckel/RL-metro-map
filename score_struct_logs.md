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
    - other relevant details
- Summary of what issues demanded the change

# Versions

## **Version 1** | [069abb4](https://github.com/BlieNuckel/RL-metro-map/commit/069abb4f8ca4c5d6914eb7ff3a044908da601692)

Logs folder: RewardFunctions_v1

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
- The current direction of movement
