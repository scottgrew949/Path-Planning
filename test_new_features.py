import sys
sys.path.insert(0, '.')
import pathplanning

env = pathplanning.GridEnvironment(21,21,1,1,19,19,0.15,42)
env.updateBelief(5,5,True,0.9,0.1)
env.updateBelief(5,5,True,0.9,0.1)
env.updateBelief(5,5,True,0.9,0.1)
print("3 hits p=", round(env.getBeliefAt(5,5),4), "l=", round(env.getLogOddsAt(5,5),3))
env.updateBelief(6,6,False,0.9,0.1)
print("1 miss p=", round(env.getBeliefAt(6,6),4), "l=", round(env.getLogOddsAt(6,6),3))

# CBS on smaller grid — space-time search is O(W*H*T) per agent per CT node
env_small = pathplanning.GridEnvironment(11,11,1,1,9,9,0.15,42)
paths = env_small.findPathsCBS([[1,1,9,9],[9,1,1,9],[1,5,9,5]])
print("CBS agents:", len(paths), "CT nodes:", env_small.getCBSNodesExpanded())
for i,p in enumerate(paths):
    print("  agent", i, len(p), "steps")
max_t = max(len(p) for p in paths)
conflicts = sum(
    1 for t in range(max_t)
    for i in range(len(paths)-1)
    for j in range(i+1,len(paths))
    if paths[i][min(t,len(paths[i])-1)] == paths[j][min(t,len(paths[j])-1)]
)
print("vertex conflicts:", conflicts)
