================================================================================
  ROBOT PATH PLANNING & REINFORCEMENT LEARNING SYSTEM
================================================================================
  PROJECT OVERVIEW
--------------------------------------------------------------------------------

A ground-up C++ implementation of a robot navigation system that bridges
classical AI planning algorithms and modern reinforcement learning. The project
simulates the core software stack of an autonomous vehicle or mobile robot:
perceiving an environment, planning a path through it, and learning from
experience to improve over time.

================================================================================
HARDWARE INTEGRATION
================================================================================

PHASE 11 — Hardware Integration (Raspberry Pi + Camera)
  [ ] Overhead camera setup
  [ ] Obstacle detection       — OpenCV colour detection or ArUco markers
  [ ] Live grid update         — detected obstacles feed into env.setObstacle()
  [ ] Real-time demo           — agent navigates grid mirroring physical world
  [ ] Visualisation overlay    — agent position + Q-values on camera feed

PHASE 12 — OpenStreetMap Integration
  [ ] OSM graph import         — libosmium: intersections as nodes, roads as edges
  [ ] Coordinate projection    — lat/lon to x/y
  [ ] Dynamic edge weights     — real-time traffic data
  [ ] Run A*/Dijkstra on map

================================================================================
