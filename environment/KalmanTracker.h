// environment/KalmanTracker.h
//
// CONCEPT — Kalman filter for moving obstacle tracking:
//   Maintains a position + velocity estimate per tracked obstacle.
//   Each tick: predict() projects state forward; update() corrects with sensor data.
//   getPredictedPosition() extrapolates the current estimate dt seconds ahead.
//
//   Self-driving analog: predicting where a pedestrian will be in 0.5 seconds
//   so the planner can route around them before they block the path.
//   Standard in production: every ADAS system uses KF or EKF for object tracking.
//
//   Integration seam: getPredictedObstaclePositions() returns predicted grid cells
//   for all tracked obstacles, ready to feed into AbstractMap::updateWithPredictions()
//   once the Kalman->planner integration phase is implemented.

#ifndef KALMAN_TRACKER_H
#define KALMAN_TRACKER_H

#include "../core/Position.h"
#include <unordered_map>
#include <vector>

class KalmanTracker
{
public:
    // processNoise:     diagonal of Q — how much the motion model can drift per tick.
    // measurementNoise: diagonal of R — sensor position uncertainty in grid cells.
    // Throws std::invalid_argument if either is <= 0.
    explicit KalmanTracker(double processNoise = 1.0, double measurementNoise = 2.0);

    // Register a new tracked obstacle at the given grid position with zero initial velocity.
    // If id already exists, reinitializes that obstacle.
    void initObstacle(int obstacleId, double startX, double startY);

    // Advance all tracked obstacles forward by dt ticks using the motion model.
    // Call once per simulation tick before processing measurements.
    void predict(double dt);

    // Incorporate a position measurement for one obstacle.
    // No-op if obstacleId has not been initialized.
    void update(int obstacleId, double measuredX, double measuredY);

    // Remove a tracked obstacle by id.
    void removeObstacle(int obstacleId);

    // True if the given obstacle id has been initialized and not removed.
    bool hasObstacle(int obstacleId) const;

    // Current estimated position for one obstacle.
    // Returns Position(-1, -1) if id not found.
    Position getEstimatedPosition(int obstacleId) const;

    // Predicted position dt ticks in the future for one obstacle (no state mutation).
    // Returns Position(-1, -1) if id not found.
    Position getPredictedPosition(int obstacleId, double dt) const;

    // Returns predicted grid positions for ALL tracked obstacles at horizonTicks ahead.
    // Used as the integration seam for AbstractMap::updateWithPredictions().
    std::vector<Position> getPredictedObstaclePositions(double horizonTicks) const;

    int getTrackedObstacleCount() const;

private:
    struct ObstacleState
    {
        double stateVec[4];       // [px, py, vx, vy]
        double covariance[4][4];
        bool   initialized;
    };

    double processNoise_;
    double measurementNoise_;
    std::unordered_map<int, ObstacleState> trackedObstacles_;

    void predictSingle(ObstacleState& obstacleState, double dt);
    void updateSingle(ObstacleState& obstacleState, double measuredX, double measuredY);

    // Matrix helpers — all operate on raw double arrays for fixed-size math.
    static void matMul4x4(const double A[4][4], const double B[4][4], double result[4][4]);
    static void matTranspose4x4(const double A[4][4], double result[4][4]);
    static void matAdd4x4(const double A[4][4], const double B[4][4], double result[4][4]);
    static void invert2x2(const double A[2][2], double result[2][2]);
    // Computes A(4x4) * B(4x2) -> result(4x2). B stored as double[4][2].
    static void matMul4x4_times_4x2(const double A[4][4], const double B[4][2], double result[4][2]);
    // Computes A(4x2) * B(2x2) -> result(4x2).
    static void matMul4x2_times_2x2(const double A[4][2], const double B[2][2], double result[4][2]);
};

#endif  // KALMAN_TRACKER_H
