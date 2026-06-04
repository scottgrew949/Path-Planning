// environment/KalmanTracker.cpp

#include "KalmanTracker.h"
#include <cmath>
#include <stdexcept>

using namespace std;

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

KalmanTracker::KalmanTracker(double processNoise, double measurementNoise)
    : processNoise_(processNoise)
    , measurementNoise_(measurementNoise)
{
    if (processNoise <= 0.0)
    {
        throw invalid_argument("KalmanTracker: processNoise must be > 0");
    }
    if (measurementNoise <= 0.0)
    {
        throw invalid_argument("KalmanTracker: measurementNoise must be > 0");
    }
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void KalmanTracker::initObstacle(int obstacleId, double startX, double startY)
{
    ObstacleState obstacleState;
    obstacleState.initialized = true;

    obstacleState.stateVec[0] = startX;
    obstacleState.stateVec[1] = startY;
    obstacleState.stateVec[2] = 0.0;   // initial velocity x
    obstacleState.stateVec[3] = 0.0;   // initial velocity y

    // High initial uncertainty — we have one position reading and no velocity history.
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            obstacleState.covariance[row][col] = (row == col) ? 10.0 : 0.0;
        }
    }

    trackedObstacles_[obstacleId] = obstacleState;
}

void KalmanTracker::predict(double dt)
{
    for (auto& [obstacleId, obstacleState] : trackedObstacles_)
    {
        predictSingle(obstacleState, dt);
    }
}

void KalmanTracker::update(int obstacleId, double measuredX, double measuredY)
{
    unordered_map<int, ObstacleState>::iterator it = trackedObstacles_.find(obstacleId);
    if (it == trackedObstacles_.end())
    {
        return;
    }
    updateSingle(it->second, measuredX, measuredY);
}

void KalmanTracker::removeObstacle(int obstacleId)
{
    trackedObstacles_.erase(obstacleId);
}

bool KalmanTracker::hasObstacle(int obstacleId) const
{
    return trackedObstacles_.count(obstacleId) > 0;
}

Position KalmanTracker::getEstimatedPosition(int obstacleId) const
{
    unordered_map<int, ObstacleState>::const_iterator it = trackedObstacles_.find(obstacleId);
    if (it == trackedObstacles_.end())
    {
        return Position(-1, -1);
    }
    const ObstacleState& obstacleState = it->second;
    int estimatedX = static_cast<int>(round(obstacleState.stateVec[0]));
    int estimatedY = static_cast<int>(round(obstacleState.stateVec[1]));
    return Position(estimatedX, estimatedY);
}

Position KalmanTracker::getPredictedPosition(int obstacleId, double dt) const
{
    unordered_map<int, ObstacleState>::const_iterator it = trackedObstacles_.find(obstacleId);
    if (it == trackedObstacles_.end())
    {
        return Position(-1, -1);
    }
    const ObstacleState& obstacleState = it->second;
    double predictedX = obstacleState.stateVec[0] + obstacleState.stateVec[2] * dt;
    double predictedY = obstacleState.stateVec[1] + obstacleState.stateVec[3] * dt;
    return Position(static_cast<int>(round(predictedX)), static_cast<int>(round(predictedY)));
}

vector<Position> KalmanTracker::getPredictedObstaclePositions(double horizonTicks) const
{
    vector<Position> predictions;
    predictions.reserve(trackedObstacles_.size());
    for (const auto& [obstacleId, obstacleState] : trackedObstacles_)
    {
        double predictedX = obstacleState.stateVec[0] + obstacleState.stateVec[2] * horizonTicks;
        double predictedY = obstacleState.stateVec[1] + obstacleState.stateVec[3] * horizonTicks;
        predictions.push_back(Position(static_cast<int>(round(predictedX)),
                                       static_cast<int>(round(predictedY))));
    }
    return predictions;
}

int KalmanTracker::getTrackedObstacleCount() const
{
    return static_cast<int>(trackedObstacles_.size());
}

// ---------------------------------------------------------------------------
// Private: predict one obstacle
// ---------------------------------------------------------------------------

void KalmanTracker::predictSingle(ObstacleState& obstacleState, double dt)
{
    // Apply constant-velocity model directly — F is sparse so explicit matmul is wasteful.
    obstacleState.stateVec[0] += obstacleState.stateVec[2] * dt;   // px += vx * dt
    obstacleState.stateVec[1] += obstacleState.stateVec[3] * dt;   // py += vy * dt
    // vx, vy unchanged under constant-velocity assumption

    // Build F for covariance propagation: F * P * F^T + Q
    double F[4][4] = {
        {1.0, 0.0,  dt, 0.0},
        {0.0, 1.0, 0.0,  dt},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };

    double FP[4][4];
    matMul4x4(F, obstacleState.covariance, FP);

    double Ft[4][4];
    matTranspose4x4(F, Ft);

    double FPFt[4][4];
    matMul4x4(FP, Ft, FPFt);

    // Add process noise Q = diag(qPos, qPos, qVel, qVel)
    double qPosition = processNoise_ * 0.25;
    double qVelocity = processNoise_;
    FPFt[0][0] += qPosition;
    FPFt[1][1] += qPosition;
    FPFt[2][2] += qVelocity;
    FPFt[3][3] += qVelocity;

    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            obstacleState.covariance[row][col] = FPFt[row][col];
        }
    }
}

// ---------------------------------------------------------------------------
// Private: update one obstacle with a position measurement
// ---------------------------------------------------------------------------

void KalmanTracker::updateSingle(ObstacleState& obstacleState, double measuredX, double measuredY)
{
    double innovation[2] = {
        measuredX - obstacleState.stateVec[0],
        measuredY - obstacleState.stateVec[1]
    };

    // S = H * P * H^T + R. H selects the first two rows of P, so:
    // H * P * H^T is the top-left 2x2 of P.
    double S[2][2];
    S[0][0] = obstacleState.covariance[0][0] + measurementNoise_;
    S[0][1] = obstacleState.covariance[0][1];
    S[1][0] = obstacleState.covariance[1][0];
    S[1][1] = obstacleState.covariance[1][1] + measurementNoise_;

    double S_inv[2][2];
    invert2x2(S, S_inv);

    // PH (4x2) = P * H^T. Since H^T selects columns 0 and 1 of P:
    double PH[4][2];
    for (int row = 0; row < 4; ++row)
    {
        PH[row][0] = obstacleState.covariance[row][0];
        PH[row][1] = obstacleState.covariance[row][1];
    }

    // K (4x2) = PH * S_inv
    double K[4][2];
    matMul4x2_times_2x2(PH, S_inv, K);

    // Update state: x = x + K * innovation
    for (int row = 0; row < 4; ++row)
    {
        obstacleState.stateVec[row] += K[row][0] * innovation[0] + K[row][1] * innovation[1];
    }

    // Update covariance: P = (I - K*H) * P
    // KH (4x4): KH[i][0]=K[i][0], KH[i][1]=K[i][1], KH[i][2..3]=0 (from H structure)
    double KH[4][4];
    for (int row = 0; row < 4; ++row)
    {
        KH[row][0] = K[row][0];
        KH[row][1] = K[row][1];
        KH[row][2] = 0.0;
        KH[row][3] = 0.0;
    }

    double I_minus_KH[4][4];
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            double identityEntry = (row == col) ? 1.0 : 0.0;
            I_minus_KH[row][col] = identityEntry - KH[row][col];
        }
    }

    double newCovariance[4][4];
    matMul4x4(I_minus_KH, obstacleState.covariance, newCovariance);

    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            obstacleState.covariance[row][col] = newCovariance[row][col];
        }
    }
}

// ---------------------------------------------------------------------------
// Private static: matrix helpers
// ---------------------------------------------------------------------------

void KalmanTracker::matMul4x4(const double A[4][4], const double B[4][4], double result[4][4])
{
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            double sum = 0.0;
            for (int inner = 0; inner < 4; ++inner)
            {
                sum += A[row][inner] * B[inner][col];
            }
            result[row][col] = sum;
        }
    }
}

void KalmanTracker::matTranspose4x4(const double A[4][4], double result[4][4])
{
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            result[col][row] = A[row][col];
        }
    }
}

void KalmanTracker::matAdd4x4(const double A[4][4], const double B[4][4], double result[4][4])
{
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            result[row][col] = A[row][col] + B[row][col];
        }
    }
}

void KalmanTracker::invert2x2(const double A[2][2], double result[2][2])
{
    double determinant = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    double inverseDeterminant = 1.0 / determinant;
    result[0][0] =  A[1][1] * inverseDeterminant;
    result[0][1] = -A[0][1] * inverseDeterminant;
    result[1][0] = -A[1][0] * inverseDeterminant;
    result[1][1] =  A[0][0] * inverseDeterminant;
}

void KalmanTracker::matMul4x4_times_4x2(const double A[4][4], const double B[4][2], double result[4][2])
{
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 2; ++col)
        {
            double sum = 0.0;
            for (int inner = 0; inner < 4; ++inner)
            {
                sum += A[row][inner] * B[inner][col];
            }
            result[row][col] = sum;
        }
    }
}

void KalmanTracker::matMul4x2_times_2x2(const double A[4][2], const double B[2][2], double result[4][2])
{
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 2; ++col)
        {
            result[row][col] = A[row][0] * B[0][col] + A[row][1] * B[1][col];
        }
    }
}
