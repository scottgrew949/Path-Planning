#include "CurriculumScheduler.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>

CurriculumScheduler::CurriculumScheduler(int totalEpisodes, int numStages,
                                         double startDensity, double endDensity)
{
    if (numStages < 1) throw std::invalid_argument("numStages must be >= 1");

    int baseLen = totalEpisodes / numStages;
    int remainder = totalEpisodes % numStages;

    stages_.reserve(numStages);
    int cursor = 0;
    for (int i = 0; i < numStages; ++i) {
        double t = (numStages == 1) ? 0.0
                                    : static_cast<double>(i) / (numStages - 1);
        double density = startDensity + t * (endDensity - startDensity);

        int len = baseLen + (i < remainder ? 1 : 0);
        Stage s;
        s.startEpisode = cursor;
        s.endEpisode   = cursor + len - 1;
        s.loopDensity  = density;
        s.stageIndex   = i;
        stages_.push_back(s);
        cursor += len;
    }
}

double CurriculumScheduler::getDensityForEpisode(int episode) const
{
    return getStageForEpisode(episode).loopDensity;
}

const CurriculumScheduler::Stage& CurriculumScheduler::getStageForEpisode(int episode) const
{
    for (const Stage& s : stages_) {
        if (episode >= s.startEpisode && episode <= s.endEpisode)
            return s;
    }
    return stages_.back();
}

bool CurriculumScheduler::isStageTransition(int episode) const
{
    for (const Stage& s : stages_) {
        if (episode == s.startEpisode)
            return true;
    }
    return false;
}

int CurriculumScheduler::getNumStages() const
{
    return static_cast<int>(stages_.size());
}

const std::vector<CurriculumScheduler::Stage>& CurriculumScheduler::getStages() const
{
    return stages_;
}

void CurriculumScheduler::printSchedule() const
{
    for (const Stage& s : stages_) {
        std::cout << "Stage " << (s.stageIndex + 1) << ": episodes "
                  << std::setw(5) << s.startEpisode << "-"
                  << std::setw(5) << s.endEpisode
                  << "  density=" << std::fixed << std::setprecision(2) << s.loopDensity;
        if (s.stageIndex == 0)
            std::cout << " (easy)";
        else if (s.stageIndex == static_cast<int>(stages_.size()) - 1)
            std::cout << " (hard)";
        std::cout << "\n";
    }
}
