#include "CurriculumScheduler.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

void CurriculumScheduler::buildFromLengths(const std::vector<int>& stageLengths,
                                            double startDensity, double endDensity)
{
    int numStages = static_cast<int>(stageLengths.size());
    stages_.reserve(numStages);
    int cursor = 0;
    for (int i = 0; i < numStages; ++i)
    {
        double t       = (numStages == 1) ? 0.0
                                          : static_cast<double>(i) / (numStages - 1);
        double density = startDensity + t * (endDensity - startDensity);
        Stage s;
        s.startEpisode = cursor;
        s.endEpisode   = cursor + stageLengths[i] - 1;
        s.loopDensity  = density;
        s.stageIndex   = i;
        stages_.push_back(s);
        cursor += stageLengths[i];
    }
}

CurriculumScheduler::CurriculumScheduler(int totalEpisodes, int numStages,
                                         double startDensity, double endDensity)
{
    if (numStages < 1) throw std::invalid_argument("numStages must be >= 1");

    int baseLen   = totalEpisodes / numStages;
    int remainder = totalEpisodes % numStages;

    std::vector<int> stageLengths(numStages);
    for (int i = 0; i < numStages; ++i)
        stageLengths[i] = baseLen + (i < remainder ? 1 : 0);

    buildFromLengths(stageLengths, startDensity, endDensity);
}

CurriculumScheduler::CurriculumScheduler(const std::vector<int>& stageLengths,
                                         double startDensity, double endDensity)
{
    if (stageLengths.empty()) throw std::invalid_argument("stageLengths must not be empty");
    for (int len : stageLengths)
        if (len < 1) throw std::invalid_argument("each stage length must be >= 1");

    buildFromLengths(stageLengths, startDensity, endDensity);
}

double CurriculumScheduler::getDensityForEpisode(int episode) const
{
    return getStageForEpisode(episode).loopDensity;
}

const CurriculumScheduler::Stage& CurriculumScheduler::getStageForEpisode(int episode) const
{
    auto it = std::find_if(stages_.begin(), stages_.end(),
        [episode](const Stage& s) {
            return episode >= s.startEpisode && episode <= s.endEpisode;
        });
    return (it != stages_.end()) ? *it : stages_.back();
}

bool CurriculumScheduler::isStageTransition(int episode) const
{
    return std::any_of(stages_.begin(), stages_.end(),
        [episode](const Stage& s) { return episode == s.startEpisode; });
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
                  << "  density = " << std::fixed << std::setprecision(2) << s.loopDensity;
        if (s.stageIndex == 0)
            std::cout << " (easy)";
        else if (s.stageIndex == static_cast<int>(stages_.size()) - 1)
            std::cout << " (hard)";
        std::cout << "\n";
    }
}
