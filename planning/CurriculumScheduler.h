#ifndef CURRICULUMSCHEDULER_H
#define CURRICULUMSCHEDULER_H

#include <vector>

class CurriculumScheduler {
public:
    struct Stage {
        int startEpisode;
        int endEpisode;
        double loopDensity;
        int stageIndex;
    };

    // Equal stage lengths — divides totalEpisodes evenly across numStages.
    CurriculumScheduler(int totalEpisodes, int numStages = 4,
                        double startDensity = 0.5, double endDensity = 0.1);

    // Explicit stage lengths — each entry is the episode count for that stage.
    // Density interpolates from startDensity to endDensity across stages.
    CurriculumScheduler(const std::vector<int>& stageLengths,
                        double startDensity = 0.5, double endDensity = 0.1);

    double getDensityForEpisode(int episode) const;
    const Stage& getStageForEpisode(int episode) const;
    bool isStageTransition(int episode) const;
    int getNumStages() const;
    const std::vector<Stage>& getStages() const;
    void printSchedule() const;

private:
    std::vector<Stage> stages_;

    void buildFromLengths(const std::vector<int>& stageLengths,
                          double startDensity, double endDensity);
};

#endif  // CURRICULUMSCHEDULER_H
