// core/RunProfile.h
// Shared run-length settings for training demos and the interactive menu.
#ifndef RUN_PROFILE_H
#define RUN_PROFILE_H

// FULL  — curriculum training (15k episodes total across stages).
// QUICK — shortened runs for menu iteration and smoke-style demos.
enum class RunProfile
{
    FULL,
    QUICK
};

// Episode counts per curriculum stage for each profile.
inline const int* curriculumStageLengths(RunProfile profile)
{
    static const int fullStages[]  = { 2500, 2500, 5000, 5000 };
    static const int quickStages[] = {  200,  200,  300,  300 };
    return (profile == RunProfile::FULL) ? fullStages : quickStages;
}

inline int curriculumStageCount()
{
    return 4;
}

inline int totalCurriculumEpisodes(RunProfile profile)
{
    int total = 0;
    for (int stageIndex = 0; stageIndex < curriculumStageCount(); ++stageIndex)
        total += curriculumStageLengths(profile)[stageIndex];
    return total;
}

inline int trainingLogInterval(RunProfile profile)
{
    return (profile == RunProfile::FULL) ? 1000 : 100;
}

#endif
