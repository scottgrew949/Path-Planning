// tests/SmokeTests.h
// Fast regression checks for core planning, probability, and environment code.
#ifndef SMOKE_TESTS_H
#define SMOKE_TESTS_H

struct SmokeTestSummary
{
    int passedCount;
    int failedCount;

    bool allPassed() const { return failedCount == 0; }
};

class SmokeTests
{
public:
    SmokeTests()  = delete;
    ~SmokeTests() = delete;

    static SmokeTestSummary runAll();
};

#endif
