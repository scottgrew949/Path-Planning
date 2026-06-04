// visualization/Visualizer.h
#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <vector>
#include <string>
#include <map>
#include "../core/Position.h"
#include "../core/Types.h"
#include "../environment/Environment.h"

class Visualizer
{
public:
    Visualizer()  = delete;
    ~Visualizer() = delete;

    // ---- Grid rendering -----------------------------------------------------

    // Print the raw grid using Cell::toChar() for each cell.
    static void displayGrid(const Environment& env);

    // Print the grid with 'path' using '*'
    static void displayPath(const Environment&           env,
                             const std::vector<Position>& path,
                             const std::string&           algorithmName);

    // ---- Statistics ---------------------------------------------------------

    // One-line: algorithm name | path length | cost | time (ms).
    static void displayStats(const std::string&           algorithmName,
                              const std::vector<Position>& path,
                              double                       elapsedMs,
                              double                       pathCost);

    static void displaySummaryTable(const std::vector<PathResult>& results);

    // Final unified table.
    // rlResults: list of {name, pathLength} pairs (-1 = no path found).
    static void displayFinalSummary(const std::vector<PathResult>&                  classical,
                                    const std::vector<std::pair<std::string, int>>& rlResults);

    // ---- Section formatting -------------------------------------------------

    // Print a titled divider spanning 60 chars: "====== Title ======"
    static void printSection(const std::string& title);

private:
    // Build a rows-length vector of strings with 'path' positions marked '*'.
    static std::vector<std::string> renderGrid(
        const Environment&           env,
        const std::vector<Position>& path
    );

    static void        printHorizontalBorder(int cols);
    static std::string padTo(const std::string& str, int width);
};

#endif 
