// visualization/Visualizer.h
// Terminal ASCII renderer for grids, paths, and algorithm comparisons.
// All methods are static — no mutable state, no instantiation needed.
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
    // Legend: '.' empty  '#' obstacle  'S' start  'G' goal  '*' path  'o' visited
    static void displayGrid(const Environment& env);

    // Print the grid with 'path' overlaid using '*', preserving START/GOAL chars.
    // 'algorithmName' is printed as a header above the grid.
    static void displayPath(const Environment&           env,
                             const std::vector<Position>& path,
                             const std::string&           algorithmName);

    // ---- Statistics ---------------------------------------------------------

    // One-line summary: algorithm name | path length | cost | time (ms).
    static void displayStats(const std::string&           algorithmName,
                              const std::vector<Position>& path,
                              double                       elapsedMs,
                              double                       pathCost);

    // Formatted table comparing all algorithm results.
    static void displaySummaryTable(const std::vector<PathResult>& results);

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

#endif  // VISUALIZER_H
