// visualization/Visualizer.cpp
#include "../core/Types.h"
#include "Visualizer.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>

using namespace std;

// ---- Private helpers --------------------------------------------------------

vector<string> Visualizer::renderGrid(const Environment& env,
                                       const vector<Position>& path)
{
    vector<string> rows;
    size_t env_height = static_cast<size_t>(env.getHeight());
    size_t env_width = static_cast<size_t>(env.getWidth());

    unordered_set<Position, PositionHash> pathSet(path.begin(), path.end());
    for(size_t y = 0; y < env_height; ++y)
    {
        string row;
        for(size_t x = 0; x < env_width; ++x)
        {
            CellType type = env.getGrid()[x][y].getCellType();
            if(pathSet.count(Position(x, y)) && type != CellType::START && type != CellType::GOAL)
            {
                row.append("*");
            }
            else
                row += env.getGrid()[x][y].toChar();
        }
        rows.push_back(row);
    }

    return rows;
}

void Visualizer::printHorizontalBorder(int cols)
{
    string hBorder(cols, '-');
    cout << "+" << hBorder << "+\n";
}

string Visualizer::padTo(const string& str, int width)
{
    if (str.size() >= static_cast<size_t>(width)) 
        return str.substr(0, width);
    else
        return str + string(width - str.size(), ' ');

}

// ---- Grid rendering ---------------------------------------------------------

void Visualizer::displayGrid(const Environment& env)
{
    auto rows = renderGrid(env, {});

    cout << "Grid (" << env.getHeight() << " x " << env.getWidth() << ")\n";
    printHorizontalBorder(env.getWidth());
    for(const auto& r : rows)
    {
        cout << '|' << r << "|\n";
    }
    printHorizontalBorder(env.getWidth());
    cout << "Start: S - Goal: G - Path: * - Visited: ·\n";
}

void Visualizer::displayPath(const Environment&      env,
                              const vector<Position>& path,
                              const string&           algorithmName)
{
    cout << algorithmName << " — path length: " << path.size() << '\n';

    if (path.empty())
        cout << "  [no path found]\n"; 
    else
    {
        auto rows = renderGrid(env, path);
        printHorizontalBorder(env.getWidth());

    for(const auto& r : rows)
    {
        cout << '|' << r << "|\n";
    }

    printHorizontalBorder(env.getWidth());
    }
}

// ---- Statistics -------------------------------------------------------------

void Visualizer::displayStats(const string&           algorithmName,
                               const vector<Position>& path,
                               double                  elapsedMs,
                               double                  pathCost)
{
    if (path.empty())
    {
        cout << "  " << left << setw(12) << algorithmName << "  NO PATH FOUND\n";
    }
    else
    {
        cout << "  " << left  << setw(12) << algorithmName
            << "  length: " << setw(4)  << path.size()
            << "  cost: "   << fixed << setprecision(2) << setw(7) << pathCost
            << "  time: "   << fixed << setprecision(3) << elapsedMs << " ms\n";
    }
}

void Visualizer::displaySummaryTable(const vector<PathResult>& results)
{
    cout << "Algorithm     | Path Length | Cost   | Time (ms) | Nodes Explored\n";
    cout << "--------------|-------------|--------|-----------|---------------\n";
    for (auto& r : results)
    {
        int    length  = r.path.size();
        double cost    = r.pathCost;
        double elapsed = r.elapsedMs;
        cout << padTo(r.algorithmName, 14) << "| " << setw(12) << length << "| " 
            << setw(7) << cost << "| " << setw(10) << elapsed << "| " << setw(14) << r.nodesExplored << '\n';
    }
    cout << "--------------|-------------|--------|-----------|---------------\n";
}

// ---- Section formatting -----------------------------------------------------

void Visualizer::printSection(const string& title)
{
    int pad = max(0, (60 - (int)title.size() - 2) / 2);
    string line = string(pad, '=') + ' ' + title + ' ' + string(pad, '=');
    cout << '\n' << line << '\n';
}
