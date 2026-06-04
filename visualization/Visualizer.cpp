// visualization/Visualizer.cpp
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
    vector<string> rows = renderGrid(env, {});

    cout << "Grid (" << env.getHeight() << " x " << env.getWidth() << ")\n";
    printHorizontalBorder(env.getWidth());
    for (const string& r : rows)
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
        vector<string> rows = renderGrid(env, path);
        printHorizontalBorder(env.getWidth());

    for (const string& r : rows)
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
    for (const PathResult& r : results)
    {
        int    length  = r.path.size();
        double cost    = r.pathCost;
        double elapsed = r.elapsedMs;
        cout << padTo(r.algorithmName, 14) << "| " << setw(12) << length << "| " 
            << setw(7) << cost << "| " << setw(10) << elapsed << "| " << setw(14) << r.nodesExplored << '\n';
    }
    cout << "--------------|-------------|--------|-----------|---------------\n";
}

void Visualizer::displayFinalSummary(const std::vector<PathResult>&                  classical,
                                      const std::vector<std::pair<std::string, int>>& rlResults)
{
    const int W_NAME  = 20;
    const int W_CAT   = 13;
    const int W_PATH  =  7;
    const int W_COST  =  9;
    const int W_TIME  =  9;
    const int W_NODES = 10;

    auto sep = [&]() {
        cout << string(W_NAME,'-') << "+"
             << string(W_CAT, '-') << "+"
             << string(W_PATH,'-') << "+"
             << string(W_COST,'-') << "+"
             << string(W_TIME,'-') << "+"
             << string(W_NODES,'-') << "\n";
    };

    cout << "\n";
    cout << padTo("Algorithm",           W_NAME) << "|"
         << padTo(" Category",           W_CAT)  << "|"
         << padTo(" Path",               W_PATH)  << "|"
         << padTo(" Cost",               W_COST)  << "|"
         << padTo(" Time ms",            W_TIME)  << "|"
         << padTo(" Nodes",              W_NODES) << "\n";
    sep();

    for (const PathResult& r : classical)
    {
        string name   = " " + r.algorithmName;
        string path   = r.path.empty() ? "  N/A" : "  " + to_string((int)r.path.size());
        string cost   = r.path.empty() ? "  N/A" : "  " + to_string((int)r.pathCost);
        string timeMs = "  " + to_string(r.elapsedMs).substr(0, 5);
        string nodes  = "  " + to_string(r.nodesExplored);

        cout << padTo(name,    W_NAME) << "|"
             << padTo(" Classical",  W_CAT)  << "|"
             << padTo(path,   W_PATH)  << "|"
             << padTo(cost,   W_COST)  << "|"
             << padTo(timeMs, W_TIME)  << "|"
             << padTo(nodes,  W_NODES) << "\n";
    }

    sep();

    for (const auto& [name, pathLen] : rlResults)
    {
        string pathStr = (pathLen < 0) ? "  N/A" : "  " + to_string(pathLen);
        cout << padTo(" " + name,   W_NAME) << "|"
             << padTo(" Tabular RL", W_CAT)  << "|"
             << padTo(pathStr,       W_PATH)  << "|"
             << padTo("  N/A",       W_COST)  << "|"
             << padTo("  N/A",       W_TIME)  << "|"
             << padTo("  N/A",       W_NODES) << "\n";
    }

    sep();
}

// ---- Section formatting -----------------------------------------------------

void Visualizer::printSection(const string& title)
{
    int pad = max(0, (60 - (int)title.size() - 2) / 2);
    string line = string(pad, '=') + ' ' + title + ' ' + string(pad, '=');
    cout << '\n' << line << '\n';
}
