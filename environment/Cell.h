// environment/Cell.h
// Represents a single tile in the grid; wraps a CellType enum.
#ifndef CELL_H
#define CELL_H
#include "../core/Types.h"
#include <string>

class Cell
{
private:
    CellType type;

public:
    Cell() : type(CellType::EMPTY) {}
    Cell(CellType t) : type(t) {}

    bool isObstacle() const;
    bool isStart() const;
    bool isEmpty() const;
    bool isGoal() const;
    std::string toChar() const;
    CellType getCellType() const;
    void setCellType(CellType t);
    ~Cell();
};

#endif