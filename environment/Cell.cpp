// environment/Cell.cpp
#include "Cell.h"
#include <iostream>

using namespace std;
bool Cell::isObstacle() const
{
    return type == CellType::OBSTACLE;
}

bool Cell::isStart() const
{
    return type == CellType::START;
}

bool Cell::isEmpty() const
{
    return type == CellType::EMPTY;
}

bool Cell::isGoal() const
{
    return type == CellType::GOAL;
}

string Cell::toChar() const
{
    switch(type)
    {
        case CellType::EMPTY:    return " ";
        case CellType::OBSTACLE: return "\u2588";
        case CellType::START:    return "S";
        case CellType::GOAL:     return "G";
        case CellType::PATH:     return "*";
        case CellType::VISITED:  return "\xC2\xB7";
        default:                 return " ";
    }
}

CellType Cell::getCellType() const
{
    return type;
}

void Cell::setCellType(CellType t)
{
    type = t;
}

Cell::~Cell() = default;