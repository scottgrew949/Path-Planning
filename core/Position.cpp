// core/Position.cpp
#include "Position.h"
#include <functional>  // std::hash

// ---- Position ---------------------------------------------------------------
Position::Position(int xPos, int yPos) : x(xPos), y(yPos) {}

bool Position::operator==(const Position& other) const
{
    return (this->x == other.x) && (this->y == other.y); 
}

bool Position::operator!=(const Position& other) const
{
    return (this->x != other.x) || (this->y != other.y); 
}

bool Position::operator<(const Position& other) const
{
    if(this->y != other.y)
        return this->y < other.y;

    return this->x < other.x;
}

void Position::print() const
{
    std::cout << "({ " << this->x << " }, {" << this->y << "})" << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Position& p)
{
    os << "(" << p.x << ", " << p.y << ")";
    return os;
}

// ---- PositionHash -----------------------------------------------------------
std::size_t PositionHash::operator()(const Position& position) const noexcept
{
    // XOR-shift combine: shift y's hash left 16 bits before XOR to reduce
    // collisions between positions that share one coordinate.
    return std::hash<int>{}(position.x) ^ (std::hash<int>{}(position.y) << 16);
}
