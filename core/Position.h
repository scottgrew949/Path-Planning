// core/Position.h
// Represents a 2D grid coordinate; designed for use in all STL containers.
#ifndef POSITION_H
#define POSITION_H

#include <iostream>

struct Position
{
    int x;   // column index (0 = left)
    int y;   // row    index (0 = top)

    explicit Position(int xPos = 0, int yPos = 0);

    bool operator==(const Position& other) const;
    bool operator!=(const Position& other) const;
    bool operator<(const Position& other)  const;  // lexicographic (y,x) for set/map

    void print() const;
    friend std::ostream& operator<<(std::ostream& os, const Position& p);
};

// ---- PositionHash -----------------------------------------------------------
// Injected into std:: via explicit template argument so that
//   unordered_map<Position, T, PositionHash>
//   unordered_set<Position,    PositionHash>
// both work without specialising std::hash (avoids UB for struct types).
struct PositionHash
{
    std::size_t operator()(const Position& position) const noexcept;
};

#endif
