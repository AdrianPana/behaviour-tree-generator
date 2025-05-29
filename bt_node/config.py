from enum import Enum

rooms = {
    0: (0.0, 0.0),
    1: (7.0, 7.0),
    2: (-1.0, 7.5),
    3: (-6.0, 4.0),
    4: (-6.0, -1.5),
    5: (-6.0, -6.5),
    6: (0.0, -9.0)
}

class Room(Enum):
    HALL = 0
    OFFICE = 1
    DINING_ROOM = 2
    CLASSROOM1 = 3
    CLASSROOM2 = 4
    BATHROOM = 5
    LOUNGE = 6