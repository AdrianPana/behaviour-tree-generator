from enum import Enum

class Config:
    rooms = {
        0: (0.0, 0.0),
        1: (7.0, 7.0),
        2: (7.5, -1.0),
        3: (4.0, -6.0),
        4: (-1.5, -6.0),
        5: (-6.5, -6.0),
        6: (-9.0, 0.0)
    }

    class Room(Enum):
        HALL = 0
        OFFICE = 1
        DINING_ROOM = 2
        CLASSROOM1 = 3
        CLASSROOM2 = 4
        BATHROOM = 5
        LOUNGE = 6

    current_room = Room.HALL

    current_target = "Alice"

    people_room_memory = {
        "Alice": Room.OFFICE,
        # "Bob": Room.DINING_ROOM
        "Bob": None
    }

    # this is the universal truth, does not change
    people_rooms = {
        Room.OFFICE: "Alice",
        Room.DINING_ROOM: "Bob"
    }