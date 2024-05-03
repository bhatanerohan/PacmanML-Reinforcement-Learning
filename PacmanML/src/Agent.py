##################################################
## Agent object
##################################################
## Author: Khoa Nguyen
## Copyright: Copyright 2023
## License: GPL
##################################################

class Agent:
    def __init__(self, color, is_hostile, position, sprite):
        self._color = color
        self._is_hostile = is_hostile
        self._position = position
        self._has_moved = False
        self._sprite = sprite

    def get_sprite(self):
        return self._sprite

    def is_hostile(self):
        return self._is_hostile

    def get_color(self):
        return self._color

    def get_y(self):
        return self._position[0]

    def get_x(self):
        return self._position[1]

    def get_position(self):
        return self._position

    def set_position(self, y, x):
        self._position = (y, x)

    def has_moved(self):
        return self._has_moved

    def set_move(self):
        self._has_moved = not self._has_moved
