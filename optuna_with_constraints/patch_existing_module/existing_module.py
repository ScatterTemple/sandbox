# module.py

from existing_module_core import print_string


class Sample:

    def __init__(self, string):
        self.string = string

    def print_me(self):
        print_string(string=self.string)  # want to change this function
