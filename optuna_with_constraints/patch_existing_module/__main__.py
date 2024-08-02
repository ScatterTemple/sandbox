# do_mock.py

# ===== mock =====
from functools import partial


class NonOverwritablePartial(partial):
    def __call__(self, /, *args, **keywords):
        stored_kwargs = self.keywords
        keywords.update(stored_kwargs)
        return self.func(*self.args, *args, **keywords)


import existing_module


def print_string_length(string):
    print(string + f" {len(string)}")


# existing_module.print_string = print_string_length
forced_kwargs = {'string': 'Good bye'}
print_good_bye = NonOverwritablePartial(existing_module.print_string, **forced_kwargs, )
existing_module.print_string = print_good_bye


# ===== import =====
from existing_module import Sample



if __name__ == "__main__":
    s = Sample("Hello")
    s.print_me()
