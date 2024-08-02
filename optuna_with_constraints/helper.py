from functools import partial


class NonOverwritablePartial(partial):
    def __call__(self, /, *args, **keywords):
        stored_kwargs = self.keywords
        keywords.update(stored_kwargs)
        return self.func(*self.args, *args, **keywords)
