from dataclasses import fields


class BaseArgs:
    def __iter__(self):
        """Enable unpacking with * operator by yielding field values"""
        for field in fields(self):
            yield getattr(self, field.name)

    def keys(self):
        return [field.name for field in fields(self)]

    def __getitem__(self, key):
        return getattr(self, key)
