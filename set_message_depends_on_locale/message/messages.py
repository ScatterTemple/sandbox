

import os
import locale
from dataclasses import dataclass

loc, encoding = locale.getlocale()


if __name__ == '__main__':
    # set target to create .pot
    from gettext import gettext as _

else:
    # get translation
    if 'japanese' in loc.lower():
        from babel.support import Translations
        translations = Translations.load(
            os.path.join(os.path.dirname(__file__), 'locales'),
            locales='ja'
        )
        _ = translations.gettext

    else:
        def _(x): return x


@dataclass
class Message:
    HELLO: str = _('hello!')
