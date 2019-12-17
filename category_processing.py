import re
from googletrans import Translator

def translate_string_list(L, dest="en"):
    """
    Translates a list of strings to targeted language
    
    Parameters:
        L (list) : list containing strings
        dest (string) : targeted language code
       
    Returns:
        list containing translated strings
    """
    translator = Translator()
    return list(map(lambda x:x.text, translator.translate(L, dest='en')))


def process_categories(category_names):
    replacement_dict = {"PC - Headsets / Headphones" : "Accessories - PC",
                       "Programs - Educational (figure)" : "Program - Educational",
                       "Games - Accessories for games" : "Accessories - Games",
                       "Gifts - Games (compact)" : "Gifts - Gift games"}
    for cat in category_names:
        if cat in replacement_dict:
            pass
        elif "Accessories" in cat:
            pass
        elif "Game consoles" in cat:
            pass
        elif re.search(r'^Games\s(.*)\s-.*$', cat):
            sub_cat = re.match(r'^Games\s(.*)\s-.*$', cat).groups()[0]
            replacement_dict.update({cat : "Games - "+sub_cat})
        elif "Payment card" in cat:
            replacement_dict.update({cat: "Payment card - Payment card"})
        elif "Games" in cat:
            pass
        elif "Movie" in cat or "Cinema" in cat:
            if "Blu-Ray" in cat:
                replacement_dict.update({cat: "Movies - Bluray"})
            else:
                replacement_dict.update({cat: "Movies - DVD"})
        elif "Books" in cat:
            replacement_dict.update({cat: cat.split(" (")[0].replace(" 1C","")})
        elif "CD" in cat:
            replacement_dict.update({cat: "Music - CD"})
        elif "Gift" in cat:
            replacement_dict.update({cat: cat.split(" (")[0]})
        elif "home" in cat.lower():
            replacement_dict.update({cat: "Program - Home and Office"})
        elif "Program" in cat:
            pass
        else:
            replacement_dict.update({cat: "Others - Others"})

    return [replacement_dict.get(x, x) for x in category_names]