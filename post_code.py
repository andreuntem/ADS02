import re
import pandas as pd
import numpy as np


def clean_spaces_split(text):
    # Remove commas, point, \n
    pattern = ','
    text = re.sub(pattern, '', text)
    pattern = r'\s+'
    text = re.sub(pattern, ' ', text).split()
    return text


def define_parts(text, pattern):
    match = lambda x: re.findall(pattern, x)
    matches = list(map(match, text))
    flags = [1 if m else 0 for m in matches]
    return np.array(flags)


def generate_flags(text):
    '''
    Postcode Standards
    X9 9XX
    X99 9XX
    XX9 9XXX
    XX99 9XX
    X9X 9XX
    XX9X 9XX
    '''
    _x99 = '^[A-Z][0-9]?[0-9]$'
    _xx99 = '^[A-Z]{2}[0-9]?[0-9]$'
    _xx9x = '^[A-Z][A-Z]?[0-9][A-Z]$'
    _9xx = '^[0-9][A-Z]{2}$'
    _x999xx = '^[A-Z][0-9]?[0-9]{2}[A-Z]{2}$'
    _xx999xx = '^[A-Z]{2}[0-9]?[0-9]{2}[A-Z]{2}$'
    _xx9x9xx = '^[A-Z]{2}[0-9][A-Z][0-9][A-Z]{2}$'

    # First part:
    case = 1
    pattern_1 = _x99 + '|' + _xx99 + '|' + _xx9x
    flags_1 = define_parts(text, pattern_1)*case

    # Second part:
    case = 2
    pattern_2 = _9xx
    flags_2 = define_parts(text, pattern_2)*case

    # Full postcode
    case = 3
    pattern_3 = _x999xx + '|' + _xx999xx + '|' + _xx9x9xx
    flags_3 = define_parts(text, pattern_3)*case

    flags = flags_1 + flags_2 + flags_3
    return flags


def compile_postcodes(list_strings, flags):
    post_codes = []
    for i, _ in enumerate(list_strings):
        if i > 0:
            code = []
            if flags[i-1] == 1 and flags[i] == 2:
                code = list_strings[i-1] + ' ' + list_strings[i]
            elif flags[i] == 3:
                code = list_strings[i][:-3] + '' + list_strings[i][-3:]
            if code:
                post_codes.append(code)
    print(post_codes)
    post_codes = ' | '.join(post_codes)
    return post_codes


def return_post_codes(df):
    
    # Clean spaces
    df['text_clean'] = df['text'].apply(clean_spaces_split)

    # Calculate flags:
    df['flags'] = df['text_clean'].apply(generate_flags)

    # Combine parts:
    df['postcodes'] = df.apply(lambda x: compile_postcodes(x['text_clean'], x['flags']), axis=1)

    return df[['postcodes']]


if __name__ == '__main__':
    df = pd.DataFrame({'text': {0: '108 Brick Kiln Rd, North Walsham NR289XR, UK There is also a system known as CEDEX 24 Jubilee Cl, Cherry Willingham, Lincoln LN3 4LD, UK red at 10.6 µm; such lasers are regularly used in industry for cutting and welding. The efficiency of a CO2 laser is unusually high: over 30% Burwood Court, 38 Canonbie Rd, Forest Hill, London SE23 3AY, UK '}})
    print(return_post_codes(df))
