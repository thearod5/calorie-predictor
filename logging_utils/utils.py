from constants import *
import json

def format_header(header, chars=HEADING_CHARS):
    return f'{chars}{header}{chars}'


def format_subheader(subheader):
    return format_header(subheader, SUBHEADING_CHARS)


def get_section_break(header=None):
    section_break = HEADING_CHARS * 2
    if header:
        section_break += LOG_CHAR * len(header)
    return section_break


def format_name_val_info(name, val):
    return f'{name}: {val}'


def format_to_json(obj):
    return json.dumps(obj, indent=4, sort_keys=True)


def format_eval_results(obj, name):
    return format_subheader(name) + '\n' + format_to_json(obj)
