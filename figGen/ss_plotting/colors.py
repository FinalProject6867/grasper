#!/usr/bin/env python
import logging
logger = logging.getLogger('ss_plotting')

SHADE = 'shade'
STANDARD = 'standard'
EMPHASIS = 'emphasis'

color_library = {}

BLUE = 'blue'
color_library[BLUE] = {}
color_library[BLUE][SHADE] = '#deebf7'
color_library[BLUE][STANDARD] = '#9ecae1'
color_library[BLUE][EMPHASIS]='#3182bd'

# Orange
ORANGE = 'orange'
color_library[ORANGE]={}
color_library[ORANGE][SHADE]='#fee6ce'
color_library[ORANGE][STANDARD]='#fdc086'
color_library[ORANGE][EMPHASIS]='#e6550d'

# Purple
PURPLE = 'purple'
color_library[PURPLE]={}
color_library[PURPLE][SHADE]='#efedf5'
color_library[PURPLE][STANDARD]='#beaed4'
color_library[PURPLE][EMPHASIS]='#756bb1'

# Green
GREEN = 'green'
color_library[GREEN]={}
color_library[GREEN][SHADE]='#e5f5e0'
color_library[GREEN][STANDARD]='#a1d99b'
color_library[GREEN][EMPHASIS]='#31a354'

# GREY
GREY = 'grey'
color_library[GREY]={}
color_library[GREY][SHADE]="#f0f0f0"
color_library[GREY][STANDARD]="#bdbdbd"
color_library[GREY][EMPHASIS]="#636363"

# RED
RED = 'red'
color_library[RED]={}
color_library[RED][SHADE]="#fee0d2"
color_library[RED][STANDARD]="#fc9272"
color_library[RED][EMPHASIS]="#de2d26"

# PINK
PINK = 'pink'
color_library[PINK]={}
color_library[PINK][SHADE]="#fde0dd"
color_library[PINK][STANDARD]="#fa9fb5"
color_library[PINK][EMPHASIS]="#c51b8a"


def get_plot_color(color=None, emphasis=False):
    """
    Returns color data for the requested color.
    @param color A string describing the general color (i.e. 'blue')
    @param emphasis If true, return a bold version of the color, 
        otherwise return a standard
    @return A hex code for the color. This can be passed directly
        to most matplotlib commands.
    """
    if not isinstance(color, basestring):
        return [ float(c) / 255. if isinstance(c, int) else c for c in color ]

    try:
        color_data = color_library[color]
    except KeyError, e:
        default_color = GREY
        color_data = color_library[default_color]
        logger.warn('Failed to find color %s in library. Returning %s.' % (color, default_color))

    if emphasis:
        return color_data[EMPHASIS]
    else:
        return color_data[STANDARD]
        

def get_shade_color(color=None):
    """
    Returns a color code for a light shade of the requested color
    @param color A string describing the general color (i.e. 'blue')
    @return A hex code for the color. This can be passed directly
        to most matplotlib commands.
    """

    try:
        color_data = color_library[color]
    except KeyError, e:
        default_color = GREY
        color_data = color_library[default_color]
        logger.warn('Failed to find color %s in library. Returning %s.' % (color, default_color))

    return color_data[SHADE]
