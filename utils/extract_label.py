def extract_label(filename):
    """ Docstring """

    if len(filename)>= 44:
        label = int(filename[-16:-13])
    else:
        label = int(filename[-10:-7])

    return label

