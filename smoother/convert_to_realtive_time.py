def convert_to_relative_time(datetimes, startdatetime):
    # Relative time in minutes
    t = (datetimes - startdatetime) / 60000000000

    return t

