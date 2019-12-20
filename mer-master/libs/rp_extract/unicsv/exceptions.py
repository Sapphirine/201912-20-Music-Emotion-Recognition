class CustomException(Exception):
    """
    A base exception that handles pretty-printing errors for command-line utils.
    """
    def __init__(self, msg):
        self.msg = msg

    def __unicode__(self):
        return self.msg

    def __str__(self):
        return self.msg


class FieldSizeLimitError(CustomException):
    """
    Exception raised when a field in the CSV file exceeds the default max
    or one provided by the user.
    """
    def __init__(self, limit):
        self.msg = (
            'CSV contains fields longer than maximum length of %i characters. '
            'Try raising the maximum with the --maxfieldsize flag.' % limit)
