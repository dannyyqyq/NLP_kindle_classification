import sys


def error_message_details(error, error_details: sys) -> str:
    """
    This function generates a detailed error message by extracting information
    from the current exception, including the filename, line number,
    and error message.

    Args:
        error: The exception object.
        error_details: The sys module, which provides information about the current exception.

    Returns:
        A string containing the detailed error message.

    Example:
        Error occurred in python script [filename.py] line number [line number] error message [error message]
    """
    (
        _,
        _,
        exc_tb,
    ) = (
        error_details.exc_info()
    )  # This retrieves the traceback of the current exception.
    file_name = (
        exc_tb.tb_frame.f_code.co_filename
    )  # Extract the filename where the error occurred.

    # Generate a detailed error message
    error_message = "Error occurred in python script [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, errror_details: sys):
        super().__init__(error_message)

        # Generate a detailed error message
        self.error_message = error_message_details(error_message, errror_details)

    def __str__(self):
        return self.error_message


# For testing purposes
# if __name__ == "__main__":
#     try:
#         # Intentional division to demonstrate error handling
#         result = 1 / 0  # Will throw ZeroDivisionError

#     except Exception as e:
#         # Log the error
#         logging.info("Division by zero")
#         # Raise the custom exception with detailed error info
#         raise CustomException(e, sys)
