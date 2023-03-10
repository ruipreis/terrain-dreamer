import ee


def initialize_google_earth():
    # If the auth mode isnt provided, browser mode is used by default.
    # then rune initialzie without arguments.

    # If initialize fails, run the following commented line without script,
    # to setup the authentication.
    # ee.Authenticate()

    ee.Initialize()
