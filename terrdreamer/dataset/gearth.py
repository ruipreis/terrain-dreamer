import ee
import json


def initialize_google_earth():
    # If the auth mode isnt provided, browser mode is used by default.
    # then rune initialzie without arguments.
    ee.Authenticate()
    ee.Initialize()


if __name__ == "__main__":
    initialize_google_earth()
