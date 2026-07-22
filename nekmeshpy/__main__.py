"""``python -m nekmeshpy`` -- run the surface mesher with the default config."""

import logging

from .algorithms.bifurcation import BifurcationMesher
from .config import Config


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    BifurcationMesher(Config()).run()


if __name__ == "__main__":
    main()
